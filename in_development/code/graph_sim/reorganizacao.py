import sys, os, pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); DAY=sys.argv[1]; S=pickle.load(open(f'{OUT}/solver_rev_{DAY}.pkl','rb')); T=S['T'].copy(); T['dur']=S['dur']; AL=S['AL']; rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
FACT=['tipo','hb','clima','pier']; folds=np.array_split(np.arange(len(T)),5)
def onehot(tr_,te_):
    A=[];B=[]
    for k in FACT:
        cats=pd.Categorical(tr_[k]); ix={c:i for i,c in enumerate(cats.categories)}
        def oh(df):
            ii=df[k].map(ix); m=ii.notna().to_numpy(); r=np.where(m)[0]; return sparse.csr_matrix((np.ones(m.sum()),(r,ii[m].astype(int).to_numpy())),shape=(len(df),len(ix)))
        A.append(oh(tr_)); B.append(oh(te_))
    A.append(sparse.csr_matrix(tr_[['chain']].to_numpy(float))); B.append(sparse.csr_matrix(te_[['chain']].to_numpy(float))); return sparse.hstack(A).tocsr(),sparse.hstack(B).tocsr()
T['res']=T.y-T.base-T.dur; pred=np.zeros(len(T))
for f in folds:
    te=np.zeros(len(T),bool); te[f]=True; Xtr,Xte=onehot(T[~te],T[te]); pred[te]=Ridge(alpha=AL).fit(Xtr,T.res[~te]).predict(Xte)
T['e']=T.res-pred
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','SCHED_TIME_UTC_mvt','EOBT_1_flt','AOBT_3_flt','RUNWAY_mvt','MVT_TIME_UTC_mvt']).rename(columns={'MVT_ID_mvt':'id'})
tr=tr[(tr.ADEP_mvt=='EGLL')&(tr.PHASE_mvt=='DEP')].copy(); ts=lambda c: pd.to_datetime(tr[c],utc=True,errors='coerce').dt.as_unit('s').astype('int64')
tr['sched']=ts('SCHED_TIME_UTC_mvt'); tr['eobt']=ts('EOBT_1_flt'); tr['aobt']=ts('AOBT_3_flt'); tr['mvt']=ts('MVT_TIME_UTC_mvt'); tr.loc[tr.aobt<0,'aobt']=np.nan; tr.loc[tr.eobt<0,'eobt']=np.nan
X=T[['id','e','y','mvt','blk','chain']].merge(tr[['id','sched','eobt','aobt','RUNWAY_mvt']],on='id',how='left',suffixes=('','_t')).rename(columns={'mvt':'mvt_'}) if False else T[['id','e','y','chain']].merge(tr[['id','sched','eobt','aobt','mvt','RUNWAY_mvt']],on='id',how='left')
dia=tr[(tr.mvt>=pd.Timestamp(DAY,tz='UTC').value//10**9)&(tr.mvt<pd.Timestamp(DAY,tz='UTC').value//10**9+86400)]
X['atraso_sched']=X.mvt-X.sched; X['atraso_eobt']=X.mvt-X.eobt; X['atraso_aobt']=X.mvt-X.aobt; X['eobt_vs_sched']=X.eobt-X.sched
def dens(r,w,col):
    o=dia[(dia.RUNWAY_mvt==r.RUNWAY_mvt)&(dia.mvt!=r.mvt)]; return int(((o[col]-r[col]).abs()<=w).sum())
X['dens_plano']=[dens(r,1200,'sched') for _,r in X.iterrows()]; X['dens_real']=[dens(r,900,'mvt') for _,r in X.iterrows()]
# "desorganizacao": quantos voos programados ANTES de mim ainda estavam no solo quando eu estava pronto (EOBT) e decolaram depois do meu EOBT
def atras_plano(r):
    o=dia[(dia.RUNWAY_mvt==r.RUNWAY_mvt)&(dia.mvt!=r.mvt)]; return int(((o.sched<r.sched)&(o.mvt>r.eobt)).sum()) if r.eobt==r.eobt else np.nan
X['pendentes_antes']=[atras_plano(r) for _,r in X.iterrows()]; X['hora']=pd.to_datetime(X.mvt,unit='s').dt.hour+pd.to_datetime(X.mvt,unit='s').dt.minute/60
print('DIA',DAY,'| voos',len(X),'| RMSE do residuo do solver (CV, ja com espera por contexto): %.1f s'%rm(X.e))
F={'so relogios (MVT-SCHED, MVT-EOBT, MVT-AOBT, EOBT-SCHED)':['atraso_sched','atraso_eobt','atraso_aobt','eobt_vs_sched'],'so desorganizacao do aeroporto (densidade planejada e real, pendentes antes)':['dens_plano','dens_real','pendentes_antes'],'relogios + desorganizacao':['atraso_sched','atraso_eobt','atraso_aobt','eobt_vs_sched','dens_plano','dens_real','pendentes_antes']}
P=dict(n_estimators=200,learning_rate=.03,num_leaves=7,min_child_samples=15,subsample=.8,subsample_freq=1,verbose=-1)
for nome,cols in F.items():
    p=np.zeros(len(X))
    for f in folds:
        te=np.zeros(len(X),bool); te[f]=True; m=lgb.LGBMRegressor(**P).fit(X.loc[~te,cols],X.e[~te]); p[te]=m.predict(X.loc[te,cols])
    print('  residuo explicado por %-72s RMSE %.1f s (ganho %.1f s)'%(nome,rm(X.e-p),rm(X.e)-rm(X.e-p)))
print('\ncorrelacoes simples com o residuo do solver (e = real - previsto):'); 
for c in ['atraso_sched','atraso_eobt','atraso_aobt','eobt_vs_sched','dens_plano','dens_real','pendentes_antes']: print('  %-16s %+.2f'%(c,X[[c,'e']].corr().iloc[0,1]))
b=X.groupby(pd.cut(X.atraso_sched/60,[-999,15,30,60,120,999])).agg(n=('e','size'),residuo_medio=('e','mean'),y_med=('y',lambda s:s.median()/60)); print('\nresiduo por atraso em relacao ao horario programado (MVT - SCHED, min):'); print(b.round(1).to_string())
