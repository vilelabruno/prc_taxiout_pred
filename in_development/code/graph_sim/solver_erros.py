import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); DAY=sys.argv[1]; S=pickle.load(open(f'{OUT}/solver_{DAY}.pkl','rb')); T=S['T'].copy(); T['dur']=S['dur']; AL=S['AL']; rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
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
T['yh']=T.base+T.dur+pred; T['e']=T.y-T.yh
oc=pd.read_parquet(os.path.expanduser('~/Projetos/oof_store/clean_oof/chain/oof_chain.parquet'),columns=['MVT_ID_mvt','clean_v53']).rename(columns={'MVT_ID_mvt':'id','clean_v53':'champ'}); T=T.merge(oc,on='id',how='left'); T['ec']=T.y-T.champ
print('DIA',DAY,'| voos',len(T),'| RMSE solver (CV) %.1f | campeao %.1f'%(rm(T.e),rm(T.ec.dropna())))
print('\nQUANTOS ACERTA NA RISCA (|erro| do solver):  <=15 s: %.1f%% | <=30 s: %.1f%% | <=60 s: %.1f%% | <=120 s: %.1f%% | >300 s: %.1f%% | >600 s: %.1f%%'%tuple(100*(T.e.abs()<=v).mean() if v!=0 else 0 for v in (15,30,60,120))+'' if False else '')
for v in (15,30,60,120): print('  |erro| <= %3d s: solver %.1f%%  | campeao %.1f%%'%(v,100*(T.e.abs()<=v).mean(),100*(T.ec.abs()<=v).mean()))
for v in (300,600): print('  |erro| >  %3d s: solver %.1f%%  | campeao %.1f%%'%(v,100*(T.e.abs()>v).mean(),100*(T.ec.abs()>v).mean()))
print('  mediana |erro|: solver %.0f s | campeao %.0f s'%(T.e.abs().median(),T.ec.abs().median()))
tot=(T.e**2).sum(); 
for k in (5,10,30,70): ix=T.e.abs().nlargest(k).index; print('  os %d piores = %.1f%% do erro quadratico do solver (no campeao, esses mesmos: %.1f%%)'%(k,100*(T.e[ix]**2).sum()/tot,100*(T.ec[ix]**2).sum()/(T.ec**2).sum()))
def tab(col,nome):
    g=T.groupby(col).agg(n=('y','size'),y_med=('y','median'),vies=('e','mean'),rmse=('e',lambda s:rm(s)),med_abs=('e',lambda s:s.abs().median()),na_risca30=('e',lambda s:(s.abs()<=30).mean()),campeao=('ec',lambda s:rm(s.dropna()))); print('\npor',nome); print(g.round(1).to_string())
T['faixa_y']=pd.cut(T.y,[0,600,800,1000,1300,1800,9999],labels=['<10 min','10-13','13-17','17-22','22-30','>30 min']); tab('faixa_y','faixa do y real'); tab('chain','tem cadeia ADS (1) ou nao (0)'); tab('tipo','tipo de aviao'); tab('hb','bloco de 6 h (0=00-06 UTC)')
print('\n10 MELHORES (|erro| menor):'); c=['cs','stand','tipo','chain','y','yh','champ','e']; print(T.reindex(T.e.abs().sort_values().index).head(10)[c].round(0).to_string(index=False))
print('\n12 PIORES:'); print(T.reindex(T.e.abs().sort_values(ascending=False).index).head(12)[c].round(0).to_string(index=False))
