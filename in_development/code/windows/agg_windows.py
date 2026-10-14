"""Agrega as janelas (voos com trilha própria) + contexto METAR/carga (todos os voos) e constrói 3 grupos para a ablação:
 Janela_obs (observado, só voos com trilha própria) ; Janela_P (probabilidadeDeJanelaAberta, modelo treinado nas trilhas e aplicado a TODOS, com cross-fitting) ; METAR_sync (todos).
Saída: winfeat_all.parquet e groups_win.json"""
import glob,json,sys,numpy as np,pandas as pd,lightgbm as lgb
OUTDIR=sys.argv[1] if len(sys.argv)>1 else 'outall'
w=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(OUTDIR+'/*.parquet'))],ignore_index=True); w['MVT_ID_mvt']=w.MVT_ID_mvt.astype('int64'); print('voos com trilha própria',len(w),'| aeroportos',w.apt.value_counts().to_dict(),flush=True)
DS=list(range(-44,31,2)); dvc=[f'dv_{d:+d}' for d in DS]; dv=w[dvc].replace(np.inf,99).values; dv=np.where(np.isfinite(dv),dv,99.0); i0=DS.index(0)
d0=dv[:,i0]
# limiar por célula (apt,rwy,sec) -> (apt,rwy) -> apt: p10 e p25 da separação real em d=0
def thr(q):
    t=np.full(len(w),np.nan)
    for keys,minn in ((['apt','rwy','sec'],60),(['apt','rwy'],60),(['apt'],1)):
        g=w.assign(d0=d0).groupby(keys).d0.agg(['size',lambda s:s.quantile(q)]); g.columns=['n','q']; g=g[g.n>=minn]
        m=w[keys].merge(g.reset_index(),on=keys,how='left').q.values; t=np.where(np.isnan(t),m,t)
    return t
T10=thr(.10); T25=thr(.25)
def feats(T,suf):
    cl=dv<T[:,None]; n=len(w); o={}
    pre=cl[:,:i0]; o['closed0'+suf]=cl[:,i0].astype(float); o['ratio0'+suf]=d0/T
    o['fclosed10'+suf]=cl[:,i0-5:i0].mean(1); o['fclosed30'+suf]=cl[:,i0-15:i0].mean(1); o['fclosed44'+suf]=pre.mean(1)
    L=np.zeros(n);R=np.zeros(n);Cb=np.zeros(n);no=np.zeros(n);mo=np.zeros(n);nb=np.zeros(n)
    for i in range(n):
        c=cl[i]; l=0
        while i0-l-1>=0 and not c[i0-l-1]: l+=1
        r=0
        while i0+r+1<len(c) and not c[i0+r+1]: r+=1
        j=i0-l-1;cc=0
        while j>=0 and c[j]: cc+=1;j-=1
        L[i]=l*2;R[i]=r*2;Cb[i]=cc*2; p=c[:i0]
        ch=np.diff(np.r_[0,(~p).astype(int),0]); st=np.where(ch==1)[0];en=np.where(ch==-1)[0]; no[i]=len(st); mo[i]=(en-st).mean()*2 if len(st) else 0
        ch=np.diff(np.r_[0,p.astype(int),0]); nb[i]=len(np.where(ch==1)[0])
    o['L'+suf]=L;o['R'+suf]=R;o['width'+suf]=L+R+2;o['closed_before'+suf]=Cb;o['n_open_win'+suf]=no;o['mean_open_win'+suf]=mo;o['n_closed_blocks'+suf]=nb
    o['minratio_pre'+suf]=(dv[:,i0-5:i0].min(1))/T
    return pd.DataFrame(o)
F=pd.concat([feats(T10,'_p10'),feats(T25,'_p25')],axis=1); F.insert(0,'MVT_ID_mvt',w.MVT_ID_mvt.values); OBS=[c for c in F.columns if c!='MVT_ID_mvt']
# ---------- probabilidadeDeJanelaAberta ----------
X=pd.read_parquet('context_all.parquet'); X['MVT_ID_mvt']=X.MVT_ID_mvt.astype('int64'); X['apt_c']=X.apt.astype('category').cat.codes; X['rwy_c']=(X.apt+'_'+X.rwy.astype(str)).astype('category').cat.codes
CTX=[c for c in X.columns if c not in ('MVT_ID_mvt','apt','rwy','mvt','sec')]+[]
yt=pd.read_parquet('../ablmac/year_table.parquet',columns=['MVT_ID_mvt','day','holdout']); yt['MVT_ID_mvt']=yt.MVT_ID_mvt.astype('int64')
X=X.merge(yt,on='MVT_ID_mvt',how='left'); X['sec']=X.sec.fillna(-1)
DL=[-30,-20,-10,-6,-4,-2,2,4,6,10]; DP=[-10,-6,-4,-2,2,4]
rows=[]
Wm=w[['MVT_ID_mvt']].copy(); Wm['T']=T10
for d in DL:
    k=DS.index(d); t=Wm.assign(d=d,open=(dv[:,k]>=T10).astype(int)); rows.append(t)
TR=pd.concat(rows,ignore_index=True).merge(X,on='MVT_ID_mvt',how='inner'); print('linhas de treino do P_open',len(TR),'| fração aberta %.2f'%TR.open.mean(),flush=True)
FEAT=CTX+['d','sec']; FEAT=[c for c in FEAT if c not in ('day','holdout')]
Pm=dict(n_estimators=200,learning_rate=0.08,num_leaves=63,min_child_samples=200,subsample=0.7,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=14)
days_all=np.sort(X.day.dropna().unique()); rng=np.random.default_rng(0); fold_of={d:int(f) for d,f in zip(days_all,rng.integers(0,5,len(days_all)))}
X['fold']=X.day.map(fold_of); TR['fold']=TR.day.map(fold_of); hold_tr=(TR.holdout==1).values
pred={d:np.full(len(X),np.nan,np.float32) for d in DP}
def fit(mask): return lgb.LGBMClassifier(**Pm).fit(TR.loc[mask,FEAT],TR.loc[mask,'open'])
def predict_rows(m,rowsmask):
    for d in DP:
        Z=X.loc[rowsmask,FEAT[:-2]].copy(); Z['d']=d; Z['sec']=X.loc[rowsmask,'sec'].values; pred[d][rowsmask]=m.predict_proba(Z[FEAT])[:,1]
print('treinando P_open (hold-out jan/jul usa só meses de treino; meses de treino por 5 folds de dias)...',flush=True)
m=fit(~hold_tr); predict_rows(m,(X.holdout==1).values)
for f in range(5):
    mk=(~hold_tr)&(TR.fold!=f).values; m=fit(mk); predict_rows(m,((X.holdout==0)&(X.fold==f)).values)
from sklearn.metrics import roc_auc_score
mh=hold_tr; mm=fit(~hold_tr); pa=mm.predict_proba(TR.loc[mh,FEAT])[:,1]; print('AUC do P_open no holdout (jan/jul, voos com trilha): %.3f'%roc_auc_score(TR.loc[mh,'open'],pa),flush=True)
for nm,sub in (('só d<0',TR.loc[mh,'d']<0),('só d>0',TR.loc[mh,'d']>0)):
    s=sub.values; print('  AUC',nm,'%.3f'%roc_auc_score(TR.loc[mh,'open'][s],pa[s]),flush=True)
P=pd.DataFrame({'MVT_ID_mvt':X.MVT_ID_mvt.values})
for d in DP: P[f'p_open_{d:+d}']=pred[d]
pre=[f'p_open_{d:+d}' for d in (-10,-6,-4,-2)]; post=['p_open_+2','p_open_+4']
P['p_open_pre_mean']=P[pre].mean(1); P['p_open_pre_min']=P[pre].min(1); P['p_open_post_mean']=P[post].mean(1); P['p_open_gap']=P.p_open_post_mean-P.p_open_pre_mean
P['p_wait_proxy']=(1-P.p_open_pre_mean)*10   # espera esperada (min) se cada minuto antes fosse independente
POP=[c for c in P.columns if c!='MVT_ID_mvt']
WX=[c for c in X.columns if c.startswith(('wx_','agl_','incloud_','frac_nuvem'))]; LOAD=[c for c in X.columns if c.startswith(('ndep_','narr_','hw_prev','hw_next'))]
M=X[['MVT_ID_mvt']+WX+LOAD]
A=M.merge(P,on='MVT_ID_mvt').merge(F,on='MVT_ID_mvt',how='left'); A.to_parquet('winfeat_all.parquet')
json.dump({'Janela_obs':OBS,'Janela_P':POP,'METAR_sync':WX,'Carga_pista':LOAD},open('groups_win.json','w'),indent=1); print('ok',A.shape,{k:len(v) for k,v in json.load(open('groups_win.json')).items()},flush=True)
