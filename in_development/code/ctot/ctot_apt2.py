"""Pesos por aeroporto para a ESPERA NO STAND (hold = y - ma = AOBT - BLOCK, alvo que as features não determinam). ETOT ancorado no EOBT (readiness antes do pushback).
Regressão por aeroporto no treino (holdout==0); avaliação no holdout (R2 e correlação) contra o modelo pooled. Saída: ctotapt2_2025.parquet (ca2_*)."""
import numpy as np,pandas as pd
cw=pd.read_parquet('ctotwin_2025.parquet'); cw['MVT_ID_mvt']=cw.MVT_ID_mvt.astype('int64')
yt=pd.read_parquet('../ablmac/year_table.parquet',columns=['MVT_ID_mvt','apt','y','ma','holdout','noaobt']); yt['MVT_ID_mvt']=yt.MVT_ID_mvt.astype('int64')
d=yt.merge(cw,on='MVT_ID_mvt',how='inner'); d['hold']=(d.y-d.ma)
FE=['cw_E_wait_eobt_min','cw_folga_eobt_lo','cw_n_blocked','cw_n_lost','cw_n_avail','cw_frac_avail']; d['ma_min']=d.ma/60
FEA=FE+['ma_min']; d[FE+['ma_min']]=d[FE+['ma_min']].fillna(0.0)
ok=d.ma.notna()&(d.hold>-300)&(d.hold<3600)
def fit(g,fe):
    tr=g[(g.holdout==0)&ok.loc[g.index]]; y=tr.hold.clip(0,3600).values; X=np.c_[np.ones(len(tr)),tr[fe].values]; b,*_=np.linalg.lstsq(X,y,rcond=None); return b
# pooled
bp=fit(d,FEA); rows=[]; out=[]
for a,g in d.groupby('apt'):
    b=fit(g,FEA); te=g[(g.holdout==1)&ok.loc[g.index]]; y=te.hold.clip(0,3600).values
    ph=np.c_[np.ones(len(te)),te[FEA].values]@b; pp=np.c_[np.ones(len(te)),te[FEA].values]@bp
    r2=lambda p: 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    rows.append(dict(apt=a,n_te=len(te),R2_apt=round(r2(ph),3),R2_pooled=round(r2(pp),3),corr_apt=round(np.corrcoef(ph,y)[0,1],3),**{k.replace('cw_',''):round(v,1) for k,v in zip(['b0']+FEA,b)}))
    Xa=np.c_[np.ones(len(g)),g[FEA].values]; fit_=Xa@b
    out.append(pd.DataFrame({'MVT_ID_mvt':g.MVT_ID_mvt.values,'ca2_hold_fit':fit_,'ca2_hold_fit_pooled':Xa.dot(bp) if False else np.c_[np.ones(len(g)),g[FEA].values]@bp,'ca2_b_lost_s':b[FEA.index('cw_n_lost')+1],'ca2_b_blocked_s':b[FEA.index('cw_n_blocked')+1],'ca2_b_Ee_s_per_min':b[1]}))
W=pd.concat(out,ignore_index=True); W['ca2_diff_apt_pooled']=W.ca2_hold_fit-W.ca2_hold_fit_pooled; W.to_parquet('ctotapt2_2025.parquet')
print(pd.DataFrame(rows).to_string(index=False)); print(W.shape)
