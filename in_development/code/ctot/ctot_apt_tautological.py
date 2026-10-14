"""Pesos do CTOT_janela por aeroporto: regressão (treino = meses sem holdout) de (ma - exot) em [n_lost, n_blocked, n_avail, 1] por aeroporto; ma winsorizado em [300,2400] s.
Saída: ctotapt_2025.parquet com ca_* (preditor de taxi calibrado, resíduo MVT-AOBT menos o preditor, e os próprios pesos por aeroporto)."""
import numpy as np,pandas as pd
cw=pd.read_parquet('ctotwin_2025.parquet'); cw['MVT_ID_mvt']=cw.MVT_ID_mvt.astype('int64')
yt=pd.read_parquet('../ablmac/year_table.parquet',columns=['MVT_ID_mvt','apt','ma','holdout']); yt['MVT_ID_mvt']=yt.MVT_ID_mvt.astype('int64')
d=yt.merge(cw[['MVT_ID_mvt','cw_exot','cw_n_lost','cw_n_blocked','cw_n_avail','cw_E_wait_min','cw_frac_avail']],on='MVT_ID_mvt',how='inner')
FE=['cw_n_lost','cw_n_blocked','cw_n_avail']; out=[]; rows=[]
for a,g in d.groupby('apt'):
    tr=g[(g.holdout==0)&g.ma.notna()&(g.ma>120)&(g.ma<3600)]
    y=(tr.ma.clip(300,2400)-tr.cw_exot).values; X=np.c_[np.ones(len(tr)),tr[FE].values]; b,*_=np.linalg.lstsq(X,y,rcond=None)
    Xa=np.c_[np.ones(len(g)),g[FE].values]; fit=g.cw_exot.values+Xa@b
    o=pd.DataFrame({'MVT_ID_mvt':g.MVT_ID_mvt.values,'ca_taxi_fit':fit,'ca_resid':g.ma.values-fit,'ca_b0':b[0],'ca_b_lost_min':b[1]/60,'ca_b_blocked_min':b[2]/60,'ca_b_avail_min':b[3]/60}); out.append(o)
    r2=1-((y-X@b)**2).sum()/((y-y.mean())**2).sum(); rows.append(dict(apt=a,n=len(tr),b0_s=round(b[0]),min_por_janela_perdida=round(b[1]/60,2),min_por_min_bloqueado=round(b[2]/60,2),min_por_min_disponivel=round(b[3]/60,2),R2=round(r2,3)))
W=pd.concat(out,ignore_index=True); W['ca_resid_abs']=W.ca_resid.abs(); W.to_parquet('ctotapt_2025.parquet'); print(pd.DataFrame(rows).to_string(index=False)); print(W.shape)
