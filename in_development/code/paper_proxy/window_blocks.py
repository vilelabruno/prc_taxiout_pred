"""Janelas de bloco (15 min): um voo só 'passa' no bloco b se origem (no bloco b) E destino (no bloco b + T_od) têm vaga; senão espera o próximo bloco livre.
 util_r(b) = max( densidade de aeronaves/p99 , movimentos(pouso+decolagem)/p99 ) da região r no bloco b  (capacidade = p99 do aeroporto nos 62 dias)
 A(b) = max(util_o(b), util_d(b+tau))  (a região mais crítica manda) ; bloqueado se A(b) >= theta
 k*(theta) = menor k >= 0 com A(b0+k) < theta  (blocos de espera)  ; contrafactual: kneg(theta) = maior j>=1 tal que A(b0-j) < theta (o voo teria passado se estivesse j blocos antes)
 b0 = bloco de a = TOBT + EXOT ; theta = quantis 0.60, 0.77 (regulados ~23% no EGLL), 0.90 de A."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, ceu_feats as CF
S=os.path.dirname(os.path.abspath(__file__))+'/'; B=900
A_,C,H,R=CF.load(S+'ceu/'); out,p95,cap=CF.series(A_,C,H,R); code=dict(zip(A_.ident,A_.code))
blk={}
for j,s in out.items():
    s=s.copy(); s['blk']=s.index//3; g=s.groupby('blk').agg(dens=('n_low','mean'),mov=('arr','sum')); g['mov']=g.mov+s.groupby('blk').dep.sum()
    idx=np.arange(g.index.min(),g.index.max()+1); g=g.reindex(idx,fill_value=0.0)
    cd=max(float(g.dens.quantile(.99)),5.0); cm=max(float(g.mov.quantile(.99)),6.0)   # piso: aeroportos pequenos
    u=np.minimum(np.maximum(g.dens/cd,g.mov/cm),3.0); blk[j]=pd.Series(u.values,index=idx)
d=pd.read_parquet(S+'egll_base.parquet'); x=pd.read_parquet(S+'egll_ceu25.parquet')[['MVT_ID_mvt','t_od','eta']]; sm=pd.read_parquet(S+'sim25.parquet')[['MVT_ID_mvt','sim_tau']]
d=d.merge(x,on='MVT_ID_mvt',how='left').merge(sm,on='MVT_ID_mvt',how='left')
mv=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True); e=pd.to_datetime(d.EOBT_1_flt,utc=True); s_=pd.to_datetime(d.SCHED_TIME_UTC_mvt,utc=True)
tobt=e.where(e.notna(),s_); mvs=mv.astype('datetime64[s, UTC]').astype('int64').astype(float); tb=tobt.astype('datetime64[s, UTC]').astype('int64').astype(float).where(tobt.notna())
tb=np.minimum(tb,mvs); a=tb+d.sim_tau.fillna(300).values+60.0                                   # chegada à pista planejada
b0=np.floor(a/B).astype(float); tau=np.round(d.t_od.values/B)
jo=code['EGLL']; uo=blk[jo]; dest=np.array([code.get(v,np.nan) for v in d.ADES_mvt],float)
def getu(ser,b): 
    b=np.asarray(b); ok=np.isfinite(b); r=np.full(len(b),np.nan); bb=b[ok].astype(int)-int(ser.index.min()); inr=(bb>=0)&(bb<len(ser)); v=np.full(len(bb),np.nan); v[inr]=ser.values[bb[inr]]; r[ok]=v; return r
K=range(-4,9)
Uo={k:getu(uo,(b0+k)) for k in K}
Ud={k:np.full(len(d),np.nan) for k in K}
for j in np.unique(dest[np.isfinite(dest)]).astype(int):
    if j not in blk: continue
    m=np.where(dest==j)[0]
    for k in K: Ud[k][m]=getu(blk[j],(b0+k+tau)[m])
Acomb={k:np.fmax(Uo[k],Ud[k]) for k in K}                  # destino ausente -> só origem
Aorig={k:Uo[k] for k in K}
F=pd.DataFrame({'MVT_ID_mvt':d.MVT_ID_mvt.astype('int64').values})
allA=np.concatenate([Acomb[0][np.isfinite(Acomb[0])]])
for q in (0.60,0.77,0.90):
    th=np.quantile(allA,q)
    for nm,AA in (('od',Acomb),('o',Aorig)):
        blocked0=AA[0]>=th; ks=np.full(len(d),np.nan); kneg=np.zeros(len(d))
        for i in range(len(d)):
            if not np.isfinite(AA[0][i]): continue
            if not blocked0[i]: ks[i]=0
            else:
                ks[i]=next((k for k in range(1,9) if np.isfinite(AA[k][i]) and AA[k][i]<th),9)
            kneg[i]=next((j for j in range(1,5) if np.isfinite(AA[-j][i]) and AA[-j][i]<th),0) if blocked0[i] else 0
        F[f'wb_k_{nm}_{int(q*100)}']=ks; F[f'wb_kneg_{nm}_{int(q*100)}']=kneg; F[f'wb_blk_{nm}_{int(q*100)}']=blocked0.astype(float)
F['wb_A0']=Acomb[0]; F['wb_Ao0']=Uo[0]; F['wb_Ad0']=Ud[0]; F['wb_Amax_fut']=np.fmax.reduce([Acomb[k] for k in (1,2,3,4)]); F['wb_Amin_fut']=np.fmin.reduce([Acomb[k] for k in (1,2,3,4)])
F['wb_A_m1']=Acomb[-1]; F['wb_A_m2']=Acomb[-2]; F['wb_A_p1']=Acomb[1]
F.to_parquet(S+'egll_wb25.parquet'); print(F.describe().T[['count','mean']].round(3).to_string())
