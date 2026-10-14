"""Todas as janelas de cada pista + referência do voo dentro da janela + janelas OCIOSAS entre 'pronto na cabeceira' e a decolagem.
Fechado = união de [t-b, t+b] de cada movimento (b_dep/b_arr calibrados: >=90% das decolagens reais em janela aberta). Aberto = complemento = janelas."""
import glob,numpy as np,pandas as pd
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/data'; T=lambda s: pd.to_datetime(s,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
cal0=pd.read_csv('rwy_calib.csv'); cal=cal0.set_index(['apt','rwy']); apm=cal0.groupby('apt')[['b_dep','b_arr']].median().clip(upper=120.0)
r=pd.read_parquet(R+'/raw/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt']); r['mvt']=T(r.MVT_TIME_UTC_mvt); r['aobt_s']=T(r.AOBT_3_flt); r=r.drop(columns=['MVT_TIME_UTC_mvt','AOBT_3_flt']); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64'); r['apt']=np.where(r.PHASE_mvt=='DEP',r.ADEP_mvt,r.ADES_mvt)
yt=r[r.PHASE_mvt=='DEP'][['MVT_ID_mvt','aobt_s']].copy()
out=[]
def closed_measure_factory(s,e):
    L=np.cumsum(e-s); Lp=np.r_[0.0,L]
    def F(x):
        k=np.searchsorted(e,x,side='right')   # blocos totalmente antes de x
        base=Lp[k]; part=np.where((k<len(s))&(x>s[np.clip(k,0,len(s)-1)]),x-s[np.clip(k,0,len(s)-1)],0.0); return base+np.clip(part,0,None)
    return F
for (a,rw),g in r.groupby(['apt','RUNWAY_mvt']):
    if (a,rw) in cal.index: bD=min(float(cal.loc[(a,rw),'b_dep']),120.0); bA=min(float(cal.loc[(a,rw),'b_arr']),120.0)
    elif a in apm.index: bD=float(apm.loc[a,'b_dep']); bA=float(apm.loc[a,'b_arr'])
    else: continue
    dep=g[g.PHASE_mvt=='DEP'].sort_values('mvt').reset_index(drop=True); arr=g[g.PHASE_mvt=='ARR'].sort_values('mvt')
    if len(dep)<20: continue
    dep=dep.drop(columns=['aobt_s']).merge(yt[['MVT_ID_mvt','aobt_s']],on='MVT_ID_mvt',how='left'); tD=dep.mvt.values; tA=arr.mvt.values
    # blocos fechados (união)
    ev=np.r_[np.c_[tD-bD,tD+bD],np.c_[tA-bA,tA+bA]] if len(tA) else np.c_[tD-bD,tD+bD]; ev=ev[np.argsort(ev[:,0])]
    s=[];e=[]; cs,ce=ev[0]
    for a1,b1 in ev[1:]:
        if a1<=ce: ce=max(ce,b1)
        else: s.append(cs);e.append(ce);cs,ce=a1,b1
    s.append(cs);e.append(ce); s=np.array(s);e=np.array(e); F=closed_measure_factory(s,e); gaps=s[1:]-e[:-1]
    # vizinhos (outros movimentos) para a posição do voo na janela
    allt=np.r_[tD,tA]; typ=np.r_[np.zeros(len(tD)),np.ones(len(tA))]; o=np.argsort(allt); allt=allt[o]; typ=typ[o]; b_of=np.where(typ==0,bD,bA)
    pos=np.searchsorted(allt,tD)   # índice do próprio voo (ou posição de inserção)
    # o próprio evento está em allt na posição 'pos' (primeira ocorrência de igual); anterior = pos-1; seguinte = pos+1 (ajuste por empates)
    pos=np.where(allt[np.clip(pos,0,len(allt)-1)]==tD,pos,pos)
    prv=np.clip(pos-1,0,len(allt)-1); nxt=np.clip(pos+1,0,len(allt)-1)
    early=tD-(allt[prv]+b_of[prv]); late=(allt[nxt]-b_of[nxt])-tD; Lown=early+late
    # pronto na cabeceira = AOBT + taxi livre (p10 do taxi da pista); espera
    taxi=tD-dep.aobt_s.values; free=np.nanpercentile(taxi[np.isfinite(taxi)&(taxi>0)],10) if np.isfinite(taxi).any() else np.nan
    Rr=np.where(np.isfinite(dep.aobt_s.values),np.minimum(dep.aobt_s.values+free,tD),np.nan); end=tD-bD
    ok=np.isfinite(Rr)&(end>Rr)
    idle=np.where(ok,(end-Rr)-(F(np.where(ok,end,0))-F(np.where(ok,Rr,0))),np.nan)           # tempo de pista ABERTO (ocioso) entre pronto e a decolagem
    nbl=np.where(ok,np.searchsorted(s,end,side='right')-np.searchsorted(s,np.where(ok,Rr,0),side='right'),np.nan)   # nº de blocos fechados no intervalo -> nº de janelas ≈ blocos+1
    wait=np.where(np.isfinite(Rr),tD-Rr,np.nan)
    d=pd.DataFrame({'MVT_ID_mvt':dep.MVT_ID_mvt.values,'rg_early_slack':early,'rg_late_slack':late,'rg_own_window':Lown,'rg_pos_frac':np.where(Lown>0,early/np.maximum(Lown,1e-9),np.nan),
        'rg_wait_ready':wait,'rg_idle_open_s':idle,'rg_idle_frac':np.where(wait>0,idle/np.maximum(wait,1e-9),np.nan),'rg_n_windows_ready':nbl+1,'rg_idle_slots':idle/max(bD,1.0),
        'rg_gap_p50':np.median(gaps),'rg_gap_p90':np.quantile(gaps,0.9),'rg_util':(len(tD)+len(tA))*60*60/ (tD.max()-tD.min()+1)})
    out.append(d)
W=pd.concat(out,ignore_index=True); W.to_parquet('rwy_ref_2026.parquet'); print(W.shape); print(W.describe().loc[['mean','25%','50%','75%']].round(2).T.to_string())
