"""Janela de CTOT reconstruída (−5/+10 min em torno do CTOT) com EXOT do grafo.
ETOT = pronto + EXOT_grafo. Em s (varredura para trás do MVT, 1 min) o voo 'estava numa janela de CTOT' se ETOT <= s+5 min (algum slot c em [s-10, s+5] com c>=ETOT); 'janela disponível' = isso E pista aberta em s.
Fechado na pista = outro movimento a < b (calibrado, >=90% das decolagens reais abertas). Janela perdida = voo pronto + pista aberta e não decolou -> soma 7 min ao taxi."""
import sys,glob,numpy as np,pandas as pd,lightgbm as lgb
MODE=sys.argv[1] if len(sys.argv)>1 else '2025'   # '2025' ou '2026'
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/data'; T=lambda s: pd.to_datetime(s,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
cal=pd.read_csv('rwy_calib.csv'); CALD={(x.apt,x.rwy):(min(x.b_dep,120.0),min(x.b_arr,120.0)) for x in cal.itertuples()}; APD=cal.groupby('apt')[['b_dep','b_arr']].median().clip(upper=120.0)
if MODE=='2025':
    parts=[]
    for m in range(1,13):
        r=pd.read_parquet(glob.glob(R+'/raw/training_2025-%02d-01_*.parquet'%m)[0],columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','SCHED_TIME_UTC_mvt']); parts.append(r)
    r=pd.concat(parts,ignore_index=True)
else: r=pd.read_parquet(R+'/raw/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','SCHED_TIME_UTC_mvt'])
r['mvt']=T(r.MVT_TIME_UTC_mvt); r['aobt']=T(r.AOBT_3_flt); r['eobt']=T(r.EOBT_1_flt); r['sch']=T(r.SCHED_TIME_UTC_mvt); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64'); r['apt']=np.where(r.PHASE_mvt=='DEP',r.ADEP_mvt,r.ADES_mvt)
# ---------- EXOT do grafo (quantil 0,15 do taxi MVT-AOBT dado só o grafo/stand/pista/aeroporto) ----------
if MODE=='2025': yt=pd.read_parquet('../ablmac/year_table.parquet',columns=['MVT_ID_mvt','apt','stand_lvl','len_full','len_int','n_turns','n_cross','n_nodes','access_m','ma','wk_c','apt_c','day'])
else:
    yt=pd.read_parquet('rank_table.parquet',columns=['MVT_ID_mvt','apt','stand_lvl','len_full','len_int','n_turns','n_cross','n_nodes','access_m','ma','wk_c','apt_c','day'])
yt['MVT_ID_mvt']=yt.MVT_ID_mvt.astype('int64'); dep=r[r.PHASE_mvt=='DEP'][['MVT_ID_mvt','RUNWAY_mvt','mvt','aobt','eobt','sch','ADES_mvt']].drop_duplicates('MVT_ID_mvt'); yt=yt.merge(dep[['MVT_ID_mvt','RUNWAY_mvt']],on='MVT_ID_mvt',how='left')
yt['rwy_c']=(yt.apt+'_'+yt.RUNWAY_mvt.astype(str)).astype('category').cat.codes
FG=['stand_lvl','len_full','len_int','n_turns','n_cross','n_nodes','access_m','wk_c','apt_c','rwy_c']
PQ=dict(objective='quantile',alpha=0.15,n_estimators=300,learning_rate=0.06,num_leaves=63,min_child_samples=200,subsample=0.7,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=14,random_state=0)
if MODE=='2025':
    days=np.sort(yt.day.unique()); rng=np.random.default_rng(0); fo=dict(zip(days,rng.integers(0,5,len(days)))); yt['fold']=yt.day.map(fo); yt['exot']=np.nan; ok=(yt.ma>120)&(yt.ma<3600)
    for f in range(5):
        m=lgb.LGBMRegressor(**PQ).fit(yt.loc[ok&(yt.fold!=f),FG],yt.loc[ok&(yt.fold!=f),'ma']); yt.loc[yt.fold==f,'exot']=m.predict(yt.loc[yt.fold==f,FG])
    print('EXOT grafo (OOF por dia): mediana %.0f s | p10 %.0f | p90 %.0f | corr com ma (voos com ma): %.3f'%(yt.exot.median(),yt.exot.quantile(.1),yt.exot.quantile(.9),yt[ok].exot.corr(yt[ok].ma)),flush=True)
else:
    y25=pd.read_parquet('../ablmac/year_table.parquet',columns=['MVT_ID_mvt','apt','stand_lvl','len_full','len_int','n_turns','n_cross','n_nodes','access_m','ma','wk_c','apt_c']); y25['MVT_ID_mvt']=y25.MVT_ID_mvt.astype('int64')
    rw25=pd.concat([pd.read_parquet(f,columns=['MVT_ID_mvt','RUNWAY_mvt','ADEP_mvt']) for f in glob.glob('slim/rawslim_*.parquet')]); rw25['MVT_ID_mvt']=rw25.MVT_ID_mvt.astype('int64'); y25=y25.merge(rw25,on='MVT_ID_mvt',how='left')
    cats=pd.concat([y25.apt+'_'+y25.RUNWAY_mvt.astype(str),yt.apt+'_'+yt.RUNWAY_mvt.astype(str)]).astype('category').cat.categories; y25['rwy_c']=pd.Categorical(y25.apt+'_'+y25.RUNWAY_mvt.astype(str),categories=cats).codes; yt['rwy_c']=pd.Categorical(yt.apt+'_'+yt.RUNWAY_mvt.astype(str),categories=cats).codes
    ok=(y25.ma>120)&(y25.ma<3600); m=lgb.LGBMRegressor(**PQ).fit(y25.loc[ok,FG],y25.loc[ok,'ma']); yt['exot']=m.predict(yt[FG]); print('EXOT 2026: mediana %.0f s'%yt.exot.median(),flush=True)
# ---------- varredura para trás ----------
K=np.arange(1,46); out=[]
for (a,rw),g in r.groupby(['apt','RUNWAY_mvt']):
    d=g[g.PHASE_mvt=='DEP'].sort_values('mvt'); ar=g[g.PHASE_mvt=='ARR'].sort_values('mvt')
    if len(d)<20: continue
    bd,ba=CALD.get((a,rw),(float(APD.loc[a,'b_dep']),float(APD.loc[a,'b_arr']))) if a in APD.index else (51.0,60.0)
    tD=d.mvt.values; tA=ar.mvt.values
    def near(times,q,own=None):
        n=len(times)
        if n==0: return np.full(q.shape,np.inf)
        idx=np.searchsorted(times,q); best=np.full(q.shape,np.inf)
        for off in (-2,-1,0,1):
            j=np.clip(idx+off,0,n-1); dd=np.abs(times[j]-q)
            if own is not None: dd=np.where(j==own,np.inf,dd)
            best=np.minimum(best,dd)
        return best
    S=tD[:,None]-K[None,:]*60.0; own=np.arange(len(tD))[:,None]*np.ones((1,len(K)),int)
    closed=(near(tD,S,own)<bd)|(near(tA,S)<ba)
    out.append((d.MVT_ID_mvt.values,closed,tD,d.aobt.values,d.eobt.values,d.sch.values,a))
rows=[]
yt=yt.set_index('MVT_ID_mvt')
for ids,closed,tD,aobt,eobt,sch,a in out:
    ex=yt.exot.reindex(ids).values; ex=np.where(np.isfinite(ex),ex,np.nanmedian(yt.exot.values))
    ref=np.where(np.isfinite(aobt),aobt,np.where(np.isfinite(eobt),eobt,sch)); etot_a=ref+ex; etot_e=np.where(np.isfinite(eobt),eobt,sch)+ex
    n=len(ids); s=tD[:,None]-K[None,:]*60.0
    inwin=(etot_a[:,None]<=s+300.0)                     # já numa janela de CTOT possível (ETOT <= s+5 min)
    avail=inwin&(~closed); blocked=inwin&closed
    # janelas perdidas: nº de blocos abertos contíguos (para trás) dentro da parte 'pronta'
    av=avail[:,::-1]                                      # ordem cronológica crescente (k=45..1)
    ch=np.diff(np.concatenate([np.zeros((n,1),int),av.astype(int),np.zeros((n,1),int)],axis=1),axis=1); nlost=(ch==1).sum(1)
    E=(tD-etot_a)/60.0; Ee=(tD-etot_e)/60.0
    o=pd.DataFrame({'MVT_ID_mvt':ids,'cw_exot':ex,'cw_E_wait_min':np.clip(E,-60,180),'cw_E_wait_eobt_min':np.clip(Ee,-60,180),
        'cw_folga_lo':np.clip((tD-10*60-etot_a)/60.0,-60,180),'cw_folga_hi':np.clip((tD+5*60-etot_a)/60.0,-60,180),'cw_folga_eobt_lo':np.clip((tD-10*60-etot_e)/60.0,-60,180),
        'cw_n_ready':inwin.sum(1),'cw_n_avail':avail.sum(1),'cw_n_blocked':blocked.sum(1),'cw_frac_avail':np.where(inwin.sum(1)>0,avail.sum(1)/np.maximum(inwin.sum(1),1),np.nan),'cw_n_lost':nlost,
        'cw_taxi_plus7':ex+420.0*nlost,'cw_taxi_plus7_cap':ex+420.0*np.minimum(nlost,3)})
    for L in (1,2,3,5,7,10,15,20,30,45):
        o[f'cw_ready_{L}']=inwin[:,L-1].astype(float); o[f'cw_avail_{L}']=avail[:,L-1].astype(float)
    rows.append(o)
W=pd.concat(rows,ignore_index=True); W.to_parquet(f'ctotwin_{MODE}.parquet'); print(W.shape); print(W.describe().loc[['mean','25%','50%','75%']].round(2).T.to_string())
