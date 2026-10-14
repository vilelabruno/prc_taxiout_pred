import sys, numpy as np, pandas as pd
sys.path.insert(0,sys.argv[3]); sys.path.insert(0,'src/physical')
from sincro import Sync
import edge_traversals as E
apt=sys.argv[1]; days=sys.argv[2].split(',')
sy=Sync(apt); g=sy.g; mo=E.matcher_for(apt)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
R=[]
for day in days:
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet'); a=a[(a.apt==apt)&(a.alt==0)].sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    m=tr[(tr.mvt>=t0+3600)&(tr.mvt<t0+23*3600)]
    for r in m.itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt+20)]
        if len(v)<10 or r.mvt-v.t.max()>20: continue
        ic=v.icao24.iloc[-1]; v=v[v.icao24==ic]
        seg=(v.t.diff().fillna(0)>300).cumsum(); v=v[seg==seg.iloc[-1]].reset_index(drop=True)
        if len(v)<10: continue
        pr,rr=g.rwy_pair(str(r.RUNWAY_mvt))
        if rr is None: continue
        P=np.c_[(v.lon-g.lon0)*g.kx,(v.lat-g.lat0)*g.ky]; t=v.t.to_numpy(); gs=v.gs.fillna(0).to_numpy()
        sn=sy.snap(P); ed=np.array([x[0] for x in sn]); dist=np.array([x[2] for x in sn]); ok=ed>=0
        if ok.sum()<5: continue
        nodes=sy.route(list(ed[ok])); Sg,Dr,cum=sy.along(nodes,P)
        # primeiro movimento
        mv=np.where(gs>2)[0]; tm=t[mv[0]] if len(mv) else np.nan
        span=r.mvt-(tm if np.isfinite(tm) else t[0])
        # cobertura nova: segundos entre tm e mvt cobertos por par de pontos casados com lacuna<=10 s
        tt=t[ok]; good=(np.diff(tt)<=10)
        lo=np.maximum(tt[:-1],tm); hi=np.minimum(tt[1:],r.mvt)
        cov_new=float(np.clip(hi-lo,0,None)[good].sum())/span if span>0 else np.nan
        # cobertura antiga: travessias de edge_traversals sobre o mesmo trecho
        try:
            sl=v[v.gs.notna()&(v.gs<=40)][['t','lat','lon','gs']].reset_index(drop=True)
            trv=E.track_traversals(sl,mo) if len(sl)>=2 else []
        except Exception: trv=[]
        cov_old=sum(max(0,min(b,r.mvt)-max(a_,tm)) for a_,b,*_ in trv)/span if span>0 else np.nan
        # MVT como ponto: distancia ao longo da pista a partir da cabeceira
        end=str(r.RUNWAY_mvt); thr=np.array(g.proj(rr['lat1'],rr['lon1'])) if end==rr['end1'] else np.array(g.proj(rr['lat2'],rr['lon2']))
        oth=np.array(g.proj(rr['lat2'],rr['lon2'])) if end==rr['end1'] else np.array(g.proj(rr['lat1'],rr['lon1']))
        u=(oth-thr)/np.hypot(*(oth-thr))
        j=np.searchsorted(t,r.mvt)
        x_mvt=np.nan; gap_mvt=np.nan; cross=np.nan
        if 0<j<len(t):
            gap_mvt=t[j]-t[j-1]; w=(r.mvt-t[j-1])/max(gap_mvt,1e-9); pm=P[j-1]+w*(P[j]-P[j-1])
            x_mvt=float((pm-thr)@u); cross=float(abs((pm-thr)[0]*u[1]-(pm-thr)[1]*u[0]))
        al=(P-thr)@u; cr=np.abs((P-thr)[:,0]*u[1]-(P-thr)[:,1]*u[0])
        onr=(cr<40)&(al>-100)
        t_lineup=t[onr][0] if onr.any() else np.nan
        R.append(dict(day=day,rwy=end,n=len(v),frac_match=ok.mean(),dist_med=np.nanmedian(dist),dist_route_p90=np.quantile(Dr,.9),
            span=span,cov_new=cov_new,cov_old=cov_old,x_mvt=x_mvt,cross_mvt=cross,gap_mvt=gap_mvt,
            mvt_minus_lineup=r.mvt-t_lineup,route_len=cum[-1],n_nodes=len(nodes),gs_mvt=np.interp(r.mvt,t,gs)))
d=pd.DataFrame(R); d.to_csv(f'{sys.argv[3]}/sincro_{apt}.csv',index=False)
pd.set_option('display.width',220)
print(apt,'partidas',len(d))
print(d[['frac_match','dist_med','dist_route_p90','cov_old','cov_new','gap_mvt','cross_mvt']].describe(percentiles=[.1,.25,.5,.75,.9]).T[['10%','25%','50%','75%','90%']].round(2))
q=d[d.gap_mvt<=10]
print('MVT como ponto na pista (gap<=10 s): posicao ao longo da pista desde a cabeceira, m')
print(q.groupby('rwy').x_mvt.describe(percentiles=[.1,.5,.9]).round(0))
