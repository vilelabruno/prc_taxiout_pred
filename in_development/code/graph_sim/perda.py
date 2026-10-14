import sys, numpy as np, pandas as pd
sys.path.insert(0,'src/physical')
import taxigraph as T
from airport_mapmatch import SurfaceMatcher
import edge_traversals as E
apt,day=sys.argv[1],sys.argv[2]
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
t0=pd.Timestamp(day,tz='UTC').value//10**9; tr=tr[(tr.mvt>=t0+4*3600)&(tr.mvt<t0+20*3600)]
a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet'); a=a[(a.apt==apt)&(a.alt==0)&a.gs.notna()].sort_values('t'); a['cs']=a.callsign.str.strip()
m=E.matcher_for(apt); g=m.g
grp={k:v for k,v in a.groupby('cs')}
R=[]
for r in tr.sample(min(120,len(tr)),random_state=0).itertuples():
    v=grp.get(r.cs)
    if v is None: continue
    v=v[(v.t>r.mvt-3600)&(v.t<=r.mvt+30)].copy()
    if len(v)<20: continue
    tot=len(v); fast=(v.gs>40).sum()
    slow=v[v.gs<=40].reset_index(drop=True)
    z=m.match(slow[['t','lat','lon','gs']])
    slow=slow.join(z[['arc','distance_m','margin_cost']])
    trusted=slow.arc.notna()&(slow.distance_m<=12)&(slow.margin_cost>=4)
    stopped=slow.gs<1
    trav=E.track_traversals(slow[['t','lat','lon','gs']],m)
    cov_t=sum(x[1]-x[0] for x in trav)
    span=v.t.max()-v.t.min()
    # tempo parado fora do stand (aprox: gs<1 apos primeiro movimento)
    mv=slow.index[slow.gs>2]
    st_after=0
    if len(mv):
        s2=slow.loc[mv[0]:]; dt=s2.t.diff().shift(-1).fillna(0).clip(upper=30)
        st_after=float(dt[s2.gs<1].sum())
    R.append(dict(pts=tot,frac_fast=fast/tot,frac_nocand=slow.arc.isna().mean(),frac_trusted=trusted.mean(),
        frac_trusted_stopped=(trusted&stopped).sum()/max(stopped.sum(),1),
        dist_med=slow.distance_m.median(),cov_time=cov_t/span,stop_after_move_s=st_after,span=span))
d=pd.DataFrame(R); pd.set_option('display.width',200)
print(apt,day,'voos',len(d)); print(d.describe(percentiles=[.25,.5,.75]).T[['mean','25%','50%','75%']].round(3))
