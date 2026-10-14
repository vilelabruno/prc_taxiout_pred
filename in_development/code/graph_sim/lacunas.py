import sys, numpy as np, pandas as pd
apt=sys.argv[1]; days=sys.argv[2].split(',')
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
R=[]
for day in days:
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet'); a=a[a.apt==apt].sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    for r in tr[(tr.mvt>=t0+3600)&(tr.mvt<t0+23*3600)].itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt+60)]
        g0=v[v.alt==0]
        if len(g0)<10 or r.mvt-g0.t.max()>20: continue
        v=v[v.icao24==g0.icao24.iloc[-1]]
        t=v.t.to_numpy(); gs=v.gs.fillna(0).to_numpy(); al=v.alt.to_numpy()
        mv=np.where((gs>2)&(al==0))[0]
        if not len(mv): continue
        tm=t[mv[0]]
        if tm>=r.mvt: continue
        w=t[(t>=tm)&(t<=r.mvt)]; w=np.r_[w,r.mvt] if w[-1]<r.mvt else w
        dg=np.diff(w); span=r.mvt-tm
        j=np.searchsorted(t,r.mvt)
        R.append(dict(span=span,c10=dg[dg<=10].sum()/span,c30=dg[dg<=30].sum()/span,
            gap10_60=dg[(dg>10)&(dg<=60)].sum()/span,gap60=dg[dg>60].sum()/span,
            gap_mvt=(t[j]-t[j-1]) if 0<j<len(t) else np.nan,
            alt_after=al[j] if j<len(t) else np.nan, gs_bef=gs[j-1] if j>0 else np.nan))
d=pd.DataFrame(R); pd.set_option('display.width',200)
print(apt,len(d)); print(d.describe(percentiles=[.1,.25,.5,.75,.9]).T[['mean','10%','25%','50%','75%','90%']].round(2))
