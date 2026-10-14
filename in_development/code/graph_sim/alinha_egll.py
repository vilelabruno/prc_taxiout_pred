import sys, numpy as np, pandas as pd, warnings, glob, os; warnings.filterwarnings('ignore')
sys.path.insert(0,'src/physical'); import taxigraph as T
from scipy.spatial import cKDTree
apt=sys.argv[1]; KT=0.514444; g=T.TaxiGraph(apt)
pts=[];kind=[]
for e in g.edges:
    a,b=g.xy[e['a']],g.xy[e['b']]; n=max(int(e['L']//4),1); pts.append(a+(b-a)*np.linspace(0,1,n+1)[:,None]); kind+=[e['kind']]*(n+1)
tree=cKDTree(np.vstack(pts)); kind=np.array(kind)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt'])
tr=tr[(tr.ADEP_mvt==apt)&(tr.PHASE_mvt=='DEP')].copy(); tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
rows=[]; sp=[]
for day in sys.argv[2].split(','):
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    # (b) espacial: pings em movimento (>5 m/s) a que distancia da aresta mais proxima
    m=a[(a.alt==0)&(a.gs*KT>5)]; X=np.c_[(m.lon-g.lon0)*g.kx,(m.lat-g.lat0)*g.ky]; d,_=tree.query(X); sp.append(pd.DataFrame({'day':day,'d':d}))
    grp={k:v for k,v in a.groupby('cs')}
    for r in tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)].itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-900)&(v.t<r.mvt+900)]
        if len(v)<4: continue
        v=v[v.icao24==v.icao24.iloc[-1]]; gs=v.gs.to_numpy(); t=v.t.to_numpy()
        # primeiro instante em que gs>=80 kt (decolagem), por interpolacao
        c=np.where((gs[:-1]<80)&(gs[1:]>=80))[0]
        if len(c): t80=np.interp(80,gs[c[0]:c[0]+2],t[c[0]:c[0]+2]); rows.append(dict(day=day,off=r.mvt-t80))
o=pd.DataFrame(rows); s=pd.concat(sp)
print(apt,'| (tempo) voos com cruzamento de 80 kt:',len(o))
if len(o):
    o['mes']=o.day.str[:7]
    print(o.groupby('mes').off.describe(percentiles=[.1,.5,.9]).round(0))
    print('por dia (mediana de MVT - t80, s):',o.groupby('day').off.median().round(0).to_dict())
print('(espaco) distancia dos pings em movimento (>5 m/s) ao grafo: percentis 10/50/75/90/95/99 (m)',s.d.quantile([.1,.5,.75,.9,.95,.99]).round(1).tolist(),'| <=15 m:',round((s.d<=15).mean(),3),'| <=50 m:',round((s.d<=50).mean(),3))
print(s.groupby('day').d.median().round(1).to_dict())
