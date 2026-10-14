import sys, numpy as np, pandas as pd
sys.path.insert(0,'src/physical')
import edge_traversals as E
apt,day=sys.argv[1],sys.argv[2]
a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet'); a=a[(a.apt==apt)&(a.alt==0)&a.gs.notna()&(a.gs<=40)]
m=E.matcher_for(apt)
ids=a.icao24.drop_duplicates().sample(60,random_state=1)
Z=[]
for i in ids:
    v=a[a.icao24==i].sort_values('t').reset_index(drop=True)
    if len(v)<5: continue
    z=m.match(v[['t','lat','lon','gs']]); z['gs']=v.gs.values; Z.append(z)
z=pd.concat(Z)
d12=z.distance_m<=12; mg=z.margin_cost>=4
print(apt,'pts',len(z),'dist<=12',round(d12.mean(),3),'margin>=4',round(mg.mean(),3),'ambos',round((d12&mg).mean(),3))
print('margem: quantis',z.margin_cost.replace(np.inf,999).quantile([.1,.25,.5,.75]).round(2).tolist())
print('parado (gs<1): margin>=4',round(mg[z.gs<1].mean(),3),' movendo:',round(mg[z.gs>=3].mean(),3))
# mesmo trecho em sentidos opostos?
u,vv=m.u,m.v
pair={}
for k in range(m.n_arcs): pair[(u[k],vv[k])]=k
twin=np.array([pair.get((vv[k],u[k]),-1) for k in range(m.n_arcs)])
print('arcos com gemeo de sentido oposto',round((twin>=0).mean(),3))
