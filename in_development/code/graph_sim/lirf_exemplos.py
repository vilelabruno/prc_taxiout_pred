import sys, numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/physical'); import taxigraph as T
S=os.path.expanduser('~/Projetos/oof_store'); apt='LIRF'
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/LIRF/cenario_b_oof.parquet')
pa=pd.read_parquet(f'{S}/defeitos_prio/pi_A.parquet'); fl=[c for c in pa.columns if c.startswith('flag90_')]; pa['any90']=pa[fl].any(axis=1)
o=o.merge(pa[['MVT_ID_mvt','any90']].rename(columns={'MVT_ID_mvt':'id'}),on='id',how='left'); o['any90']=o.any90.fillna(False)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt','CALLSIGN_flt','FLIGHT_mvt'])
o=o.merge(tr.rename(columns={'MVT_ID_mvt':'id'}),on='id',how='left')
o['mvt']=pd.to_datetime(o.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); o['blk']=pd.to_datetime(o.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
o['hora']=pd.to_datetime(o.mvt,unit='s').dt.hour; o['cs']=o.CALLSIGN_flt.fillna(o.FLIGHT_mvt).str.strip()
d=o[(~o.any90)&(o.hora>=6)&(o.y<21600)].copy(); d['pred']=d.b_relogio_mais_grafo; d['e']=d.y-d.pred
w=d.reindex(d.e.abs().sort_values(ascending=False).index).head(60)
g=T.TaxiGraph(apt); Rt=T.Router(g); idx=T.build_stand_index(g.A)
cache={}; sel=[]
for r in w.itertuples():
    if r.day not in cache:
        p=f'data/processed/adsb_surface/{r.day}.parquet'
        cache[r.day]=pd.read_parquet(p,filters=[('apt','==',apt)]) if os.path.exists(p) else None
    a=cache[r.day]
    if a is None or not isinstance(r.cs,str): continue
    a=a[(a.callsign.str.strip()==r.cs)&(a.t>=r.blk-900)&(a.t<=r.mvt+60)&(a.alt==0)]
    if len(a)>=5: sel.append((r,a.sort_values('t')))
print('piores com ADS-B de solo:',len(sel),'de 60')
sel=sel[:6]
fig,ax=plt.subplots(2,3,figsize=(21,14)); ax=ax.ravel()
for A,(r,a) in zip(ax,sel):
    la,lo,lv=T.map_stand(r.STAND_mvt,idx); sp=np.array(g.proj(la,lo))
    P=np.c_[(a.lon-g.lon0)*g.kx,(a.lat-g.lat0)*g.ky]; t=a.t.to_numpy(); gs=a.gs.to_numpy()
    for e in g.edges: 
        p,q=g.xy[e['a']],g.xy[e['b']]; A.plot([p[0],q[0]],[p[1],q[1]],color='#c5c9cd' if e['kind']=='taxi' else '#7a8086',lw=1 if e['kind']=='taxi' else 3,zorder=1)
    rt=Rt.route(la,lo,str(r.RUNWAY_mvt)) if lv<9 else None
    if rt: 
        pth=g.xy[rt['path']]; A.plot(pth[:,0],pth[:,1],color='#c0392b',lw=3,zorder=3,label=f"rota do grafo ({rt['len_full']:.0f} m, {rt['n_turns']} curvas)")
    sc=A.scatter(P[:,0],P[:,1],c=(t-r.blk)/60,cmap='viridis',s=26,zorder=4,edgecolors='k',linewidths=.25)
    A.scatter(*sp,marker='*',s=320,color='#f1c40f',edgecolors='k',zorder=6,label=f'stand {r.STAND_mvt}')
    dt=np.diff(t,prepend=t[0]); parado=(gs<1)&(np.hypot(*(P-sp).T)>150); tp=dt[parado&(dt<=600)].sum()
    dist=np.hypot(*(P-sp).T); tz=(t[dist>100][0]-r.blk) if (dist>100).any() else np.nan
    A.set_title(f"{r.day}  {r.cs}  stand {r.STAND_mvt}  pista {r.RUNWAY_mvt}\nreal y={r.y:.0f} s ({r.y/60:.0f} min) | grafo previu {r.pred:.0f} s | campeao {r.champ:.0f} s\npings {len(a)} | parado fora do stand (>150 m): {tp/60:.0f} min | saiu do stand (>100 m) em BLOCK{(tz/60 if tz==tz else float('nan')):+.0f} min",fontsize=10)
    xs=np.r_[P[:,0],sp[0]]; ys=np.r_[P[:,1],sp[1]]; m=500; A.set_xlim(xs.min()-m,xs.max()+m); A.set_ylim(ys.min()-m,ys.max()+m); A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([]); A.legend(loc='lower left',fontsize=8)
    plt.colorbar(sc,ax=A,fraction=.035,label='minutos desde o BLOCK')
plt.tight_layout(); plt.savefig(f'{S}/sincronia_grafo/lirf_exemplos_grafo.png',dpi=100)
