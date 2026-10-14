import sys, re, numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/physical'); import taxigraph as T
S=os.path.expanduser('~/Projetos/oof_store'); apt='EGLL'; KT=0.514444
p=pd.read_parquet(f'{S}/sincronia_grafo/espera_posterior_{apt}.parquet'); p['err']=p.y-p.yhat
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt','CALLSIGN_flt','FLIGHT_mvt']).rename(columns={'MVT_ID_mvt':'id'})
p=p.merge(tr,on='id'); p['mvt']=pd.to_datetime(p.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); p['blk']=pd.to_datetime(p.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); p['cs']=p.CALLSIGN_flt.fillna(p.FLIGHT_mvt).str.strip()
print('n',len(p),'RMSE yhat',round(np.sqrt((p.err**2).mean()),1),'| os 10 maiores |erro| = ',round(((p.err.abs().nlargest(10))**2).sum()/(p.err**2).sum(),3),'do erro quadratico; 100 maiores:',round(((p.err.abs().nlargest(100))**2).sum()/(p.err**2).sum(),3))
print('subestimado (y>yhat):',round(((p.err>0)*p.err**2).sum()/(p.err**2).sum(),3))
norm=lambda c: re.sub(r'(?<=\d)[A-Z]{1,2}$','',c); g=T.TaxiGraph(apt); Rt=T.Router(g); idx=T.build_stand_index(g.A); cache={}
def trilha(r):
    if r.day not in cache: cache[r.day]=pd.read_parquet(f'data/processed/adsb_surface/{r.day}.parquet',filters=[('apt','==',apt)])
    a=cache[r.day]; a=a[(a.alt==0)&(a.t>r.mvt-5400)&(a.t<=r.mvt)]; a=a.assign(cs=a.callsign.str.strip())
    q=a[a.cs==r.cs]
    if len(q)<3: q=a[a.cs.map(lambda c: norm(c) if isinstance(c,str) else '')==norm(r.cs)]
    if len(q)<3: return None
    q=q[q.icao24==q.icao24.iloc[-1]]; return q.sort_values('t')
p['day']=p.d.dt.strftime('%Y-%m-%d')
sub=p.nlargest(3,'err'); sup=p.nsmallest(3,'err'); sel=pd.concat([sub,sup])
fig,ax=plt.subplots(2,3,figsize=(21,14)); ax=ax.ravel()
for A,r in zip(ax,sel.itertuples()):
    q=trilha(r); la,lo,lv=T.map_stand(r.STAND_mvt,idx); sp=np.array(g.proj(la,lo)) if lv<9 else None
    for e in g.edges:
        a_,b_=g.xy[e['a']],g.xy[e['b']]; A.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#c5c9cd' if e['kind']=='taxi' else '#7a8086',lw=1 if e['kind']=='taxi' else 3,zorder=1)
    if q is None: A.set_title(f'{r.day} {r.cs} sem trilha'); continue
    P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; t=q.t.to_numpy(); gs=q.gs.to_numpy()*KT
    rt=Rt.route(la,lo,str(r.RUNWAY_mvt)) if lv<9 else None
    if rt: pth=g.xy[rt['path']]; A.plot(pth[:,0],pth[:,1],color='#c0392b',lw=3,zorder=3,label=f"rota simulada stand->pista ({rt['len_full']:.0f} m)")
    sc=A.scatter(P[:,0],P[:,1],c=(t-r.blk)/60,cmap='viridis',s=26,zorder=4,edgecolors='k',linewidths=.25)
    if sp is not None: A.scatter(*sp,marker='*',s=320,color='#f1c40f',edgecolors='k',zorder=6,label=f'stand {r.STAND_mvt}')
    Hreal=(r.y-(r.mvt-t[0]))/60; dtp=np.diff(t,prepend=t[0]); par=(gs<1)&(dtp<=600); tp=dtp[par].sum()/60
    tipo='SUBESTIMOU' if r.err>0 else 'SUPERESTIMOU'
    A.set_title(f"{tipo}  {r.day}  {r.cs}  stand {r.STAND_mvt}  pista {r.RUNWAY_mvt}\nreal y={r.y/60:.0f} min | simulado {r.yhat/60:.0f} min | campeao {r.champ/60:.0f} min\n1o ping em BLOCK+{(t[0]-r.blk)/60:.0f} min (H real {Hreal:.0f}, simulado {r.Hhat/60:.0f}) | parado visto: {tp:.0f} min | pings {len(q)}",fontsize=10)
    xs=np.r_[P[:,0],sp[0] if sp is not None else P[0,0]]; ys=np.r_[P[:,1],sp[1] if sp is not None else P[0,1]]; m=400; A.set_xlim(xs.min()-m,xs.max()+m); A.set_ylim(ys.min()-m,ys.max()+m); A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([]); A.legend(loc='lower left',fontsize=8)
    plt.colorbar(sc,ax=A,fraction=.035,label='minutos desde o BLOCK')
plt.tight_layout(); plt.savefig(f'{S}/sincronia_grafo/egll_piores_simulacao.png',dpi=100)
print(sel[['day','cs','STAND_mvt','RUNWAY_mvt','y','yhat','champ','Hhat','err']].round(0).to_string(index=False))
