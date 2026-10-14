"""Plot 3D: BAW105 (EGLL 27R -> OMDB, 2025-01-17) decolando e encaixando no gate da janela de CTOT (-5/+10 min) no meio do ceu, com o trafego real ao redor (ADS-B)."""
import numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
la0,lo0=51.4706,-0.4619; KX=111.32*np.cos(np.radians(la0)); KY=110.57
o=pd.read_parquet('trk/trf_2025-01-17.parquet'); o=o[o.apt=='EGLL'].copy(); o['x']=(o.lon-lo0)*KX; o['y']=(o.lat-la0)*KY; o=o[o.gs.fillna(200)>=90]
own=o[o.icao=='4005ba'].sort_values('t'); t0=own.t.min(); M=1737155155.0   # MVT do voo (23:05:55 UTC)
own=own[(own.t>=t0)&(own.t<=t0+400)]; own['tm']=(own.t-t0)/60
oth=o[(o.icao!='4005ba')&(o.t>=t0-240)&(o.t<=t0+480)&(np.hypot(o.x,o.y)<40)&(o.alt<14000)]
fig=plt.figure(figsize=(12,8.5)); ax=fig.add_subplot(111,projection='3d')
for ic,g in oth.groupby('icao'):
    g=g.sort_values('t'); 
    if len(g)>8: ax.plot(g.x,g.y,g.alt/1000,color='0.65',lw=.7,alpha=.7)
sc=ax.scatter(own.x,own.y,own.alt/1000,c=own.tm,cmap='viridis',s=9,depthshade=False); ax.plot(own.x,own.y,own.alt/1000,color='k',lw=1.2,alpha=.6)
cb=fig.colorbar(sc,ax=ax,shrink=.55,pad=.02); cb.set_label('minutos desde a decolagem (BAW105)')
# gate: moldura retangular perpendicular a trajetoria, no ponto em que a aeronave esta a ~2,5 min da decolagem (meio do ceu)
k=np.argmin(np.abs(own.tm.values-2.5)); p=np.array([own.x.values[k],own.y.values[k],own.alt.values[k]/1000]); d=np.array([own.x.values[min(k+3,len(own)-1)]-own.x.values[max(k-3,0)],own.y.values[min(k+3,len(own)-1)]-own.y.values[max(k-3,0)]]); d=d/np.hypot(*d); n=np.array([-d[1],d[0]])
W,H=11.0,4.5; n3=np.array([n[0],n[1],0.0]); up=np.array([0,0,H/2]); c=[p+n3*W/2+up,p-n3*W/2+up,p-n3*W/2-up,p+n3*W/2-up]
ax.add_collection3d(Poly3DCollection([c],facecolor=(0.9,0.2,0.2,0.18),edgecolor=(0.8,0.1,0.1,1),linewidths=2.4))
for i in range(4): ax.plot(*zip(c[i],c[(i+1)%4]),color=(0.8,0.1,0.1),lw=2.4)
ax.text(p[0],p[1],p[2]+H/2+1.3,'GATE: janela de CTOT [CTOT−5, CTOT+10] min\nETOT = EOBT + EXOT(grafo)  |  CTOT ∈ [MVT−10, MVT+5]',color=(0.7,0,0),fontsize=10,ha='center')
ax.scatter([0],[0],[0],color='r',s=40); ax.text(0,0,0.2,'cabeceira 27R (EGLL)',fontsize=9)
ax.set_xlim(-35,35); ax.set_ylim(-35,35); ax.set_zlim(0,14); ax.set_xlabel('leste (km)'); ax.set_ylabel('norte (km)'); ax.set_zlabel('altitude (mil ft)'); ax.view_init(elev=16,azim=-62)
ax.set_title('BAW105 · EGLL→OMDB · 2025-01-17 — decolagem encaixando no gate da janela de CTOT\n(trajetória real ADS-B em cores; tráfego vizinho em cinza)',fontsize=12)
plt.tight_layout(); plt.savefig('repo_assets/ctot_gate_3d.png',dpi=130); print('ok',own.shape,oth.icao.nunique())
