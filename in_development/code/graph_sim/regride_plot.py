import sys, glob, numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import regride as R, taxigraph as T
from sincro import Sync
S=os.path.expanduser('~/Projetos/oof_store'); apt='EGLL'; sy=Sync(apt); g=sy.g; Rt=T.Router(g); idx=T.build_stand_index(g.A); KT=0.514444
ref=pd.concat([pd.read_parquet(p) for p in glob.glob(f'{S}/sincronia_grafo/regride/{apt}_*.parquet')]); VBAR=float(ref.v_bar.median()); C_SEM=807.0
casos=[('2025-07-30','SHT14L'),('2025-01-07','SHT18Z'),('2025-01-17','BAW48HY'),('2025-01-18','BAW4MJ'),('2025-01-20','AUA45XD'),('2025-01-05','SHT18E')]
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt'])
tr=tr[(tr.ADEP_mvt==apt)&(tr.PHASE_mvt=='DEP')].copy(); tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
fig,ax=plt.subplots(2,3,figsize=(22,15)); ax=ax.ravel()
def desenha_grafo(A):
    for e in g.edges:
        a_,b_=g.xy[e['a']],g.xy[e['b']]; A.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#d0d3d6' if e['kind']=='taxi' else '#9aa0a6',lw=1 if e['kind']=='taxi' else 3,zorder=1)
for A,(day,cs) in zip(ax,casos):
    t0=pd.Timestamp(day,tz='UTC').value//10**9; r=tr[(tr.cs==cs)&(tr.mvt>=t0)&(tr.mvt<t0+86400)].iloc[0]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}; gn={}
    for k,v in grp.items(): gn.setdefault(R.norm(k),[]).append(v)
    R.END_GAP=1e9; q,lig=R.pings(grp,gn,r.cs,r.mvt); R.END_GAP=600.0   # pings brutos, para mostrar; a validade e decidida abaixo
    la,lo,lv=T.map_stand(r.STAND_mvt,idx); sp=np.array(g.proj(la,lo)); desenha_grafo(A); pts=[sp]
    rt=Rt.route(la,lo,str(r.RUNWAY_mvt)); pth=g.xy[rt['path']]; Lrota=rt['len_full']
    A.scatter(*sp,marker='*',s=380,color='#f1c40f',edgecolors='k',zorder=8,label=f'stand {r.STAND_mvt}')
    valido=q is not None and (r.mvt-q.t.iloc[-1])<=R.END_GAP
    if valido:
        P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; t=q.t.to_numpy(); gs=q.gs.to_numpy()*KT
        i,d,D=R.regride(sy,Rt,la,lo,P,t,gs)
        A.scatter(P[:i,0],P[:i,1],s=16,color='#9e9e9e',zorder=3,label=f'descartados ({i} pings antes do ponto de parada)')
        sc=A.scatter(P[i:,0],P[i:,1],c=(t[i:]-r.blk)/60,cmap='viridis',s=26,zorder=5,edgecolors='k',linewidths=.25); cb=plt.colorbar(sc,ax=A,fraction=.035); cb.set_label('min desde o BLOCK (pings usados)')
        A.scatter(*P[i],s=300,facecolors='none',edgecolors='#c0392b',lw=2.5,zorder=7,label=f'ponto de parada: {d:.0f} m do stand')
        e=sy.snap(P[i:i+1])[0]
        if e[0]>=0:
            opts,_=Rt.attach(la,lo); Le=sy.L[e[0]]; bc=(np.inf,None)
            for nd,acc in opts:
                for end,off in ((sy.ea[e[0]],e[1]*Le),(sy.eb[e[0]],(1-e[1])*Le)):
                    c=acc+off+sy.D[nd,end]
                    if c<bc[0]: bc=(c,(nd,end))
            path=sy.path(int(bc[1][0]),int(bc[1][1])); XY=np.vstack([sp,g.xy[path],P[i]]); A.plot(XY[:,0],XY[:,1],color='#c0392b',lw=2.5,ls='--',zorder=6,label='Dijkstra: ponto de parada -> stand')
        pts+= [P[i:],P[:i] if i>0 else P[i:i+1]]
        sim=(t[i]-D_ if False else 0)
        tit2=f"cadeia valida: parada em BLOCK{(t[i]-r.blk)/60:+.0f} min, {(r.mvt-t[i])/60:.0f} min antes do MVT | {len(P)-i} pings usados de {len(P)}"
    else:
        if q is not None:
            P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; A.scatter(P[:,0],P[:,1],s=18,color='#9e9e9e',zorder=3,label=f'pings de solo SEM cadeia ate o MVT ({len(P)}; ultimo a {(r.mvt-q.t.iloc[-1])/60:.0f} min do MVT)'); pts.append(P)
        A.plot(pth[:,0],pth[:,1],color='#c0392b',lw=3,zorder=6,label=f'ancora na pista: Dijkstra stand->pista ({Lrota:.0f} m)')
        A.scatter(*pth[-1],marker='s',s=160,color='#2c3e50',zorder=8,label='ponto inicial: pista (MVT)')
        tit2=f"REJEITADO (nenhum ping a ate 10 min do MVT) -> ancora na pista: rota {Lrota:.0f} m / {VBAR:.1f} m/s = {Lrota/VBAR/60:.0f} min + constante {C_SEM/60:.0f} min = {(Lrota/VBAR+C_SEM)/60:.0f} min"
    pts.append(pth); pts=np.vstack([np.atleast_2d(x) for x in pts]); lo_,hi_=pts.min(0),pts.max(0); cx,cy=(lo_+hi_)/2; span=max(hi_[0]-lo_[0],hi_[1]-lo_[1],600)/2*1.12
    A.set_xlim(cx-span,cx+span); A.set_ylim(cy-span,cy+span); A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([])
    A.set_title(f"{day} {cs} stand {r.STAND_mvt} pista {r.RUNWAY_mvt} | y real {(r.mvt-r.blk)/60:.0f} min\n{tit2}",fontsize=9.5); A.legend(loc='lower left',fontsize=7.5,framealpha=.9)
plt.tight_layout(); plt.savefig(f'{S}/sincronia_grafo/egll_regride.png',dpi=100)
