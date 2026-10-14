import sys, glob, numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import regride as R, link_pista as LP, taxigraph as T
from sincro import Sync
apt='EGLL'; S=os.path.expanduser('~/Projetos/oof_store'); sy=Sync(apt); g=sy.g; Rt=T.Router(g); idx=T.build_stand_index(g.A); KT=0.514444
casos=[('2025-07-30','SHT14L'),('2025-01-07','SHT18Z'),('2025-01-17','BAW48HY'),('2025-01-18','BAW4MJ'),('2025-01-20','AUA45XD'),('2025-01-05','SHT18E')]; EXTRA={}; OUT_NAME='egll_janela.png'
if os.environ.get('CASOS'):
    cc=pd.read_csv(os.environ['CASOS']); casos=list(zip(cc.day,cc.cs)); EXTRA={(a_,b_):(c_,d_) for a_,b_,c_,d_ in zip(cc.day,cc.cs,cc.yh,cc.champ)}; OUT_NAME=os.environ.get('OUT','egll_janela.png')
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt']).rename(columns={'MVT_ID_mvt':'id'})
tr=tr[(tr.ADEP_mvt==apt)&(tr.PHASE_mvt=='DEP')].copy(); tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
def pt(P,k_f):                                   # ponto = (xy, aresta, fracao)
    return dict(xy=P,k=k_f[0],f=k_f[1])
def poly_entre(A,B):
    """polilinha pelo grafo (Dijkstra) entre dois pontos consecutivos do ADS; reta se algum esta no patio (fora do grafo)"""
    if A['k']<0 or B['k']<0: return [A['xy'],B['xy']]
    if A['k']==B['k']: return [A['xy'],B['xy']]
    kA,kB=A['k'],B['k']; best=(np.inf,None)
    for ea_,da in ((sy.ea[kA],A['f']*sy.L[kA]),(sy.eb[kA],(1-A['f'])*sy.L[kA])):
        for eb_,db in ((sy.ea[kB],B['f']*sy.L[kB]),(sy.eb[kB],(1-B['f'])*sy.L[kB])):
            c=da+sy.D[ea_,eb_]+db
            if c<best[0]: best=(c,(ea_,eb_))
    a_,b_=best[1]; nodes=sy.path(int(a_),int(b_)); return [A['xy']]+[g.xy[n] for n in nodes]+[B['xy']]
def poly_no_ate(n,B):                           # do no da pista (MVT) ate o primeiro ponto (ultimo ping)
    xy0=g.xy[n]
    if B['k']<0: return [xy0,B['xy']]
    kB=B['k']; best=(np.inf,None)
    for eb_,db in ((sy.ea[kB],B['f']*sy.L[kB]),(sy.eb[kB],(1-B['f'])*sy.L[kB])):
        c=sy.D[n,eb_]+db
        if c<best[0]: best=(c,eb_)
    nodes=sy.path(int(n),int(best[1])); return [xy0]+[g.xy[m] for m in nodes]+[B['xy']]
D_PB=70.0   # recuo do pushback ao longo do eixo do stand (m): mediana do EHAM (xp ~ -68 m); premissa para o EGLL
def stand_hdg(sp):
    j=int(np.argmin(np.hypot(*(np.array([g.proj(x['lat'],x['lon']) for x in g.A.stands])-sp).T))); return g.A.stands[j]['hdg']
def acesso_taxiway(F):
    """ponto da aresta de taxi mais proximo de F (a frente do stand): (aresta, fracao, xy)"""
    best=(np.inf,None)
    for k in range(len(sy.ea)):
        if sy.kind[k]!='taxi': continue
        A_,B_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; ab=B_-A_; den=ab@ab; f=0. if den==0 else float(np.clip((F-A_)@ab/den,0,1)); Q=A_+f*ab; d=float(np.hypot(*(F-Q)))
        if d<best[0]: best=(d,(k,f,Q))
    return best[1]
def pt(P,k_f):                                   # ponto = (xy, aresta, fracao)
    return dict(xy=P,k=k_f[0],f=k_f[1])
_,seg=LP.prepara(apt); stands=np.array([g.proj(s['lat'],s['lon']) for s in g.A.stands])
fig,ax=plt.subplots(2,3,figsize=(22,14)); ax=ax.ravel()
for A,(day,cs) in zip(ax,casos):
    t0=pd.Timestamp(day,tz='UTC').value//10**9; r=tr[(tr.cs==cs)&(tr.mvt>=t0)&(tr.mvt<t0+86400)].iloc[0]
    # aeroporto por baixo
    for e in g.edges:
        a_,b_=g.xy[e['a']],g.xy[e['b']]; A.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#c3c7cb' if e['kind']=='taxi' else '#6b7177',lw=1.1 if e['kind']=='taxi' else 4,zorder=1,solid_capstyle='round')
    A.scatter(stands[:,0],stands[:,1],s=10,facecolors='none',edgecolors='#b9bec3',lw=.6,zorder=2)
    la,lo,lv=T.map_stand(r.STAND_mvt,idx); sp=np.array(g.proj(la,lo)); rt=Rt.route(la,lo,str(r.RUNWAY_mvt)); pth=g.xy[rt['path']]; anchor=pth[-1]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}; gn={}
    for k,v in grp.items(): gn.setdefault(R.norm(k),[]).append(v)
    q,lig=R.pings(grp,gn,r.cs,r.mvt)
    # confere o icao24 com o aviao que esta acelerando na pista no MVT
    L=LP.liga_dia(apt,day,g,seg,tr[(tr.id==r.id)]); ic_f=L.icao_fis.iloc[0]
    pts=[sp[None],pth]; txt='SEM cadeia ate o MVT: ancora na pista'
    q2=q
    hd=stand_hdg(sp); h_=np.radians(90-hd); u=np.array([np.cos(h_),np.sin(h_)]); F=sp-D_PB*u            # F = frente do stand (fim do pushback), no eixo do nariz
    kQ,fQ,Q=acesso_taxiway(F); PQ=pt(Q,(kQ,fQ)); apron=np.vstack([Q,F,sp])                                  # patio: taxiway -> F -> recuo reto pelo eixo ate o stand
    if q is None:
        XYr=np.vstack(poly_no_ate(rt['path'][-1],PQ)); A.plot(XYr[:,0],XYr[:,1],color='#c0392b',lw=2.4,zorder=4,label='SEM pings na janela: Dijkstra pista -> frente do stand')
        pts+=[XYr]; seg_ad=[XYr]; Wp=None
    else:
        P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; t=q.t.to_numpy(); gs=q.gs.to_numpy()*KT
        i,d,D=R.regride(sy,Rt,la,lo,P,t,gs); W=P[i:]; tm=(r.mvt-t[i:])/60; Wp=W
        sn=sy.snap(W); pts_=[pt(W[j],sn[j][:2]) for j in range(len(W))]
        linhas=[poly_no_ate(rt['path'][-1],pts_[-1])]
        for j in range(len(pts_)-1,0,-1): linhas.append(poly_entre(pts_[j],pts_[j-1]))
        XY=np.vstack(linhas); A.plot(XY[:,0],XY[:,1],color='#c0392b',lw=2.4,zorder=4,label='interpolacao: MVT -> ping -> ping anterior ... (Dijkstra a cada ADS)')
        sc=A.scatter(W[:,0],W[:,1],c=tm,cmap='plasma_r',s=30,zorder=6,edgecolors='k',linewidths=.3); cb=plt.colorbar(sc,ax=A,fraction=.035); cb.set_label('minutos antes do MVT (so pings da janela)')
        A.scatter(*P[i],s=320,facecolors='none',edgecolors='#0b5394',lw=2.6,zorder=7,label=f'ultimo ponto ADS (mais perto do stand): {d:.0f} m')
        e=sy.snap(P[i:i+1])[0]
        XY2=np.vstack(poly_entre(pt(P[i],e[:2]),PQ)) if e[0]>=0 else np.vstack([P[i],Q])
        A.plot(XY2[:,0],XY2[:,1],color='#0b5394',lw=3,ls='--',zorder=5,label='Dijkstra: ultimo ponto ADS -> acesso da taxiway em frente ao stand'); pts+=[XY2,W,XY]; seg_ad=[XY,XY2]
        txt=f"janela: {len(W)} pings, de {tm.max():.0f} a {tm.min():.0f} min antes do MVT"
    A.plot(apron[:,0],apron[:,1],color='#e67e22',lw=3.2,zorder=7,label=f'SIMULACAO DE PATIO: taxiway -> frente do stand -> recuo pelo eixo ({D_PB:.0f} m)'); A.scatter(*F,marker='D',s=70,color='#e67e22',edgecolors='k',zorder=8)
    A.add_patch(plt.Circle(sp,150,fill=False,ec='#e67e22',lw=1.8,ls=':',zorder=3)); A.scatter(*sp,marker='*',s=420,color='#f1c40f',edgecolors='k',zorder=9,label='stand | circulo 150 m: inicio da simulacao de patio')
    A.scatter(*anchor,marker='s',s=170,color='#111',zorder=9,label='MVT (pista)')
    Ps=np.vstack(pts); lo_,hi_=Ps.min(0),Ps.max(0); c0=(lo_+hi_)/2; span=max(hi_[0]-lo_[0],hi_[1]-lo_[1],900)/2*1.15
    A.set_xlim(c0[0]-span,c0[0]+span); A.set_ylim(c0[1]-span,c0[1]+span); A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([])
    chk='confere' if (q is not None and ic_f is not None and ic_f==q.icao24.iloc[-1]) else ('NAO confere' if (q is not None and ic_f is not None) else 'sem aviao acelerando na pista no MVT')
    ins=A.inset_axes([0.015,0.60,0.36,0.36]); z=260
    for e in g.edges:
        a_,b_=g.xy[e['a']],g.xy[e['b']]
        if min(a_[0],b_[0])<sp[0]+z and max(a_[0],b_[0])>sp[0]-z and min(a_[1],b_[1])<sp[1]+z and max(a_[1],b_[1])>sp[1]-z: ins.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#c3c7cb' if e['kind']=='taxi' else '#6b7177',lw=1.2 if e['kind']=='taxi' else 4,zorder=1)
    ins.scatter(stands[:,0],stands[:,1],s=14,facecolors='none',edgecolors='#aab0b5',lw=.7,zorder=2)
    for sg_ in seg_ad: ins.plot(sg_[:,0],sg_[:,1],color='#c0392b' if sg_ is seg_ad[0] else '#0b5394',lw=2.2,zorder=4)
    if Wp is not None: ins.scatter(Wp[:,0],Wp[:,1],c='#6a1b9a',s=16,zorder=6,edgecolors='k',linewidths=.2)
    ins.plot(apron[:,0],apron[:,1],color='#e67e22',lw=3.4,zorder=7); ins.scatter(*F,marker='D',s=60,color='#e67e22',edgecolors='k',zorder=8); ins.scatter(*sp,marker='*',s=300,color='#f1c40f',edgecolors='k',zorder=9)
    ins.annotate('',xy=sp,xytext=sp-u*0.001,arrowprops=None); ins.set_xlim(sp[0]-z,sp[0]+z); ins.set_ylim(sp[1]-z,sp[1]+z); ins.set_aspect('equal'); ins.set_xticks([]); ins.set_yticks([]); ins.set_title(f'zoom: patio (eixo do stand {hd:.0f} graus)',fontsize=7.5)
    ex=EXTRA.get((day,cs)); extra_t=f" | grafo {ex[0]/60:.0f} min, campeao {ex[1]/60:.0f} min" if ex else ''
    A.set_title(f"{day} {cs} stand {r.STAND_mvt} pista {r.RUNWAY_mvt} | y REAL {(r.mvt-r.blk)/60:.0f} min{extra_t}\n{txt}",fontsize=9.5); A.legend(loc='lower left',fontsize=7.2,framealpha=.92)
plt.tight_layout(); plt.savefig(f'{S}/sincronia_grafo/{OUT_NAME}',dpi=100)
