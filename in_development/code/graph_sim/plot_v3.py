import sys, glob, numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import regride as R, taxigraph as T
from sincro3 import Sync3
apt='EGLL'; S=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); g=T.TaxiGraph(apt); idx=T.build_stand_index(g.A); KT=0.514444
d=pd.read_parquet(f'{S}/grafo_v3_{apt}.parquet'); d['eg']=d.y-d.yh; d['ec']=d.y-d.champ; d['delta']=d.eg.abs()-d.ec.abs()
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt']).rename(columns={'MVT_ID_mvt':'id'}); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip(); d=d.merge(tr[['id','cs']],on='id')
sy=Sync3(apt); Rt=T.Router(g)
def caminho_nos(Pm,x,y):
    out=[y]
    while out[-1]!=x and Pm[x,out[-1]]>=0: out.append(int(Pm[x,out[-1]]))
    return out[::-1]
def poly_pings(PP,Pm,Du):
    """interpola entre pings consecutivos PELO GRAFO (taxiways), em vez de reta"""
    k,f=sy.snap_many(PP); L=[PP[0:1]]
    for i in range(len(PP)-1):
        if k[i]<0 or k[i+1]<0 or k[i]==k[i+1]: L.append(PP[i+1:i+2]); continue
        A_,B_=k[i],k[i+1]; best=(np.inf,None)
        for ea_,da in ((sy.ea[A_],f[i]*sy.L[A_]),(sy.eb[A_],(1-f[i])*sy.L[A_])):
            for eb_,db in ((sy.ea[B_],f[i+1]*sy.L[B_]),(sy.eb[B_],(1-f[i+1])*sy.L[B_])):
                c=da+Du[ea_,eb_]+db
                if c<best[0]: best=(c,(ea_,eb_))
        if best[1] is None or not np.isfinite(best[0]): L.append(PP[i+1:i+2]); continue
        nodes=caminho_nos(Pm,int(best[1][0]),int(best[1][1])); L.append(np.vstack([g.xy[n] for n in nodes]+[PP[i+1]]))
    return np.vstack(L)
rg=pd.concat([pd.read_parquet(p) for p in glob.glob(f'{S}/regride/{apt}_*.parquet')]).set_index('id'); tref=pd.read_parquet(f'{S}/trilhas_ref_{apt}.parquet'); SPp=np.array([g.proj(x['lat'],x['lon']) for x in g.A.stands])
w=d.reindex(d.eg.abs().sort_values(ascending=False).index).head(6)
fig,ax=plt.subplots(2,3,figsize=(22,14)); ax=ax.ravel()
for A,r in zip(ax,w.itertuples()):
    for e in g.edges:
        a_,b_=g.xy[e['a']],g.xy[e['b']]; A.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#c3c7cb' if e['kind']=='taxi' else '#6b7177',lw=1.1 if e['kind']=='taxi' else 4,zorder=1)
    A.scatter(SPp[:,0],SPp[:,1],s=10,facecolors='none',edgecolors='#b9bec3',lw=.6,zorder=2)
    la,lo,lv=T.map_stand(r.stand,idx); sp=np.array(g.proj(la,lo)); j=int(np.argmin(np.hypot(*(SPp-sp).T))); h=np.radians(90-g.A.stands[j]['hdg']); u=np.array([np.cos(h),np.sin(h)]); F=sp-70*u
    pts=[sp[None]]; lab=[]
    # rota completa usada (voo de referencia): toda a trilha observada dele, da vizinhanca do stand ate a pista
    if r.fonte>0:
        q=tref[tref.id==r.ref_id].sort_values('tau'); M=sy.matriz(r.rwy); PPq=q[['x','y']].to_numpy(); RT=poly_pings(PPq,M[3],M[2])
        A.plot(RT[:,0],RT[:,1],color='#27ae60',lw=3,zorder=4,label=f'ULTIMA ROTA COMPLETA usada (voo ref. {int(r.ref_id)}, nivel {int(r.fonte)}); interpolada pelo grafo entre os pings'); A.scatter(PPq[::3,0],PPq[::3,1],s=14,color='#145a32',zorder=5); pts.append(RT)
    # janela de ADS-B do proprio voo (regredida desde o MVT)
    if r.tem==1 and r.id in rg.index and np.isfinite(r.px):
        row=rg.loc[r.id]; a=pd.read_parquet(f'data/processed/adsb_surface/{r.day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}; gn={}
        for k,v in grp.items(): gn.setdefault(R.norm(k),[]).append(v)
        qq,_=R.pings(grp,gn,r.cs,r.mvt_s)
        if qq is not None:
            P=np.c_[(qq.lon-g.lon0)*g.kx,(qq.lat-g.lat0)*g.ky][int(row.i_stop):]; tm=(r.mvt_s-qq.t.to_numpy()[int(row.i_stop):])/60
            sc=A.scatter(P[:,0],P[:,1],c=tm,cmap='plasma_r',s=30,zorder=7,edgecolors='k',linewidths=.3); cb=plt.colorbar(sc,ax=A,fraction=.035); cb.set_label('minutos antes do MVT (pings da janela)'); pts.append(P)
            A.scatter(px:=r.px,r.py,s=300,facecolors='none',edgecolors='#0b5394',lw=2.6,zorder=8,label='ultimo ponto ADS')
    if r.fonte==0:                                                                  # sem rota observada: Dijkstra no grafo completo (restricoes de pista)
        try:
            M=sy.matriz(r.rwy); D_,P_,Du_,Pu_=M; opts,_=Rt.attach(la,lo); pair,cands,rl=g.targets(r.rwy); node=min(cands,key=lambda c:c[1])[0]
            nd=min(opts,key=lambda o:o[1]+D_[o[0],node])[0]; nodes=caminho_nos(P_,int(nd),int(node)); XY=np.vstack([sp]+[g.xy[n] for n in nodes]); A.plot(XY[:,0],XY[:,1],color='#c0392b',lw=3,zorder=4,label='SEM rota observada: Dijkstra no grafo completo'); pts.append(XY)
        except Exception as e: print('dijkstra',e)
    if r.tem==0:
        try: pair,cands,rl=g.targets(r.rwy); an=g.xy[min(cands,key=lambda c:c[1])[0]]; A.scatter(*an,marker='s',s=170,color='#111',zorder=9,label='MVT (entrada da pista)'); pts.append(an[None])
        except Exception: pass
    A.plot([F[0],sp[0]],[F[1],sp[1]],color='#e67e22',lw=4,zorder=6,label='patio: frente do stand -> recuo pelo eixo (70 m)'); A.scatter(*F,marker='D',s=70,color='#e67e22',edgecolors='k',zorder=8); A.scatter(*sp,marker='*',s=420,color='#f1c40f',edgecolors='k',zorder=9,label=f'stand {r.stand}')
    Ps=np.vstack(pts); lo_,hi_=Ps.min(0),Ps.max(0); c0=(lo_+hi_)/2; span=max(hi_[0]-lo_[0],hi_[1]-lo_[1],900)/2*1.15; A.set_xlim(c0[0]-span,c0[0]+span); A.set_ylim(c0[1]-span,c0[1]+span); A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([])
    src={0:'SEM rota observada (Dijkstra, grafo completo)',1:'rota do mesmo stand e pista',2:'rota do mesmo stand, outra pista',3:'rota de stand vizinho'}[int(r.fonte)]
    A.set_title(f"{r.day} {r.cs} stand {r.stand} pista {r.rwy} | REAL {r.y/60:.0f} min | grafo {r.yh/60:.0f} | campeao {r.champ/60:.0f} | delta {r.delta/60:+.0f} min\n{src}",fontsize=9.5); A.legend(loc='lower left',fontsize=7.2,framealpha=.92)
plt.tight_layout(); plt.savefig(f'{S}/grafo_v3_piores_rotas.png',dpi=100)
# estatisticas: piores 20 e delta
w20=d.reindex(d.eg.abs().sort_values(ascending=False).index).head(20); fig,ax=plt.subplots(1,3,figsize=(21,6.5)); c1='#c0392b'; c2='#2a6f97'; top=w20.iloc[::-1]; yy=np.arange(20)
ax[0].barh(yy+.2,top.eg.abs()/60,.4,color=c1,label='|erro| do grafo'); ax[0].barh(yy-.2,top.ec.abs()/60,.4,color=c2,label='|erro| do campeao'); ax[0].set_yticks(yy); ax[0].set_yticklabels([f"{x.day[5:]} {x.cs}" for x in top.itertuples()],fontsize=8); ax[0].set_xlabel('minutos'); ax[0].set_title('20 piores erros do grafo v3 e o campeao nos mesmos voos'); ax[0].legend()
ax[1].scatter(d.ec/60,d.eg/60,s=3,alpha=.25,color='#7f8c8d'); ax[1].scatter(w20.ec/60,w20.eg/60,s=30,color=c1,zorder=3); lim=max(abs(d.eg).max(),abs(d.ec).max())/60; ax[1].plot([-lim,lim],[-lim,lim],color='k',lw=.8); ax[1].set_xlabel('erro do campeao (min)'); ax[1].set_ylabel('erro do grafo (min)'); ax[1].set_title('erro do grafo x erro do campeao')
ax[2].hist((d.delta/60).clip(-30,60),bins=140,color=c2); ax[2].set_yscale('log'); ax[2].axvline(0,color='k',lw=.8); ax[2].set_xlabel('delta = |erro grafo| - |erro campeao| (min)'); ax[2].set_title('delta (positivo: o grafo erra mais)')
plt.tight_layout(); plt.savefig(f'{S}/grafo_v3_piores_delta.png',dpi=105)
print('RMSE grafo',round(np.sqrt((d.eg**2).mean()),1),'| campeao',round(np.sqrt((d.ec**2).mean()),1))
print(w[['day','cs','stand','y','yh','champ','delta','fonte','tem']].assign(y=lambda x:x.y/60,yh=lambda x:x.yh/60,champ=lambda x:x.champ/60,delta=lambda x:x.delta/60).round(0).to_string(index=False))
