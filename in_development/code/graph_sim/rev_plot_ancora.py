import sys,os,pickle,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_campo as SC, multi as M
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); DAY,CS=sys.argv[1],sys.argv[2]; G=SC.graf(); sy=G['sy']; g=sy.g; Dia=pickle.load(open(f'{SC.R0}/{DAY}.pkl','rb')); r=[x for x in Dia['D'] if x['cs']==CS][0]
S=pickle.load(open(f'{OUT}/solver_rev_{DAY}.pkl','rb')); T=S['T']; T['dur']=S['dur']; t=T[T.cs==CS].iloc[0]
Gd=SC.pickle.load(open(f'{SC.R0}/_guias.pkl','rb')); gd=Gd.get((DAY,[i for i,x in enumerate(Dia['D']) if x['cs']==CS][0]))
ed=gd['ed_guide'] if gd is not None else (r['ed_h'] if r['chain'] and r['ed_h'] else r['ed']); Ls=np.array([e[1] for e in ed]); tot=Ls.sum()
print('trecho SIMULADO (stand -> P*): %d arestas, %.0f m (distancia reta stand->P* = %.0f m) | duracao simulada %.0f s -> velocidade media %.1f m/s (livre = %.1f m/s)'%(len(ed),tot,np.hypot(*(r['Pstar']-r['xy'])),t.dur,tot/max(t.dur,1e-9),SC.VREF))
P=r['pings']; tp=r['tp']; dt=np.diff(tp); rwy=r['rwy']; D_,P_,Du,Pu=sy.matriz(rwy)
def nodes(a,b):
    cur=b; out=[cur]
    while cur!=a and Pu[a,cur]>=0: cur=int(Pu[a,cur]); out.append(cur)
    return out[::-1]
k,f=sy.snap_many(P); segs=[]; gaps=[]
for i in range(len(P)-1):
    if k[i]>=0 and k[i+1]>=0 and k[i]!=k[i+1]:
        best=(np.inf,None)
        for ea_,da in ((sy.ea[k[i]],f[i]*sy.L[k[i]]),(sy.eb[k[i]],(1-f[i])*sy.L[k[i]])):
            for eb_,db in ((sy.ea[k[i+1]],f[i+1]*sy.L[k[i+1]]),(sy.eb[k[i+1]],(1-f[i+1])*sy.L[k[i+1]])):
                c=da+Du[ea_,eb_]+db
                if c<best[0]: best=(c,(ea_,eb_))
        pl=np.vstack([P[i]]+[g.xy[n] for n in nodes(int(best[1][0]),int(best[1][1]))]+[P[i+1]])
    else: pl=np.vstack([P[i],P[i+1]])
    (gaps if dt[i]>60 else segs).append((pl,dt[i]))
fig,A_=plt.subplots(figsize=(15,15))
for e in g.edges: a_,b_=g.xy[e['a']],g.xy[e['b']]; A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#e3e5e8' if e['kind']=='taxi' else '#b5babf',lw=1 if e['kind']=='taxi' else 4,zorder=1)
for pl,_ in segs: A_.plot(pl[:,0],pl[:,1],color='k',lw=2.5,zorder=4)
for pl,d_ in gaps: A_.plot(pl[:,0],pl[:,1],color='#c0392b',lw=3,ls='--',zorder=4)
A_.scatter(P[:,0],P[:,1],s=45,color='k',zorder=5)
ed_xy=M.poly_ed(type('N',(),dict(sy=sy,g=g))(),ed,r['xy']); A_.plot(ed_xy[:,0],ed_xy[:,1],color='#1f618d',lw=6,zorder=6,label='trecho SIMULADO: stand -> P* (%.0f m, %.0f s)'%(tot,t.dur))
A_.scatter(*r['xy'],marker='*',s=900,color='#f1c40f',edgecolors='k',zorder=9,label='stand'); A_.scatter(*r['Pstar'],marker='s',s=330,color='k',zorder=9,label='ANCORA ADS-B = P* (ultimo ponto mais perto do stand)'); A_.scatter(*P[-1],marker='D',s=330,color='#27ae60',edgecolors='k',zorder=9,label='MVT (ultimo ping, na pista)')
A_.plot([],[],color='k',lw=2.5,label='ADS-B OBSERVADO P* -> MVT (%.0f s), pings ligados por Dijkstra'%(r['mvt']-r['tstar'])); A_.plot([],[],color='#c0392b',lw=3,ls='--',label='lacuna do ADS-B: %.0f s sem ping, salto de %.0f m'%(max(d_ for _,d_ in gaps) if gaps else 0,np.hypot(*(P[1]-P[0])) if False else max(np.hypot(*(a[-1]-a[0])) for a,_ in gaps) if gaps else 0)); A_.legend(loc='lower right',fontsize=11)
mx=150; A_.set_xlim(g.xy[:,0].min()-mx,g.xy[:,0].max()+mx); A_.set_ylim(g.xy[:,1].min()-mx,g.xy[:,1].max()+mx); A_.set_aspect('equal'); A_.set_xticks([]); A_.set_yticks([])
A_.set_title('%s %s | y real %.0f s = (MVT - ancora) %.0f + patio %.0f + simulacao %.0f + espera que sobra %.0f'%(DAY,CS,t.y,r['mvt']-r['tstar'],r['t_patio'],t.dur,t.y-t.base-t.dur),fontsize=12); plt.tight_layout(); plt.savefig(f'{OUT}/rev_ancora_{CS}.png',dpi=80); print('plot salvo')
