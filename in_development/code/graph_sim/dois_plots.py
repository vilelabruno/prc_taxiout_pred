import sys,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_dia as SD, sim_run as SR, foco as F, multi as M, multi_u as MU, warnings; warnings.filterwarnings('ignore')
S='/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo'; DIA,CS=sys.argv[1],sys.argv[2]; X=float(sys.argv[3])*1000; T=float(sys.argv[4]); GAM=float(sys.argv[5])
d=SD.Dia('EHAM',DIA); D,A,Oobs,own=SR.prepara(d); g=d.g; sy=d.sy; pr=dict(F.P_DEF)
j=[i for i,r in enumerate(D) if r['cs']==CS and r['chain'] and r['obs'] is not None][0]; r=D[j]; ref=D[F.referencia(d,D,j)]
um,umed,nn=MU.unidade(d); print('UNIDADE = media da distancia entre dois pings consecutivos do ADS-B no aeroporto: %.1f m (mediana %.1f m, %d pares)'%(um,umed,nn))
V=d.a.gs.to_numpy()*SD.KT; ok=(d.a.k.to_numpy()>=0)&(V>1)&(V<15); vref=float(np.quantile(V[ok],0.5)); print('velocidade livre: %.1f m/s -> um passo de %.1f s por unidade | rastro: alcance X = %.1f km em todas as direcoes do grafo, dura T = %.0f s, gamma = %s'%(vref,um/vref,X/1000,T,GAM))
def poly(P,rwy):
    k,f=sy.snap_many(P); D_,P_,Du,Pu=sy.matriz(rwy); L=[P[0:1]]
    for i in range(len(P)-1):
        if k[i]<0 or k[i+1]<0 or k[i]==k[i+1]: L.append(P[i+1:i+2]); continue
        best=(np.inf,None)
        for ea_,da in ((sy.ea[k[i]],f[i]*sy.L[k[i]]),(sy.eb[k[i]],(1-f[i])*sy.L[k[i]])):
            for eb_,db in ((sy.ea[k[i+1]],f[i+1]*sy.L[k[i+1]]),(sy.eb[k[i+1]],(1-f[i+1])*sy.L[k[i+1]])):
                c=da+Du[ea_,eb_]+db
                if c<best[0]: best=(c,(ea_,eb_))
        if best[1] is None or not np.isfinite(best[0]): L.append(P[i+1:i+2]); continue
        a,b=int(best[1][0]),int(best[1][1]); cur=b; path=[cur]
        while cur!=a and Pu[a,cur]>=0: cur=int(Pu[a,cur]); path.append(cur)
        L.append(np.vstack([g.xy[n] for n in path[::-1]]+[P[i+1]]))
    return np.vstack(L)
t_real=r['tstar']; t_alvo=r['mvt']-SD.ROLL_DEP
PD=r['pings']; segsD=F.segs_de_pings(d,r['rwy'],PD); gD=np.hypot(*np.diff(PD,axis=0).T)/np.maximum(np.diff(r['tp']),1e-6); vD=float(np.quantile(gD[(gD>1)&(gD<15)],.5)); PLD=poly(PD,r['rwy'])
PR=ref['pings']; iS=int(np.argmin(np.hypot(*(PR-r['xy']).T))); segsR=F.segs_de_pings(d,r['rwy'],PR[iS:]); gR=np.hypot(*np.diff(PR,axis=0).T)/np.maximum(np.diff(ref['tp']),1e-6); vR=float(np.quantile(gR[(gR>1)&(gR<15)],.5)); PLR=np.vstack([r['xy'][None],poly(PR[iS:],r['rwy'])])
kS,fS=sy.snap_many(PR[iS:iS+1])
if kS[0]>=0: c,ed=d.caminho(r['rwy'],d.stand_info(r['stand'])['opts'],to_edge=(kS[0],fS[0])); segsR=ed+segsR
VAR={'R':('rota recente (voo %s) como referencia, ate o meu stand'%ref['cs'],segsR,vR,PLR,'#1e8449'),'D':('Dijkstra ligando cada ponto do ADS ao seguinte (rota real do voo)',segsD,vD,PLD,'#1f618d')}
t_a=t_real-600; t_b=t_alvo+60; others=M.monta_agentes(d,D,A,t_a,t_b,vref,foco=r); print('avioes simulados juntos: %d outros + o voo em foco'%len(others))
t_ini=min([a['ini'] for a in others]+[t_real])-30; t_fim=t_alvo+180; t_snap=t_real+0.5*(t_alvo-t_real)
for nome,(desc,segs,v,PL,cor) in VAR.items():
    Ls=np.array([s_[1] for s_ in segs],float); foc=dict(tipo='foco',id=r['id'],cs=r['cs'],segs=segs,Ls=Ls,ks=np.array([s_[0] for s_ in segs]),ini=t_real,fim_real=t_alvo,PL=PL,v=v,foco=True); ag=others+[foc]
    i=len(ag)-1; alvo=t_alvo-t_real
    def dur(g_): rr_=MU.simular_u(d,ag,g_,X,T,um,vref,t_ini,t_fim+1500); return (rr_['chegada'][i]-t_real if np.isfinite(rr_['chegada'][i]) else 1e9),rr_
    d0,_=dur(0.0); lo,hi=0.0,GAM
    while dur(hi)[0]<alvo and hi<20: lo=hi; hi*=2
    if d0>=alvo: g_f=0.0
    else:
        for _ in range(11):
            mid=(lo+hi)/2
            if dur(mid)[0]<alvo: lo=mid
            else: hi=mid
        g_f=(lo+hi)/2
    dd,res=dur(g_f); print('  sim %s: sem rastros %.0f s -> gamma ajustado = %.3f faz o voo em foco levar %.0f s (real %.0f s)'%(nome,d0,g_f,dd,alvo)); GAMF=g_f
    s=int((t_snap-res['t'][0])/res['dt']); I=MU.campo(d,res,s,X,T); fig,A_=plt.subplots(figsize=(15,15))
    for e in g.edges: a_,b_=g.xy[e['a']],g.xy[e['b']]; A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#dfe1e4' if e['kind']=='taxi' else '#b5babf',lw=1 if e['kind']=='taxi' else 4,zorder=1)
    imax=max(I.max(),1e-6)
    for k in np.argsort(I):
        if I[k]<0.03: continue
        a_,b_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; q=min(I[k]/imax,1); A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color=plt.cm.YlOrRd(0.15+0.85*q),lw=2+6*q,alpha=.85,zorder=2,solid_capstyle='round')
    A_.plot(PL[:,0],PL[:,1],color=cor,lw=5,zorder=5,solid_capstyle='round'); A_.scatter(*PL[0],marker='*',s=700,color='#f1c40f',edgecolors='k',zorder=8); A_.scatter(*PL[-1],marker='s',s=330,color='k',zorder=8)
    P=res['XY'][s]; ok_=np.isfinite(P[:-1,0]); A_.scatter(P[:-1][ok_,0],P[:-1][ok_,1],s=55,color='k',zorder=7)
    A_.plot([],[],color=cor,lw=5,label='rota: da pista (quadrado) ao stand (estrela)'); A_.plot([],[],color='#e74c3c',lw=5,label=f'rastro emitido pelos outros avioes (todas as direcoes do grafo, ate {X/1000:.1f} km; mais escuro = mais forte)'); A_.scatter([],[],s=55,color='k',label='outros avioes taxiando (origem dos rastros)'); A_.legend(loc='lower right',fontsize=12)
    mx=150; A_.set_xlim(g.xy[:,0].min()-mx,g.xy[:,0].max()+mx); A_.set_ylim(g.xy[:,1].min()-mx,g.xy[:,1].max()+mx); A_.set_aspect('equal'); A_.set_xticks([]); A_.set_yticks([])
    A_.set_title(f'{DIA} {CS}: {desc} | unidade = {um:.0f} m (media entre dois pontos ADS) | instante {(t_snap-t_alvo)/60:+.1f} min do MVT | gamma AJUSTADO a este voo = {GAMF:.2f}',fontsize=12); plt.tight_layout(); plt.savefig(f'{S}/dois_{nome}.png',dpi=80); plt.close()
