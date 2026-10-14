import sys,time,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.collections import LineCollection
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_dia as SD, sim_run as SR, foco as F, multi as M, warnings; warnings.filterwarnings('ignore')
S='/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo'; DIA=sys.argv[1]; CS=sys.argv[2]; GAM=float(sys.argv[3]) if len(sys.argv)>3 else 0.01; pr=dict(F.P_DEF)
d=SD.Dia('EHAM',DIA); D,A,Oobs,own=SR.prepara(d); g=d.g; sy=d.sy
j=[i for i,r in enumerate(D) if r['cs']==CS and r['chain'] and r['obs'] is not None][0]; r=D[j]; ref=D[F.referencia(d,D,j)]
V=d.a.gs.to_numpy()*SD.KT; ok=(d.a.k.to_numpy()>=0)&(V>1)&(V<15); VQ={q:float(np.quantile(V[ok],q)) for q in (0.3,0.5,0.7,0.8)}; print('quantis da velocidade do ADS-B nas taxiways (m/s):',{k:round(v,1) for k,v in VQ.items()}); v_free=VQ[0.5]
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
PD=r['pings']; segsD=F.segs_de_pings(d,r['rwy'],PD); gD=np.hypot(*np.diff(PD,axis=0).T)/np.maximum(np.diff(r['tp']),1e-6); vD=float(np.quantile(gD[(gD>1)&(gD<15)],.8)); PLD=poly(PD,r['rwy'])
PR=ref['pings']; iS=int(np.argmin(np.hypot(*(PR-r['xy']).T))); segsR=F.segs_de_pings(d,r['rwy'],PR[iS:]); gR=np.hypot(*np.diff(PR,axis=0).T)/np.maximum(np.diff(ref['tp']),1e-6); vR=float(np.quantile(gR[(gR>1)&(gR<15)],.8)); PLR=np.vstack([r['xy'][None],poly(PR[iS:],r['rwy'])])
kS,fS=sy.snap_many(PR[iS:iS+1])
if kS[0]>=0: c,ed=d.caminho(r['rwy'],d.stand_info(r['stand'])['opts'],to_edge=(kS[0],fS[0])); segsR=ed+segsR
VAR={'R':('rota recente (voo %s, guia ate o MEU stand)'%ref['cs'],segsR,vR,PLR),'D':('Dijkstra + ADS ponto a ponto (rota real do voo)',segsD,vD,PLD)}
t_a=t_real-600; t_b=t_alvo+60
def erro_agentes(vf,g_):
    oth=M.monta_agentes(d,D,A,t_a,t_b,vf,foco=r); ti=min([a['ini'] for a in oth]+[t_real])-30; res=M.simular(d,oth,g_,pr,ti,t_alvo+180); e=[res['chegada'][i]-a['fim_real'] for i,a in enumerate(oth) if np.isfinite(res['chegada'][i])]; return np.mean(np.abs(e)),np.mean(e),len(e)
print('\nCALIBRACAO com TODOS os avioes da janela (erro de chegada simulada - real):')
best=None
for q,vf in VQ.items():
    for g_ in (0.0,0.01,0.03,0.1,0.3,1.0):
        ea,em,n=erro_agentes(vf,g_); print(f'  velocidade livre p{int(q*100)} = {vf:.1f} m/s, gamma={g_}: n={n} | erro absoluto medio {ea:.0f} s | erro medio {em:+.0f} s')
        if best is None or ea<best[0]: best=(ea,q,vf,g_)
print('  -> escolhido: velocidade livre p%d = %.1f m/s, gamma = %s (erro absoluto medio %.0f s)'%(int(best[1]*100),best[2],best[3],best[0])); v_free=best[2]; GAM=best[3]
if len(sys.argv)>4: GAM=float(sys.argv[4]); print('  >> gamma FORCADO para ilustrar o mecanismo: %s (o melhor pela calibracao e %s)'%(GAM,best[3]))
others=M.monta_agentes(d,D,A,t_a,t_b,v_free,foco=r); print('avioes na simulacao (todos simulados, andando juntos): %d outros (%d partidas, %d chegadas) + o voo em foco'%(len(others),sum(a['tipo']=='dep' for a in others),sum(a['tipo']=='arr' for a in others)))
t_ini=min([a['ini'] for a in others]+[t_real])-30; t_fim=t_alvo+180; RES={}
for nome,(desc,segs,v,PL) in VAR.items():
    Ls=np.array([s[1] for s in segs],float); foc=dict(tipo='foco',id=r['id'],cs=r['cs'],segs=segs,Ls=Ls,ks=np.array([s[0] for s in segs]),ini=t_real,fim_real=t_alvo,PL=PL,v=v,foco=True); ag=others+[foc]
    for g_ in (0.0,GAM):
        t0=time.time(); RES[(nome,g_)]=(M.simular(d,ag,g_,pr,t_ini,t_fim),ag); res=RES[(nome,g_)][0]; i=len(ag)-1
        print('  sim %s gamma=%s: voo em foco chega ao fim do caminho em %.0f s apos o inicio (real %.0f s) | erro de chegada %+.0f s | %.0f s de calculo'%(nome,g_,res['chegada'][i]-t_real,t_alvo-t_real,res['chegada'][i]-t_alvo,time.time()-t0),flush=True)
# validacao multiagente: erro de chegada (simulada - real) de todos os avioes com verdade conhecida (chegadas: on-block; partidas com ADS completo: MVT-45 s)
print('\nERRO DE CHEGADA DE TODOS OS AVIOES DA JANELA (simulado - real), por avioes com verdade conhecida:')
for g_ in (0.0,GAM):
    res,ag=RES[('D',g_)]; e=[]; tp=[]
    for i,a in enumerate(ag[:-1]):
        if np.isfinite(res['chegada'][i]): e.append(res['chegada'][i]-a['fim_real']); tp.append(a['tipo'])
    e=np.array(e); tp=np.array(tp); print(f'  gamma={g_}: n={len(e)} | erro medio {e.mean():+.0f} s | erro absoluto medio {np.abs(e).mean():.0f} s | chegadas n={int((tp=="arr").sum())}: |erro| {np.abs(e[tp=="arr"]).mean():.0f} s | partidas n={int((tp=="dep").sum())}: |erro| {np.abs(e[tp=="dep"]).mean():.0f} s')
res0,ag0=RES[('D',GAM)]; vel=res0['SPD']; print('reducao de velocidade pelos rastros na janela: media %.0f%% | max %.0f%% | avioes com reducao >10%%: %d de %d'%(100*(1-np.nanmean(vel)),100*(1-np.nanmin(vel)),int(((np.nanmin(vel,axis=0))<0.9).sum()),len(ag0)))
COR={'R':'#1e8449','D':'#1f618d'}
for nome,(desc,segs,v,PL) in VAR.items():
    res,ag=RES[(nome,GAM)]; ifoco=len(ag)-1; fig,A_=plt.subplots(figsize=(15,15))
    for e in g.edges: a_,b_=g.xy[e['a']],g.xy[e['b']]; A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#e3e5e8' if e['kind']=='taxi' else '#b5babf',lw=1 if e['kind']=='taxi' else 4,zorder=1)
    s0=int((t_real-res['t'][0])/5); s1=int((t_alvo-res['t'][0])/5)
    for i,a in enumerate(ag):
        if i==ifoco: continue
        X=res['XY'][s0:s1+1,i]; X=X[np.isfinite(X[:,0])]
        if len(X)>=2: A_.plot(X[:,0],X[:,1],color='#e67e22' if a['tipo']=='dep' else '#8e44ad',lw=2.0,alpha=.85,zorder=3,solid_capstyle='round')
    A_.plot(PL[:,0],PL[:,1],color=COR[nome],lw=5,zorder=5,solid_capstyle='round'); A_.scatter(*PL[0],marker='*',s=700,color='#f1c40f',edgecolors='k',zorder=8); A_.scatter(*PL[-1],marker='s',s=350,color='k',zorder=8)
    A_.plot([],[],color=COR[nome],lw=6,label='rota: da pista (quadrado) ao stand (estrela)'); A_.plot([],[],color='#e67e22',lw=2.5,label='rastros dos outros avioes: partidas'); A_.plot([],[],color='#8e44ad',lw=2.5,label='rastros dos outros avioes: chegadas'); A_.legend(loc='lower right',fontsize=13)
    mx=150; A_.set_xlim(g.xy[:,0].min()-mx,g.xy[:,0].max()+mx); A_.set_ylim(g.xy[:,1].min()-mx,g.xy[:,1].max()+mx); A_.set_aspect('equal'); A_.set_xticks([]); A_.set_yticks([])
    A_.set_title(f'{DIA} {CS}: {desc}',fontsize=15); plt.tight_layout(); plt.savefig(f'{S}/multi_{nome}.png',dpi=80); plt.close()
