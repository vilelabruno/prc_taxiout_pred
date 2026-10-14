import sys,numpy as np,pandas as pd,warnings,os,matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_dia as SD, sim_run as SR, foco as F, taxigraph as T
S='/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo'; GAM=0.01; DIA=sys.argv[1]; pr=dict(F.P_DEF)
d=SD.Dia('EHAM',DIA); D,A,Oobs,own=SR.prepara(d); g=d.g; sy=d.sy
cand=[j for j in F.candidatos(d,D,A) if F.referencia(d,D,j) is not None]; rng=np.random.default_rng(3); ordem=list(rng.permutation(cand))
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
def sobre(PL,frac):
    c=np.r_[0,np.cumsum(np.hypot(*np.diff(PL,axis=0).T))]; x=np.clip(frac,0,1)*c[-1]; i=min(int(np.searchsorted(c,x)),len(c)-1); i=max(i,1); w=(x-c[i-1])/max(c[i]-c[i-1],1e-9); return PL[i-1]+w*(PL[i]-PL[i-1])
# escolhe o voo em foco: o que tem mais avioes na janela (entre os candidatos sorteados)
best=None
for j in ordem[:14]:
    r=D[j]; dep=F.deposito(d,D,A,r['tstar']-1500,r['mvt'],r['icao'],r['id'],pr['wads']); n=dep['n_obs']+dep['n_sim']
    if best is None or n>best[0]: best=(n,j,dep)
n_out,j,dep=best; r=D[j]; ref=D[F.referencia(d,D,j)]; print('voo em foco:',r['cs'],'stand',r['stand'],'pista',r['rwy'],'| avioes na janela:',n_out,'(observados no ADS-B %d, simulados %d)'%(dep['n_obs'],dep['n_sim']),'| rota recente: voo',ref['cs'],'(mesmo stand e pista: %s)'%(ref['stand']==r['stand'] and ref['rwy']==r['rwy']))
t_real=r['tstar']; t_alvo=r['mvt']-SD.ROLL_DEP
PD=r['pings']; segsD=F.segs_de_pings(d,r['rwy'],PD); gsD=np.hypot(*np.diff(PD,axis=0).T)/np.maximum(np.diff(r['tp']),1e-6); vD=float(np.quantile(gsD[(gsD>1)&(gsD<15)],.8)); PLD=poly(PD,r['rwy'])
PR=ref['pings']; iS=int(np.argmin(np.hypot(*(PR-r['xy']).T))); segsR=F.segs_de_pings(d,r['rwy'],PR[iS:]); gsR=np.hypot(*np.diff(PR,axis=0).T)/np.maximum(np.diff(ref['tp']),1e-6); vR=float(np.quantile(gsR[(gsR>1)&(gsR<15)],.8)); PLR=np.vstack([r['xy'][None],poly(PR[iS:],r['rwy'])])
kS,fS=sy.snap_many(PR[iS:iS+1])
if kS[0]>=0: c,ed=d.caminho(r['rwy'],d.stand_info(r['stand'])['opts'],to_edge=(kS[0],fS[0])); segsR=ed+segsR
sim={}
for nome,segs,v in (('D',segsD,vD),('R',segsR,vR)):
    for g_ in (0.0,GAM):
        s0=F.prever_inicio(segs,v,g_,dep,d.dEE,t_alvo,t_real,pr); tf,trj=F.avanca(segs,v,g_,dep,d.dEE,s0,pr,gravar=True); sim[(nome,g_)]=(s0,trj,sum(x[1] for x in segs))
print('inicio real do movimento (s relativos ao MVT-45s): %.0f | duracao real %.0f s'%(t_real-t_alvo,t_alvo-t_real))
for k,(s0,trj,Lt) in sim.items(): print('  sim %s gamma=%s: inicio previsto erro %+.0f s | duracao simulada %.0f s | caminho %.0f m'%(k[0],k[1],s0-t_real,t_alvo-s0,Lt))
# --- quem influencia o voo em foco (todos os avioes impactam todos?)
infl=set(); tot_obs=set()
for (nome,g_),(s0,trj,Lt) in sim.items():
    if g_==0: continue
    for (t,pos,k,rh,v) in trj:
        if k<0: continue
        m=(dep['t']<=t)&(t-dep['t']<=pr['T']); dtm=t-dep['t'][m]; Rr=np.minimum(pr['R0']+pr['vexp']*dtm/60,pr['Rmax']); K=np.clip(1-d.dEE[int(k),dep['ek'][m]]/Rr,0,None)*(1-dtm/pr['T']); infl|=set(dep['ag'][m][K>0])
n_ag=len(set(dep['ag'])); print('avioes na janela:',n_ag,'| avioes cujo rastro chegou a influenciar o voo em algum instante:',len(infl),'(%.0f%%)'%(100*len(infl)/max(n_ag,1)))
rho_max=max(trj[:,3].max() for (nome,g_),(s0,trj,Lt) in sim.items() if g_>0); rho_med=np.mean([trj[:,3].mean() for (nome,g_),(s0,trj,Lt) in sim.items() if g_>0]); print('exposicao ao rastro dos outros (rho) ao longo do taxi: media %.1f, maxima %.1f -> velocidade reduzida em ate %.0f%% (gamma=%s)'%(rho_med,rho_max,100*(1-1/(1+GAM*rho_max)),GAM))
# --- papeis dos outros avioes (partida/chegada) para colorir
papel={}
for rr in D:
    if rr['obs'] is not None: papel[rr['obs']]='dep'
for rr in A:
    if rr['obs'] is not None: papel[rr['obs']]='arr'
# --- Figura 1: tres instantes da janela, com o aeroporto por baixo
fig,ax=plt.subplots(1,3,figsize=(24,9)); SP=np.array([g.proj(x['lat'],x['lon']) for x in g.A.stands])
for A_,fr in zip(ax,(0.2,0.55,0.9)):
    tq=t_real+fr*(t_alvo-t_real)
    for e in g.edges: a_,b_=g.xy[e['a']],g.xy[e['b']]; A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#d4d7da' if e['kind']=='taxi' else '#8b9096',lw=.8 if e['kind']=='taxi' else 3,zorder=1)
    # zona de rastros: exposicao em cada aresta de taxi neste instante (alcance que cresce)
    rh=np.array([F.rho(dep,d.dEE,k,tq,pr) for k in range(d.nE)]); vmax=max(rh.max(),1e-6)
    for k in np.where(rh>0.05)[0]:
        a_,b_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color=plt.cm.YlOrRd(min(rh[k]/vmax,1)),lw=2+6*min(rh[k]/vmax,1),alpha=.7,zorder=2,solid_capstyle='round')
    # os outros avioes taxiando agora: observados (triangulos) e simulados (losangulos vazios), com o alcance do rastro (circulo)
    aa=d.a[(d.a.t>=tq-12)&(d.a.t<=tq+12)&(d.a.icao24!=r['icao'])].drop_duplicates('icao24',keep='last'); nobs=0
    for q in aa.itertuples():
        c={'dep':'#1e8449','arr':'#6c3483'}.get(papel.get(q.icao24),'#7f8c8d'); A_.scatter(q.x,q.y,marker='^',s=70,color=c,edgecolors='k',linewidths=.4,zorder=6); nobs+=1
        A_.add_patch(plt.Circle((q.x,q.y),min(pr['R0']+pr['vexp']*0/60,pr['Rmax']),fill=False,ec=c,lw=.8,alpha=.5,zorder=3))
    # posicao do voo em foco: real e simuladas
    ir=np.searchsorted(r['tp'],tq); ir=min(max(ir,1),len(PD)-1); A_.scatter(*PD[ir],marker='*',s=420,color='#f1c40f',edgecolors='k',zorder=9,label='REAL (ADS-B)')
    A_.plot(PLD[:,0],PLD[:,1],color='#2980b9',lw=1.4,alpha=.6,zorder=4); A_.plot(PLR[:,0],PLR[:,1],color='#27ae60',lw=1.4,alpha=.6,zorder=4)
    for (nome,g_),cor,mk in ((('D',GAM),'#2980b9','o'),(('R',GAM),'#27ae60','s')):
        s0,trj,Lt=sim[(nome,g_)]; row=trj[np.searchsorted(trj[:,0],tq)-1] if tq>trj[0,0] else trj[0]; frac=row[1]/Lt if tq<=trj[-1,0] else 1.0; pos=sobre(PLD if nome=='D' else PLR,frac if tq>=s0 else 0); A_.scatter(*pos,marker=mk,s=230,color=cor,edgecolors='k',zorder=8,label=f'sim {nome} (gamma={g_})')
    pts=np.vstack([PLD,PLR,SP[np.argmin(np.hypot(*(SP-r['xy']).T))][None]]); c0=(pts.min(0)+pts.max(0))/2; span=max(np.ptp(pts[:,0]),np.ptp(pts[:,1]),1200)/2*1.25; A_.set_xlim(c0[0]-span,c0[0]+span); A_.set_ylim(c0[1]-span,c0[1]+span); A_.set_aspect('equal'); A_.set_xticks([]); A_.set_yticks([])
    A_.set_title(f'{DIA} {r["cs"]} stand {r["stand"]} | t = {(tq-t_alvo)/60:+.1f} min do MVT | {nobs} outros avioes no ADS-B nesse instante',fontsize=10)
    if fr==0.2: A_.legend(loc='lower left',fontsize=8)
plt.tight_layout(); plt.savefig(f'{S}/foco_janela_mapas.png',dpi=95)
# --- Figura 2: progresso, exposicao e erro
fig,ax=plt.subplots(1,3,figsize=(24,6.8)); 
tp=r['tp']; cumP=np.r_[0,np.cumsum(np.hypot(*np.diff(PD,axis=0).T))]; ax[0].plot((tp-t_alvo)/60,cumP/cumP[-1],color='k',lw=2.5,label='REAL (ADS-B)')
for (nome,g_),(s0,trj,Lt) in sim.items(): ax[0].plot((trj[:,0]-t_alvo)/60,np.minimum(trj[:,1]/Lt,1),lw=2,ls='-' if g_>0 else '--',color='#2980b9' if nome=='D' else '#27ae60',label=f'sim {nome} gamma={g_}')
ax[0].set_xlabel('minutos em relacao ao MVT'); ax[0].set_ylabel('fracao do caminho percorrida'); ax[0].set_title('progresso do voo em foco: real x simulado'); ax[0].legend(fontsize=8)
s0,trj,Lt=sim[('D',GAM)]; sc=ax[1].scatter((trj[:,0]-t_alvo)/60,trj[:,1]/Lt,c=trj[:,3],cmap='YlOrRd',s=14); plt.colorbar(sc,ax=ax[1],label='exposicao ao rastro dos outros (rho)'); ax[1].set_xlabel('minutos em relacao ao MVT'); ax[1].set_ylabel('posicao no caminho (fracao)'); ax[1].set_title('onde e quando o rastro dos outros reduz a velocidade (sim D)')
ax[2].bar(range(4),[sim[('D',0.0)][0]-t_real,sim[('D',GAM)][0]-t_real,sim[('R',0.0)][0]-t_real,sim[('R',GAM)][0]-t_real],color=['#85c1e9','#2980b9','#82e0aa','#27ae60']); ax[2].set_xticks(range(4)); ax[2].set_xticklabels(['D sem rastros','D com rastros','R sem rastros','R com rastros'],fontsize=9); ax[2].axhline(0,color='k'); ax[2].set_ylabel('erro do inicio previsto do movimento (s)'); ax[2].set_title('quanto cada simulacao erra o inicio do taxi (previsto - real)')
plt.tight_layout(); plt.savefig(f'{S}/foco_janela_curvas.png',dpi=95)
