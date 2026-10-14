"""Simulacao multiagente em tempo real: TODOS os avioes da janela andam ao mesmo tempo e soltam rastros (alcance cresce, forca decai); o rastro dos OUTROS reduz a velocidade de cada um."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import sim_dia as SD, sim_run as SR, foco as F
def poly_ed(d,ed,sp=None):
    """polilinha de uma lista de arestas consecutivas (caminho completo entre nos)"""
    sy=d.sy; g=d.g; ks=[e[0] for e in ed]
    if len(ks)==1: nodes=[sy.ea[ks[0]],sy.eb[ks[0]]]
    else:
        a0,b0=sy.ea[ks[0]],sy.eb[ks[0]]; shared={a0,b0}&{sy.ea[ks[1]],sy.eb[ks[1]]}; cur=b0 if (a0 not in shared) is False else a0
        cur=a0 if b0 in shared else b0; nodes=[cur]
        for k in ks:
            nxt=sy.eb[k] if sy.ea[k]==nodes[-1] else sy.ea[k]; nodes.append(nxt)
    P=np.vstack([g.xy[n] for n in nodes]); return np.vstack([sp[None],P]) if sp is not None else P
def posicao(PL,frac):
    c=np.r_[0,np.cumsum(np.hypot(*np.diff(PL,axis=0).T))]; x=np.clip(frac,0,1)*c[-1]; i=max(min(int(np.searchsorted(c,x)),len(c)-1),1); w=(x-c[i-1])/max(c[i]-c[i-1],1e-9); return PL[i-1]+w*(PL[i]-PL[i-1])
def monta_agentes(d,D,A,t_a,t_b,v_free,foco=None):
    """todos os avioes que taxiam na janela: partidas (stand->pista, comecando no inicio observado quando ha ADS completo, senao no nominal) e chegadas (pista->stand, a partir do pouso)"""
    ag=[]
    for j,r in enumerate(D):
        if not r['ok'] or (foco is not None and r['id']==foco['id']): continue
        Ls=np.array([e[1] for e in r['ed']],float); fim=r['mvt']-SD.ROLL_DEP
        ini=r['tstar'] if (r['chain'] and np.hypot(*(r['Pstar']-r['xy']))<=150) else fim-Ls.sum()/v_free
        if ini>t_b or fim+300<t_a: continue
        ag.append(dict(tipo='dep',id=r['id'],cs=r['cs'],segs=r['ed'],Ls=Ls,ks=np.array([e[0] for e in r['ed']]),ini=ini,fim_real=fim,PL=poly_ed(d,r['ed'],r['xy']),v=v_free,foco=False))
    for r in A:
        if not r['ok']: continue
        Ls=np.array([e[1] for e in r['ed']],float); ini=r['t_in']
        if ini>t_b or r['t_out']<t_a: continue
        ag.append(dict(tipo='arr',id=r['id'],cs=r['cs'],segs=r['ed'],Ls=Ls,ks=np.array([e[0] for e in r['ed']]),ini=ini,fim_real=r['t_out'],PL=poly_ed(d,r['ed']),v=v_free,foco=False))
    return ag
def simular(d,ag,gamma,pr,t_ini,t_fim,dt=5.0):
    n=len(ag); ns=int((t_fim-t_ini)/dt)+1; E=-np.ones((ns,n),int); POS=np.full((ns,n),np.nan); SPD=np.full((ns,n),np.nan); XY=np.full((ns,n,2),np.nan)
    pos=np.zeros(n); done=np.zeros(n,bool); chegada=np.full(n,np.nan); tot=np.array([a['Ls'].sum() for a in ag]); cum=[np.cumsum(a['Ls']) for a in ag]; Ts=int(pr['T']/dt)
    for s in range(ns):
        t=t_ini+s*dt; act=(~done)&(np.array([a['ini'] for a in ag])<=t)
        for i in np.where(act)[0]:
            idx=min(int(np.searchsorted(cum[i],pos[i]+1e-9)),len(cum[i])-1); E[s,i]=ag[i]['ks'][idx]; POS[s,i]=pos[i]; XY[s,i]=posicao(ag[i]['PL'],pos[i]/tot[i])
        for i in np.where(act)[0]:
            if gamma>0 and E[s,i]>=0:
                lo=max(0,s-Ts); rows=E[lo:s+1,:]; dtm=((s-np.arange(lo,s+1))*dt)[:,None]; Rr=np.minimum(pr['R0']+pr['vexp']*dtm/60.0,pr['Rmax'])
                dist=d.dEE[E[s,i],np.where(rows>=0,rows,0)]; K=np.clip(1-dist/Rr,0,None)*np.clip(1-dtm/pr['T'],0,None); K[rows<0]=0; K[:,i]=0
                rho=float(K.max(axis=0).sum())
            else: rho=0.0
            v=ag[i]['v']/(1+gamma*rho); SPD[s,i]=v/ag[i]['v']; pos[i]+=v*dt
            if pos[i]>=tot[i]-1e-6: done[i]=True; chegada[i]=t+dt
    return dict(E=E,POS=POS,SPD=SPD,XY=XY,chegada=chegada,t=t_ini+np.arange(ns)*dt)
