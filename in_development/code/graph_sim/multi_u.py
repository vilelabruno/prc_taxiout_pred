"""Simulacao por UNIDADES: a unidade e a distancia media entre dois pings consecutivos do ADS-B no aeroporto. A cada unidade que um aviao avanca ele EMITE rastro em todas as direcoes do grafo ate X km;
o rastro (que decai no tempo) reduz a velocidade de quem esta dentro do alcance."""
import sys, os, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
def unidade(d):
    a=d.a[d.a.k>=0]; ic=a.icao24.to_numpy(); t=a.t.to_numpy(); x=a.x.to_numpy(); y=a.y.to_numpy(); gs=a.gs.to_numpy()*0.514444
    m=(ic[1:]==ic[:-1])&(t[1:]-t[:-1]>0)&(t[1:]-t[:-1]<=12)&(gs[1:]>1)&(gs[1:]<15); dd=np.hypot(x[1:]-x[:-1],y[1:]-y[:-1])[m]; return float(dd.mean()),float(np.median(dd)),int(m.sum())
def simular_u(d,ag,gamma,X,T,u,vref,t_ini,t_fim):
    n=len(ag); dt=u/vref; ns=int((t_fim-t_ini)/dt)+1; E=-np.ones((ns,n),int); XY=np.full((ns,n,2),np.nan); SPD=np.full((ns,n),np.nan)
    pos=np.zeros(n); prog=np.zeros(n); done=np.zeros(n,bool); chegada=np.full(n,np.nan); tot=np.array([a['Ls'].sum() for a in ag]); cum=[np.cumsum(a['Ls']) for a in ag]; ini=np.array([a['ini'] for a in ag]); rr=np.array([a['v'] for a in ag])/vref; Ts=int(T/dt)
    from multi import posicao
    for s in range(ns):
        t=t_ini+s*dt; act=np.where((~done)&(ini<=t))[0]
        for i in act:
            idx=min(int(np.searchsorted(cum[i],pos[i]+1e-9)),len(cum[i])-1); e_i=ag[i]['ks'][idx]; XY[s,i]=posicao(ag[i]['PL'],pos[i]/tot[i])
            rho=0.0
            if gamma>0 and e_i>=0 and s>0:
                lo=max(0,s-Ts); rows=E[lo:s,:]; dtm=((s-np.arange(lo,s))*dt)[:,None]
                dist=d.dEE[e_i,np.where(rows>=0,rows,0)]; K=np.clip(1-dist/X,0,None)*np.clip(1-dtm/T,0,None); K[rows<0]=0; K[:,i]=0; rho=float(K.max(axis=0).sum())
            f=1.0/(1.0+gamma*rho); SPD[s,i]=f; prog[i]+=rr[i]*f
            if prog[i]>=1.0:                                  # avancou 1 unidade: emite rastro na aresta atual
                prog[i]-=1.0; pos[i]+=u; E[s,i]=e_i
                if pos[i]>=tot[i]-1e-6: done[i]=True; chegada[i]=t+dt
    return dict(E=E,XY=XY,SPD=SPD,chegada=chegada,t=t_ini+np.arange(ns)*dt,dt=dt)
def campo(d,res,s,X,T):
    """intensidade do rastro em cada aresta no passo s: soma, por aviao, do maior efeito entre suas emissoes recentes (distancia pelo grafo em todas as direcoes)"""
    dt=res['dt']; Ts=int(T/dt); lo=max(0,s-Ts); rows=res['E'][lo:s+1,:]; dtm=((s-np.arange(lo,s+1))*dt)[:,None]*np.ones_like(rows)
    pts=np.where(rows>=0); e=rows[pts]; ag=pts[1]; dm=dtm[pts]; D=d.dEE[:,e]; K=np.clip(1-D/X,0,None)*np.clip(1-dm/T,0,None)[None,:]
    I=np.zeros(d.nE)
    for a in np.unique(ag): I+=K[:,ag==a].max(axis=1)
    return I
