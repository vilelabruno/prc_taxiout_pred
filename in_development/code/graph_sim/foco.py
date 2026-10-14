"""Simulacao em tempo real de UM voo em foco (ADS completo) com todos os outros avioes da janela: (R) rota recente como guia ate o MEU stand, (D) Dijkstra + ADS ponto a ponto; rastros que crescem."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import taxigraph as T, regride as R
from sim_dia import Dia, KT, ROLL_DEP, ROLL_ARR
import sim_run as SR
P_DEF=dict(T=900.0,R0=100.0,vexp=60.0,Rmax=600.0,wads=3.0,dt=5.0)       # duracao do rastro (s), alcance inicial (m), expansao (m/min), alcance maximo (m), peso do rastro vindo do ADS-B, passo (s)
def hop(d,rwy,k1,f1,k2,f2):
    sy=d.sy; D,P,Du,Pu=sy.matriz(rwy)
    if k1==k2: return [(k1,abs(f2-f1)*sy.L[k1])]
    best=(np.inf,None)
    for ea_,da in ((sy.ea[k1],f1*sy.L[k1]),(sy.eb[k1],(1-f1)*sy.L[k1])):
        for eb_,db in ((sy.ea[k2],f2*sy.L[k2]),(sy.eb[k2],(1-f2)*sy.L[k2])):
            c=da+Du[ea_,eb_]+db
            if c<best[0]: best=(c,(ea_,eb_,da,db))
    if best[1] is None or not np.isfinite(best[0]): return [(-1,float(np.hypot(*(d.g.xy[sy.ea[k1]]-d.g.xy[sy.ea[k2]]))))]
    a,b,da,db=best[1]; cur=int(b); path=[cur]
    while cur!=a and Pu[a,cur]>=0: cur=int(Pu[a,cur]); path.append(cur)
    path=path[::-1]; out=[(k1,da)]+[(sy.ek[(x,y)],sy.L[sy.ek[(x,y)]]) for x,y in zip(path[:-1],path[1:]) if (x,y) in sy.ek]+[(k2,db)]
    return [(k,l) for k,l in out if l>0]
def segs_de_pings(d,rwy,P,stand_opts=None,stand_xy=None):
    """caminho ponto a ponto: Dijkstra entre pings consecutivos (grafo completo); pings no patio viram trechos retos (aresta -1, sem rastro)"""
    k,f=d.sy.snap_many(P); out=[]
    for i in range(len(P)-1):
        if k[i]>=0 and k[i+1]>=0: out+=hop(d,rwy,k[i],f[i],k[i+1],f[i+1])
        else: out.append((-1,float(np.hypot(*(P[i+1]-P[i])))))
    return out
def deposito(d,D,A,t_a,t_b,ic_foco,id_foco,wads):
    """rastros de TODOS os outros avioes da janela: observados no ADS-B (peso wads) e, para quem nao tem ADS-B, simulados (peso 1)"""
    a=d.a[(d.a.t>=t_a)&(d.a.t<=t_b)&(d.a.k>=0)&(d.a.icao24!=ic_foco)]; ag=list(a.icao24.astype('category').cat.codes.to_numpy()); ek=list(a.k.to_numpy()); tt=list(a.t.to_numpy().astype(float)); w=[wads]*len(a); papel={}
    papel_obs=set(a.icao24.unique()); ncode=max(ag)+1 if ag else 0; sim=0
    for j,r in enumerate(D):
        if not r['ok'] or r['obs'] is not None or r['id']==id_foco: continue
        Ls=np.array([e[1] for e in r['ed']],float); tot=Ls.sum()/d.v; fim=r['mvt']-ROLL_DEP; ini=fim-tot
        if fim<t_a or ini>t_b: continue
        cum=np.cumsum(Ls)/Ls.sum()
        for tq in np.arange(max(ini,t_a),min(fim,t_b),5.0): 
            i=min(int(np.searchsorted(cum,(tq-ini)/tot)),len(Ls)-1); ag.append(ncode); ek.append(r['ed'][i][0]); tt.append(float(tq)); w.append(1.0)
        ncode+=1; sim+=1
    for r in A:
        if not r['ok'] or r['obs'] is not None: continue
        if r['t_out']<t_a or r['t_in']>t_b: continue
        Ls=np.array([e[1] for e in r['ed']],float); cum=np.cumsum(Ls)/Ls.sum(); tot=max(r['t_out']-r['t_in'],1.0)
        for tq in np.arange(max(r['t_in'],t_a),min(r['t_out'],t_b),5.0):
            i=min(int(np.searchsorted(cum,(tq-r['t_in'])/tot)),len(Ls)-1); ag.append(ncode); ek.append(r['ed'][i][0]); tt.append(float(tq)); w.append(1.0)
        ncode+=1; sim+=1
    ag=np.array(ag); ek=np.array(ek); tt=np.array(tt); w=np.array(w,float); o=np.lexsort((tt,ag))
    return dict(ag=ag[o],ek=ek[o],t=tt[o],w=w[o],n_obs=len(papel_obs),n_sim=sim)
def rho(dep,dEE,eF,t,pr):
    """exposicao do voo em foco ao rastro dos outros: por aviao, o maior efeito entre seus pontos de rastro (alcance cresce com o tempo, forca decai linearmente)"""
    m=(dep['t']<=t)&(t-dep['t']<=pr['T'])
    if not m.any(): return 0.0
    dtm=t-dep['t'][m]; Rr=np.minimum(pr['R0']+pr['vexp']*dtm/60.0,pr['Rmax']); dist=dEE[eF,dep['ek'][m]]
    K=np.clip(1-dist/Rr,0,None)*(1-dtm/pr['T'])*dep['w'][m]; ag=dep['ag'][m]; b=np.flatnonzero(np.r_[True,ag[1:]!=ag[:-1]])
    return float(np.maximum.reduceat(K,b).sum())
def avanca(segs,v_free,gamma,dep,dEE,t_ini,pr,gravar=False):
    """simula passo a passo: velocidade = v_free/(1+gamma*rho) na aresta atual; devolve instante de chegada ao fim do caminho (e a trajetoria)"""
    Ls=np.array([s[1] for s in segs],float); ks=np.array([s[0] for s in segs]); tot=Ls.sum(); i=0; pos=0.0; t=t_ini; trj=[]; cum=np.cumsum(Ls)
    while pos<tot-1e-6 and t<t_ini+7200:
        while i<len(Ls)-1 and pos>=cum[i]-1e-9: i+=1
        r_=rho(dep,dEE,ks[i],t,pr) if (ks[i]>=0 and gamma>0) else 0.0; v=v_free/(1+gamma*r_)
        if gravar: trj.append((t,pos,ks[i],r_,v))
        pos+=v*pr['dt']; t+=pr['dt']
    return (t,np.array(trj)) if gravar else t
def prever_inicio(segs,v_free,gamma,dep,dEE,t_alvo,t_real,pr):
    """acha o instante de partida do movimento tal que a simulacao chegue ao fim do caminho em t_alvo (bissecao)"""
    lo,hi=t_alvo-2700,t_alvo
    for _ in range(14):
        mid=(lo+hi)/2
        if avanca(segs,v_free,gamma,dep,dEE,mid,pr)>t_alvo: hi=mid
        else: lo=mid
    return (lo+hi)/2
def candidatos(d,D,A):
    out=[]
    for j,r in enumerate(D):
        if not (r['ok'] and r['chain'] and r['obs'] is not None): continue
        if np.hypot(*(r['Pstar']-r['xy']))>150: continue                      # a cadeia chega a vizinhanca do stand
        tp=r['tp']
        if len(tp)<60 or np.diff(tp).max()>60: continue                      # ADS completo, sem buracos grandes
        out.append(j)
    return out
def referencia(d,D,j):
    """ultima rota completa ANTERIOR (ADS completo desde o stand): 1) mesmo stand e pista; 2) mesmo stand, outra pista; 3) stand a <=300 m, mesma pista"""
    r=D[j]
    def bom(q): return q['ok'] and q['chain'] and np.hypot(*(q['Pstar']-q['xy']))<=150 and len(q['tp'])>=40 and np.diff(q['tp']).max()<=60
    for nivel in (1,2,3):
        best=None
        for i,q in enumerate(D):
            if i==j or not bom(q) or q['mvt']>=r['mvt']: continue
            if nivel==1 and not (q['stand']==r['stand'] and q['rwy']==r['rwy']): continue
            if nivel==2 and q['stand']!=r['stand']: continue
            if nivel==3 and not (q['rwy']==r['rwy'] and np.hypot(*(q['xy']-r['xy']))<=300): continue
            if best is None or q['mvt']>D[best]['mvt']: best=i
        if best is not None: return best
    return None
