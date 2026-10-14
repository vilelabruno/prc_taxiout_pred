"""Grafo v3 COMPLETO (taxiway + pista, sem sentido nas taxiways). Restricao na pista, por pista de decolagem: a pista do voo so e percorrida no sentido da decolagem;
outras pistas so em cruzamento curto (<=150 m); nunca para taxiar ao longo de uma pista."""
import sys, numpy as np
sys.path.insert(0,'src/physical')
import taxigraph as T
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
CRUZ=150.0
class Sync3:
    def __init__(s,apt,tol=30.0):
        g=s.g=T.TaxiGraph(apt); s.tol=tol
        E=[e for e in g.edges if g.main[e['a']] and g.main[e['b']]]; s.E=E
        s.ea=np.array([e['a'] for e in E]); s.eb=np.array([e['b'] for e in E]); s.L=np.array([e['L'] for e in E]); s.kind=np.array([e['kind'] for e in E]); s.name=[e['name'] for e in E]
        tk=[k for k in range(len(E)) if s.kind[k]=='taxi']; s.tk=np.array(tk); cloud,own=[],[]
        for k in tk:
            A=g.xy[s.ea[k]]; B=g.xy[s.eb[k]]; n=max(1,int(np.ceil(s.L[k]/4))); cloud.append(np.linspace(A,B,n+1)); own+=[k]*(n+1)
        s.own=np.array(own); s.tree=cKDTree(np.vstack(cloud)); s.ek={}
        for k in range(len(E)):
            for a,b in ((s.ea[k],s.eb[k]),(s.eb[k],s.ea[k])):
                if (a,b) not in s.ek or s.L[k]<s.L[s.ek[(a,b)]]: s.ek[(a,b)]=k
        s._m={}
    def matriz(s,end):
        """distancias/predecessores para o voo que decola em `end` (cache por cabeceira)"""
        if end in s._m: return s._m[end]
        g=s.g; pair,r=g.rwy_pair(end); r_=[];c_=[];w_=[]
        if r is not None:
            thr=np.array(g.proj(r['lat1'],r['lon1'])) if end==r['end1'] else np.array(g.proj(r['lat2'],r['lon2'])); oth=np.array(g.proj(r['lat2'],r['lon2'])) if end==r['end1'] else np.array(g.proj(r['lat1'],r['lon1'])); u=(oth-thr)/np.hypot(*(oth-thr))
        for k in range(len(s.E)):
            a,b,L=s.ea[k],s.eb[k],s.L[k]
            if s.kind[k]=='taxi': r_+=[a,b]; c_+=[b,a]; w_+=[L,L]
            elif r is not None and s.name[k]==pair:                 # pista de decolagem do voo: so no sentido da decolagem
                fwd=(g.xy[b]-g.xy[a])@u>0; (r_.append(a),c_.append(b)) if fwd else (r_.append(b),c_.append(a)); w_.append(L)
            elif L<=CRUZ: r_+=[a,b]; c_+=[b,a]; w_+=[L,L]            # outras pistas: so cruzamento curto
        M=coo_matrix((w_,(r_,c_)),shape=(g.n,g.n)).tocsr(); D,P=dijkstra(M,directed=True,return_predecessors=True)
        Mu=coo_matrix((np.r_[s.L,s.L],(np.r_[s.ea,s.eb],np.r_[s.eb,s.ea])),shape=(g.n,g.n)).tocsr(); Du,Pu=dijkstra(Mu,directed=False,return_predecessors=True)   # reserva (grafo completo sem restricao)
        s._m[end]=(D,P,Du,Pu); return s._m[end]
    def snap_many(s,P):
        d,i=s.tree.query(P); k=s.own[i]; ok=d<=s.tol; f=np.zeros(len(P))
        for j in np.where(ok)[0]:
            A=s.g.xy[s.ea[k[j]]]; B=s.g.xy[s.eb[k[j]]]; ab=B-A; den=ab@ab; f[j]=0. if den==0 else float(np.clip((P[j]-A)@ab/den,0,1))
        return np.where(ok,k,-1),f
