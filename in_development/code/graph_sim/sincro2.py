"""Grafo v2: so taxiways (NUNCA pista), direcionado (aresta de mao unica so no sentido a->b), distancias a partir do stand (para frente)."""
import sys, numpy as np
sys.path.insert(0,'src/physical')
import taxigraph as T
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
class Sync2:
    def __init__(s,apt,tol=30.0):
        g=s.g=T.TaxiGraph(apt); s.tol=tol
        E=[e for e in g.edges if e['kind']=='taxi' and g.main[e['a']] and g.main[e['b']]]
        s.E=E; s.ea=np.array([e['a'] for e in E]); s.eb=np.array([e['b'] for e in E]); s.L=np.array([e['L'] for e in E]); s.ow=np.array([bool(e['oneway']) for e in E])
        cloud,own=[],[]
        for k in range(len(E)):
            A=g.xy[s.ea[k]]; B=g.xy[s.eb[k]]; n=max(1,int(np.ceil(s.L[k]/4))); cloud.append(np.linspace(A,B,n+1)); own+=[k]*(n+1)
        s.own=np.array(own); s.tree=cKDTree(np.vstack(cloud))
        r=list(s.ea)+[s.eb[k] for k in range(len(E)) if not s.ow[k]]; c=list(s.eb)+[s.ea[k] for k in range(len(E)) if not s.ow[k]]; w=list(s.L)+[s.L[k] for k in range(len(E)) if not s.ow[k]]
        M=coo_matrix((w,(r,c)),shape=(g.n,g.n)).tocsr(); s.D,s.P=dijkstra(M,directed=True,return_predecessors=True)
        Mu=coo_matrix((np.r_[s.L,s.L],(np.r_[s.ea,s.eb],np.r_[s.eb,s.ea])),shape=(g.n,g.n)).tocsr(); s.Du,s.Pu=dijkstra(Mu,directed=False,return_predecessors=True)   # reserva: sem sentido
        s.ek={}
        for k in range(len(E)):
            for a,b in ((s.ea[k],s.eb[k]),(s.eb[k],s.ea[k])):
                if (a,b) not in s.ek or s.L[k]<s.L[s.ek[(a,b)]]: s.ek[(a,b)]=k
    def snap_many(s,P):
        """aresta de taxi mais proxima de cada ponto (dentro da tolerancia) e a fracao ao longo dela"""
        d,i=s.tree.query(P); k=s.own[i]; ok=d<=s.tol; f=np.zeros(len(P))
        for j in np.where(ok)[0]:
            A=s.g.xy[s.ea[k[j]]]; B=s.g.xy[s.eb[k[j]]]; ab=B-A; den=ab@ab; f[j]=0. if den==0 else float(np.clip((P[j]-A)@ab/den,0,1))
        return np.where(ok,k,-1),f
    def para_ponto(s,opts,k,f):
        """menor caminho (para frente) do stand ate o ponto (aresta k, fracao f): devolve (distancia, no de entrada do stand, no por onde entra na aresta, direcao) respeitando mao unica"""
        best=(np.inf,None,None); a,b,L=s.ea[k],s.eb[k],s.L[k]
        for nd,acc in opts:
            c=acc+s.D[nd,a]+f*L
            if c<best[0]: best=(c,nd,a)
            if not s.ow[k]:
                c=acc+s.D[nd,b]+(1-f)*L
                if c<best[0]: best=(c,nd,b)
        return best
    def arestas(s,nd,via):
        """arestas percorridas de nd ate via (nos), pela matriz direcionada"""
        out=[]; cur=via; path=[cur]
        while cur!=nd and s.P[nd,cur]>=0: cur=int(s.P[nd,cur]); path.append(cur)
        path=path[::-1]
        for a,b in zip(path[:-1],path[1:]): out.append(s.ek.get((a,b)))
        return [x for x in out if x is not None]
