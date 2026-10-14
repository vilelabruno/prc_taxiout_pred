"""Sincronia espaco-temporal ADS-B <-> grafo (sem sentido, taxi+pista, sem descartar parados)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'src/physical')
import taxigraph as T
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
TOL=25.0
class Sync:
    def __init__(s,apt):
        g=s.g=T.TaxiGraph(apt)
        E=[e for e in g.edges if g.main[e['a']] and g.main[e['b']]]
        s.ea=np.array([e['a'] for e in E]); s.eb=np.array([e['b'] for e in E]); s.L=np.array([e['L'] for e in E])
        s.kind=np.array([e['kind'] for e in E])
        cloud,own=[],[]
        for k in range(len(E)):
            A,B=g.xy[s.ea[k]],g.xy[s.eb[k]]; n=max(1,int(np.ceil(s.L[k]/4)))
            cloud.append(np.linspace(A,B,n+1)); own+= [k]*(n+1)
        s.own=np.array(own); s.tree=cKDTree(np.vstack(cloud))
        M=coo_matrix((np.r_[s.L,s.L],(np.r_[s.ea,s.eb],np.r_[s.eb,s.ea])),shape=(g.n,g.n)).tocsr()
        s.D,s.P=dijkstra(M,directed=False,return_predecessors=True)
    def proj(s,k,p):
        A,B=s.g.xy[s.ea[k]],s.g.xy[s.eb[k]]; ab=B-A; den=ab@ab
        f=0. if den==0 else float(np.clip((p-A)@ab/den,0,1)); q=A+f*ab
        return f,float(np.hypot(*(p-q)))
    def snap(s,P):
        """aresta por ponto: mais proxima, com histerese (fica na anterior se <=4 m pior)."""
        out=[];prev=-1
        for p in P:
            ix=s.tree.query_ball_point(p,TOL)
            if not ix: out.append((-1,np.nan,np.nan)); continue
            best=None
            for k in np.unique(s.own[ix]):
                f,d=s.proj(k,p)
                if best is None or d<best[2]: best=(k,f,d)
            if prev>=0 and prev!=best[0]:
                f,d=s.proj(prev,p)
                if d<=TOL and d<=best[2]+4: best=(prev,f,d)
            out.append(best); prev=best[0]
        return out
    def path(s,x,y):
        out=[y]
        while out[-1]!=x and s.P[x,out[-1]]>=0: out.append(int(s.P[x,out[-1]]))
        return out[::-1]
    def route(s,edges):
        seq=[edges[0]]
        for e in edges[1:]:
            if e!=seq[-1]: seq.append(e)
        if len(seq)==1:
            return [int(s.ea[seq[0]]),int(s.eb[seq[0]])]
        nodes=[]
        for i,(e1,e2) in enumerate(zip(seq[:-1],seq[1:])):
            best=min(((x,y) for x in (s.ea[e1],s.eb[e1]) for y in (s.ea[e2],s.eb[e2])),key=lambda q:s.D[q])
            if i==0:
                o=s.eb[e1] if best[0]==s.ea[e1] else s.ea[e1]; nodes.append(int(o))
            nodes+= s.path(int(best[0]),int(best[1]))
        o=s.eb[seq[-1]] if nodes[-1]==s.ea[seq[-1]] else s.ea[seq[-1]]; nodes.append(int(o))
        r=[]
        for n in nodes:
            if r and r[-1]==n: continue
            if len(r)>=2 and r[-2]==n: r.pop(); continue   # remove ida-e-volta espuria
            r.append(n)
        return r
    def along(s,nodes,P):
        X=s.g.xy[nodes]; seg=X[1:]-X[:-1]; Ls=np.hypot(*seg.T); cum=np.r_[0,np.cumsum(Ls)]
        S=[];D=[];cur=0
        for p in P:
            lo,hi=max(0,cur-2),min(len(seg),cur+40)
            A=X[lo:hi]; ab=seg[lo:hi]; den=(ab**2).sum(1); den[den==0]=1
            f=np.clip(((p-A)*ab).sum(1)/den,0,1); Q=A+f[:,None]*ab; d=np.hypot(*(p-Q).T)
            j=int(d.argmin()); cur=lo+j; S.append(cum[cur]+f[j]*Ls[cur]); D.append(d[j])
        return np.array(S),np.array(D),cum
