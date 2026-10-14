"""Simulador minimo: y_hat = saida_do_stand + L_dijkstra(stand->espera da pista) / v_efetiva.
v_efetiva e saida estimadas so com voos cujo MVT < BLOCK do alvo (ultima hora + ultimos 30 dias, mesma pista)."""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import taxigraph as T
OBS=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/obs'); OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/sim')
K=5.0   # pseudo-contagem do encolhimento hora -> mes

def rotas(apt,d):
    """Dijkstra so em taxiway (sem atalho por pista), do no do stand ate o ponto de espera de comprimento total
    (nos de espera ate 400 m da cabeceira). Mediana L/distancia observada ~0,95 em EHAM/LSZH/LEBL/EDDF."""
    sy=Sync(apt); g=sy.g; m=sy.kind!='runway'
    M=coo_matrix((np.r_[sy.L[m],sy.L[m]],(np.r_[sy.ea[m],sy.eb[m]],np.r_[sy.eb[m],sy.ea[m]])),shape=(g.n,g.n)).tocsr()
    idx=T.build_stand_index(g.A); Dr={}
    for rw in d.rwy.unique():
        _,tg,_=g.targets(rw)
        if tg:
            a0=min(al for _,_,al in tg); Dr[rw]=dijkstra(M,directed=False,indices=[x for x,_,al in tg if al<=a0+400]).min(0)
    L=np.full(len(d),np.nan); cache={}
    for i,(st,rw) in enumerate(zip(d.stand,d.rwy)):
        if (st,rw) not in cache:
            la,lo,_=T.map_stand(st,idx); v=np.nan
            if np.isfinite(la) and rw in Dr:
                n0,dd=g.stand_node(la,lo); v=dd+Dr[rw][n0]
            cache[(st,rw)]=v if np.isfinite(v) else np.nan
        L[i]=cache[(st,rw)]
    return L

def med(blk,ref_t,val,lo):
    """mediana de val dos voos de referencia com ref_t em [blk-lo, blk)"""
    a=np.searchsorted(ref_t,blk-lo); b=np.searchsorted(ref_t,blk)
    return np.array([np.median(val[i:j]) if j>i else np.nan for i,j in zip(a,b)]), b-a

def janela(blk,ref_t,num,den,lo,hi):
    """soma de num/den dos voos de referencia com ref_t em [blk-lo, blk-hi)"""
    a=np.searchsorted(ref_t,blk-lo); b=np.searchsorted(ref_t,blk-hi)
    c=np.r_[0,np.cumsum(num)]; e=np.r_[0,np.cumsum(den)]
    return c[b]-c[a], e[b]-e[a], b-a

def simula(apt):
    d=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{OBS}/{apt}_*.parquet'))],ignore_index=True)
    d['y']=d.mvt-d.blk; d=d[(d.y>0)&(d.y<7200)].reset_index(drop=True)
    d['L']=rotas(apt,d)
    ref=d[d.adsb & d.stand_start & d.L.notna()].copy()
    ref['tmv']=ref.mvt-ref.t_move; ref['ex']=ref.t_move-ref.blk
    ref=ref[(ref.tmv>30)&(ref.ex>-120)&(ref.ex<3600)]
    d['v_h']=d['v_m']=d['v']=d['ex_hat']=np.nan
    for rw,gi in d.groupby('rwy').groups.items():
        r=ref[ref.rwy==rw].sort_values('mvt'); 
        if len(r)<20: continue
        b=d.loc[gi,'blk'].to_numpy(); t=r.mvt.to_numpy(); vv=(r.L/r.tmv).to_numpy()
        vh,ch=med(b,t,vv,3600); vm,cm=med(b,t,vv,30*86400)
        vm=np.where(cm>=10,vm,np.nan); w=ch/(ch+K); v=np.where(ch>0,w*vh+(1-w)*vm,vm)
        d.loc[gi,'v_h']=vh; d.loc[gi,'v_m']=vm; d.loc[gi,'v']=v
    # saida do stand: mediana movel 30 dias do aeroporto (stand com >=10 refs usa a do stand)
    r=ref.sort_values('mvt'); t=r.mvt.to_numpy(); ex=r.ex.to_numpy(); sts=r.stand.to_numpy()
    out=np.full(len(d),np.nan)
    for i,(b,st) in enumerate(zip(d.blk,d.stand)):
        a=np.searchsorted(t,b-30*86400); z=np.searchsorted(t,b)
        if z-a<30: continue
        e=ex[a:z]; m=sts[a:z]==st
        out[i]=np.median(e[m]) if m.sum()>=10 else np.median(e)
    d['ex_hat']=out
    d['y_sim']=d.ex_hat+d.L/d.v
    # variante com a propria trilha (permitida ate o MVT): parte de movimento observada
    own=d.adsb & d.stand_start.astype('boolean').fillna(False).astype(bool)
    d['y_own']=np.where(own,d.ex_hat+(d.mvt-d.t_move),d.y_sim)
    # baseline: mediana de y da pista nos 30 dias anteriores
    bl=np.full(len(d),np.nan)
    for rw,gi in d.groupby('rwy').groups.items():
        s=d.loc[gi].sort_values('mvt'); tt=s.mvt.to_numpy(); yy=s.y.to_numpy()
        for j,b in zip(gi,d.loc[gi,'blk']):
            a=np.searchsorted(tt,b-30*86400); z=np.searchsorted(tt,b)
            if z-a>=10: bl[j]=np.median(yy[a:z])
    d['y_base']=bl; d['apt']=apt
    os.makedirs(OUT,exist_ok=True); d.to_parquet(f'{OUT}/{apt}.parquet'); return d

if __name__=='__main__':
    for apt in sys.argv[1].split(','):
        d=simula(apt); print(apt,len(d),flush=True)
