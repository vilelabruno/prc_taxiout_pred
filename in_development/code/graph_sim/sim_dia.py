"""Simulacao de um dia inteiro (todas as partidas e chegadas): cada voo deixa RASTRO nas arestas do grafo; o rastro dos outros (com duracao T min e alcance R m, peso w_ADS para rastro observado) reduz a velocidade."""
import sys, os, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import taxigraph as T, regride as R
from sincro import Sync
from sincro3 import Sync3
KT=0.514444; ROLL_DEP=45.0; ROLL_ARR=40.0; D_PB=70.0; V_PATIO=2.0
class Dia:
    def __init__(s,apt,day):
        s.apt=apt; s.day=day; s.t0=pd.Timestamp(day,tz='UTC').value//10**9
        s.sy=Sync3(apt); s.g=s.sy.g; s.Rt=T.Router(s.g); s.idx=T.build_stand_index(s.g.A); s.old=Sync(apt)
        tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt']).rename(columns={'MVT_ID_mvt':'id'})
        tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
        s.dep=tr[(tr.ADEP_mvt==apt)&(tr.PHASE_mvt=='DEP')&(tr.mvt>=s.t0)&(tr.mvt<s.t0+86400)].reset_index(drop=True)
        s.arr=tr[(tr.ADES_mvt==apt)&(tr.PHASE_mvt=='ARR')&(tr.blk>=s.t0)&(tr.mvt<s.t0+86400)&(tr.blk>tr.mvt)&(tr.blk-tr.mvt<2400)].reset_index(drop=True)
        fa=f'data/processed/adsb_surface/{day}.parquet'
        if not os.path.exists(fa): fa=f'data/processed/adsb_surface_extra/{day}.parquet'
        if os.path.exists(fa): a=pd.read_parquet(fa,filters=[('apt','==',apt)]); a=a[a.alt==0].sort_values('t').reset_index(drop=True); a['cs']=a.callsign.str.strip()
        else: a=pd.DataFrame({'t':np.zeros(0),'lon':np.zeros(0),'lat':np.zeros(0),'gs':np.zeros(0),'alt':np.zeros(0),'icao24':np.zeros(0,dtype=object),'callsign':np.zeros(0,dtype=object),'cs':np.zeros(0,dtype=object)})
        X=np.c_[(a.lon.to_numpy()-s.g.lon0)*s.g.kx,(a.lat.to_numpy()-s.g.lat0)*s.g.ky]; d,i=s.sy.tree.query(X); a['k']=np.where(d<=s.sy.tol,s.sy.own[i],-1); a['x']=X[:,0]; a['y']=X[:,1]; s.a=a
        s.grp={k:v for k,v in a.groupby('cs')}; s.gn={}
        for k,v in s.grp.items(): s.gn.setdefault(R.norm(k),[]).append(v)
        s.nE=len(s.sy.E); s.NB=1440
        # distancia entre arestas (grafo completo, sem restricao) para o alcance dos rastros
        Du=s.sy.matriz(next(iter([r['RUNWAY_mvt'] for _,r in s.dep.iterrows()])))[2]; ea,eb=s.sy.ea,s.sy.eb
        s.dEE=np.minimum.reduce([Du[np.ix_(ea,ea)],Du[np.ix_(ea,eb)],Du[np.ix_(eb,ea)],Du[np.ix_(eb,eb)]])
        s.info={}; s.anc={}; s.v=None
    def stand_info(s,st):
        if st in s.info: return s.info[st]
        la,lo,lv=T.map_stand(st,s.idx)
        if lv>=9: s.info[st]=None; return None
        sp=np.array(s.g.proj(la,lo)); SP=np.array([s.g.proj(x['lat'],x['lon']) for x in s.g.A.stands]); j=int(np.argmin(np.hypot(*(SP-sp).T))); h=np.radians(90-s.g.A.stands[j]['hdg']); u=np.array([np.cos(h),np.sin(h)]); F=sp-D_PB*u
        best=np.inf
        for k in s.sy.tk:
            A_,B_=s.g.xy[s.sy.ea[k]],s.g.xy[s.sy.eb[k]]; ab=B_-A_; den=ab@ab; f=0. if den==0 else float(np.clip((F-A_)@ab/den,0,1)); best=min(best,float(np.hypot(*(F-(A_+f*ab)))))
        opts,_=s.Rt.attach(la,lo); s.info[st]=dict(xy=sp,F=F,t_patio=(best+D_PB)/V_PATIO,opts=opts); return s.info[st]
    def anchor(s,rwy):
        if rwy not in s.anc:
            try: pair,c,rl=s.g.targets(rwy); s.anc[rwy]=min(c,key=lambda x:x[1])[0] if c else None
            except Exception: s.anc[rwy]=None
        return s.anc[rwy]
    def caminho(s,rwy,opts,to_node=None,to_edge=None):
        D,P,Du,Pu=s.sy.matriz(rwy); sy=s.sy; best=(np.inf,None,None,True,0.)
        if to_node is not None:
            for nd,acc in opts:
                c=acc+D[nd,to_node]
                if c<best[0]: best=(c,nd,to_node,True,0.)
            if not np.isfinite(best[0]):
                for nd,acc in opts:
                    c=acc+Du[nd,to_node]
                    if c<best[0]: best=(c,nd,to_node,False,0.)
        else:
            k,f=to_edge; a,b,L=sy.ea[k],sy.eb[k],sy.L[k]
            for dirc,DD in ((True,D),(False,Du)):
                for nd,acc in opts:
                    for end,off in ((a,f*L),(b,(1-f)*L)):
                        c=acc+DD[nd,end]+off
                        if c<best[0]: best=(c,nd,end,dirc,off)
                if np.isfinite(best[0]): break
        if not np.isfinite(best[0]): return np.nan,[]
        c,nd,via,dirc,off=best; P_=P if dirc else Pu; cur=via; path=[cur]
        while cur!=nd and P_[nd,cur]>=0: cur=int(P_[nd,cur]); path.append(cur)
        path=path[::-1]; ed=[(sy.ek[(a_,b_)],sy.L[sy.ek[(a_,b_)]]) for a_,b_ in zip(path[:-1],path[1:]) if (a_,b_) in sy.ek]
        if to_edge is not None: ed.append((to_edge[0],off))
        return c,ed
    def liga(s,cs,t_ini,t_fim):
        """icao24 do voo no ADS-B (indicativo exato, senao normalizado) com pings de solo na janela [t_ini,t_fim]"""
        q=s.grp.get(cs); cands=[q] if q is not None else s.gn.get(R.norm(cs),[])
        best=None
        for qq in cands:
            w=qq[(qq.t>=t_ini)&(qq.t<=t_fim)]
            if len(w): 
                ic=w.icao24.iloc[-1]; best=(len(w),ic) if best is None or len(w)>best[0] else best
        return best[1] if best else None
    def kernel(s,Tm,Rm):
        A=np.eye(s.nE) if Rm<=0 else np.clip(1-s.dEE/Rm,0,None); A[~np.isfinite(A)]=0
        kv=np.array([1.0]) if Tm<=0 else np.r_[1.0,np.clip(1-np.arange(1,int(Tm)+1)/Tm,0,None)]
        return A,kv
    def rho(s,O,A,kv):
        r=np.zeros_like(O)
        for lag,w in enumerate(kv):
            if w<=0: continue
            r[:,lag:]+=w*O[:,:s.NB-lag] if lag>0 else w*O
        return A@r
