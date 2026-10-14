"""Features fisicas por partida (id = MVT_ID_mvt): ancora do ADS-B + rota stand->1o ping (Dijkstra) + esperas + pushback.
Nao usa BLOCK nem y em nenhuma feature (BLOCK so aparece na coluna 'blk_rotulo' para validacao)."""
import sys, os, re, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
import taxigraph as T
KT=0.514444; norm=lambda c: re.sub(r'(?<=\d)[A-Z]{1,2}$','',c); OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/fisico'); _S={}
def tau(D,v,a=0.6):
    v=max(v,0.5); da=v*v/(2*a); return v/a+(D-da)/v if D>da else 2*D/v
def dia(args):
    apt,day=args; out=f'{OUT}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    if apt not in _S:
        sy=Sync(apt); _S[apt]=(sy,T.Router(sy.g),T.build_stand_index(sy.g.A))
    sy,Rt,idx=_S[apt]; g=sy.g
    full=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt'])
    ar=full[(full.PHASE_mvt=='ARR')&(full.ADES_mvt==apt)]
    ARR={k:np.sort(pd.to_datetime(v.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64').to_numpy()) for k,v in ar.groupby('STAND_mvt')}   # on-block das chegadas (conhecido tambem em 2026)
    tr=full[(full.PHASE_mvt=='DEP')&(full.ADEP_mvt==apt)].copy()
    tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
    t0=pd.Timestamp(day,tz='UTC').value//10**9; tr=tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)]
    p=f'data/processed/adsb_surface/{day}.parquet'
    a=pd.read_parquet(p,filters=[('apt','==',apt)]).sort_values('t') if os.path.exists(p) else pd.DataFrame(columns=['callsign','t','alt','icao24','lat','lon','gs'])
    a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}
    gn={}
    for k,v in grp.items(): gn.setdefault(norm(k),[]).append(v)
    R=[]
    for r in tr.itertuples():
        rec=dict(id=r.MVT_ID_mvt,day=day,mvt=r.mvt,blk_rotulo=r.blk,tem=0)
        la,lo,lv=T.map_stand(r.STAND_mvt,idx); rec['lig']=0; gd=pd.DataFrame()
        arr=ARR.get(r.STAND_mvt); lb=-1e18
        if arr is not None:
            j=np.searchsorted(arr,r.mvt-120)
            if j>0 and r.mvt-arr[j-1]<=36*3600: lb=arr[j-1]
        rec['lb_chegada']=lb if lb>-1e17 else np.nan
        def tenta(v):
            v=v[(v.t>max(r.mvt-5400,lb))&(v.t<=r.mvt+30)]; g0=v[(v.alt==0)&(v.t<=r.mvt)]
            if len(g0)<3: return None
            g0=g0[g0.icao24==g0.icao24.iloc[-1]]; sg=(g0.t.diff().fillna(0)>300).cumsum(); return g0[sg==sg.iloc[-1]].reset_index(drop=True)
        v=grp.get(r.cs); gd=tenta(v) if v is not None else None
        if gd is not None: rec['lig']=1
        else:
            best=None
            for q in gn.get(norm(r.cs),[]):
                for ic,qq in q[(q.alt==0)&(q.t<=r.mvt+60)].groupby('icao24'):
                    de=r.mvt-qq.t.max()
                    if -60<=de<=180 and (best is None or de<best[0]): best=(de,q[q.icao24==ic])
            if best is not None:
                gd=tenta(best[1])
                if gd is not None: rec['lig']=2
        if gd is not None:
            if len(gd)>=3:
                P=np.c_[(gd.lon-g.lon0)*g.kx,(gd.lat-g.lat0)*g.ky]; t=gd.t.to_numpy(); gs=gd.gs.to_numpy()*KT
                rec.update(tem=1,n=len(gd),t_first=t[0],t_last=t[-1],gs_first=gs[0],gs_last=gs[-1],mvt_ult=r.mvt-t[-1],mvt_prim=r.mvt-t[0])
                dtp=np.diff(t,prepend=t[0])
                if lv<9:
                    sp=np.array(g.proj(la,lo)); d=np.hypot(*(P-sp).T); rec['d0']=d[0]
                    rec['esp_fora']=float(dtp[(gs<1)&(d>150)&(dtp<=600)].sum())          # tempo parado fora do patio (>150 m do stand), lacunas <=10 min
                    rec['esp_pat']=float(dtp[(gs<1)&(d<=150)&(dtp<=600)].sum())
                    fast=np.where((gs>=3)&(np.r_[gs[1:],0]>=3)&(np.r_[gs[2:],0,0]>=3))[0]
                    if len(fast): rec['t_k']=t[fast[0]]; rec['mvt_k']=r.mvt-t[fast[0]]
                    # pushback: comeca parado no stand
                    if d[0]<=20 and gs[0]<=0.5:
                        dis=np.hypot(*(P-P[0]).T); j25=np.where(dis>=25)[0]
                        if len(j25):
                            pk=np.where(dis[:j25[0]]<=4)[0]
                            if len(pk):
                                io=pk[-1]+1; rec.update(t_prev=t[io-1],t_io=t[io],onset=0.5*(t[io-1]+t[io]),mvt_onset=r.mvt-0.5*(t[io-1]+t[io]),gap_onset=t[io]-t[io-1])
                    # rota do stand ao 1o ping: Dijkstra se casa no grafo; senao reta
                    opts,_=Rt.attach(la,lo); sn=sy.snap(P[:1])[0]
                    if d[0]>150 and sn[0]>=0:
                        L=sy.L[sn[0]]; best=np.inf
                        for n,acc in opts: best=min(best,acc+sn[1]*L+sy.D[n,sy.ea[sn[0]]],acc+(1-sn[1])*L+sy.D[n,sy.eb[sn[0]]])
                        rec['D_rota']=best; rec['D_tipo']=1
                    else: rec['D_rota']=d[0]; rec['D_tipo']=0
                    rec['tau_acc']=tau(rec['D_rota'],gs[0] if gs[0]>0.8 else (gs[1] if len(gs)>1 else 1.0))
        R.append(rec)
    os.makedirs(OUT,exist_ok=True); pd.DataFrame(R).to_parquet(out); return out
if __name__=='__main__':
    apt=sys.argv[1]; months=sys.argv[2].split(',')
    days=[d.strftime('%Y-%m-%d') for m in months for d in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    with Pool(int(sys.argv[3]) if len(sys.argv)>3 else 8) as pool:
        for o in pool.imap_unordered(dia,[(apt,d) for d in days]): print(o,flush=True)
