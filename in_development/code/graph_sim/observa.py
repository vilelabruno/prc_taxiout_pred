"""Por dia: partidas ligadas ao MVT, casadas no grafo -> velocidade media empregada e componentes observados."""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
import taxigraph as T
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/obs')
_S={}
def rwy_geom(g,end):
    pr,rr=g.rwy_pair(end)
    if rr is None: return None
    a=np.array(g.proj(rr['lat1'],rr['lon1'])); b=np.array(g.proj(rr['lat2'],rr['lon2']))
    thr,oth=(a,b) if end==rr['end1'] else (b,a)
    return thr,(oth-thr)/np.hypot(*(oth-thr))
def dia(args):
    apt,day=args
    out=f'{OUT}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    if apt not in _S: _S[apt]=Sync(apt)
    sy=_S[apt]; g=sy.g
    tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','RUNWAY_mvt','STAND_mvt'])
    tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
    tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
    tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
    tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    tr=tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    A=g.A; idx=T.build_stand_index(A)
    R=[]
    for r in tr.itertuples():
        rec=dict(day=day,mvt=r.mvt,blk=r.blk,rwy=str(r.RUNWAY_mvt),stand=r.STAND_mvt,cs=r.cs,adsb=False)
        v=grp.get(r.cs)
        if v is not None:
            v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt+60)]
            gd=v[v.alt==0]
            if len(gd)>=8 and r.mvt-gd.t.max()<=30:
                v=v[v.icao24==gd.icao24.iloc[-1]]; gd=v[(v.alt==0)&(v.t<=r.mvt)]
                if len(gd)<8: R.append(rec); continue
                seg=(gd.t.diff().fillna(0)>300).cumsum(); gd=gd[seg==seg.iloc[-1]].reset_index(drop=True)
                Pp=np.c_[(gd.lon-g.lon0)*g.kx,(gd.lat-g.lat0)*g.ky]; t=gd.t.to_numpy(); gs=gd.gs.fillna(0).to_numpy()
                sn=sy.snap(Pp); ed=np.array([x[0] for x in sn]); ok=ed>=0
                if ok.sum()>=5:
                    nodes=sy.route(list(ed[ok])); S,Dr,cum=sy.along(nodes,Pp)
                    S=np.maximum.accumulate(S)
                    mv=np.where(gs>2)[0]; i0=mv[0] if len(mv) else 0
                    la,lo,_=T.map_stand(r.STAND_mvt,idx)
                    d0=np.nan
                    if np.isfinite(la): d0=float(np.hypot(*(Pp[0]-np.array(g.proj(la,lo)))))
                    geo=rwy_geom(g,rec['rwy'])
                    x_last=float((Pp[-1]-geo[0])@geo[1]) if geo else np.nan
                    # velocidade media empregada: do 1o movimento observado ate o ultimo ponto de solo antes do MVT, so taxi (gs<=40)
                    tx=np.where(gs<=40)[0]; i1=tx[-1] if len(tx) else len(t)-1
                    rec.update(adsb=True,t_first=t[0],t_move=t[i0],stand_start=bool(np.isfinite(d0) and d0<=150 and i0>0),
                               ds_taxi=S[i1]-S[i0],dt_taxi=t[i1]-t[i0],t_taxi_end=t[i1],
                               t_last=t[-1],x_last=x_last,route_len=cum[-1],frac_match=ok.mean(),d0=d0)
        R.append(rec)
    os.makedirs(OUT,exist_ok=True); pd.DataFrame(R).to_parquet(out); return out
if __name__=='__main__':
    apt=sys.argv[1]; months=sys.argv[2].split(',')
    days=[d.strftime('%Y-%m-%d') for m in months for d in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    with Pool(int(sys.argv[3]) if len(sys.argv)>3 else 8) as p:
        for o in p.imap_unordered(dia,[(apt,d) for d in days]): print(o,flush=True)
