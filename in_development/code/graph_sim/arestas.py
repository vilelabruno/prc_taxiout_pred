"""Por dia: tempo de cada voo em cada aresta (nos consecutivos da rota casada). dt = chegada ao no i+1 - chegada ao no i,
entao a espera num no entra na aresta que sai dele; as arestas ladrilham o tempo."""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/arestas')
_S={}
def chegada(S,t,x):
    """primeiro instante em que S>=x, interpolado; (tempo, indice anterior) ou (nan,-1)"""
    j=int(np.searchsorted(S,x,'left'))
    if j==0: return (t[0],0) if S[0]==x else (np.nan,-1)
    if j>=len(S): return np.nan,-1
    d=S[j]-S[j-1]; return (t[j-1]+(x-S[j-1])/d*(t[j]-t[j-1]) if d>0 else t[j]), j-1
def dia(args):
    apt,day=args; out=f'{OUT}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    if apt not in _S: _S[apt]=Sync(apt)
    sy=_S[apt]; g=sy.g
    tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt'])
    tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
    tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
    tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
    t0=pd.Timestamp(day,tz='UTC').value//10**9; tr=tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}; R=[]
    for r in tr.itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt+60)]; gd=v[v.alt==0]
        if len(gd)<8 or r.mvt-gd.t.max()>30: continue
        v=v[v.icao24==gd.icao24.iloc[-1]]; gd=v[(v.alt==0)&(v.t<=r.mvt)]
        if len(gd)<8: continue
        seg=(gd.t.diff().fillna(0)>300).cumsum(); gd=gd[seg==seg.iloc[-1]].reset_index(drop=True)
        Pp=np.c_[(gd.lon-g.lon0)*g.kx,(gd.lat-g.lat0)*g.ky]; t=gd.t.to_numpy().astype(float)
        ed=np.array([x[0] for x in sy.snap(Pp)]); ok=ed>=0
        if ok.sum()<5: continue
        nodes=sy.route(list(ed[ok])); S,_,cum=sy.along(nodes,Pp); S=np.maximum.accumulate(S)
        arr=[chegada(S,t,x) for x in cum]; dt_pts=np.diff(t)
        for i in range(len(nodes)-1):
            (ta,ja),(tb,jb)=arr[i],arr[i+1]
            if ja<0 or jb<0: continue
            gap=dt_pts[ja:jb+1].max() if jb>=ja and len(dt_pts) else 0
            R.append((r.mvt,str(r.RUNWAY_mvt),min(nodes[i],nodes[i+1]),max(nodes[i],nodes[i+1]),cum[i+1]-cum[i],ta,tb-ta,gap))
    os.makedirs(OUT,exist_ok=True)
    pd.DataFrame(R,columns=['mvt','rwy','na','nb','L','t','dt','gap']).to_parquet(out); return out
if __name__=='__main__':
    apt=sys.argv[1]; months=sys.argv[2].split(',')
    days=[d.strftime('%Y-%m-%d') for m in months for d in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    with Pool(8) as p:
        for o in p.imap_unordered(dia,[(apt,d) for d in days]): pass
    print(apt,'ok',flush=True)
