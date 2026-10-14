"""Regressao desde o MVT pelos pings de solo do voo ate o ponto ADS conhecido mais proximo do stand SEM ultrapassa-lo.
Distancia ao stand: por Dijkstra no grafo quando o ping casa numa aresta (taxiway/pista); euclidiana quando esta no patio (sem aresta).
Anda para tras ping a ping; para quando a distancia ao stand volta a subir (ja passou do stand: pings da chegada anterior ficam fora)."""
import sys, os, re, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
import taxigraph as T
KT=0.514444; HIST=40.0; END_GAP=600.0; CHAIN_GAP=900.0; OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/regride'); _S={}
norm=lambda c: re.sub(r'(?<=\d)[A-Z]{1,2}$','',c) if isinstance(c,str) else ''
def pings(grp,gn,cs,mvt):
    q=grp.get(cs)
    def pick(q):
        q=q[(q.alt==0)&(q.t<=mvt)&(q.t>mvt-4*3600)]
        if len(q)<2: return None
        q=q[q.icao24==q.icao24.iloc[-1]].sort_values('t').reset_index(drop=True)
        return q if mvt-q.t.iloc[-1]<=END_GAP else None          # a cadeia tem que comecar no MVT
    r=pick(q) if q is not None else None
    if r is not None: return r,1
    best=None
    for qq in gn.get(norm(cs),[]):
        for ic,x in qq[(qq.alt==0)&(qq.t<=mvt+60)].groupby('icao24'):
            de=mvt-x.t.max()
            if -60<=de<=180 and (best is None or de<best[0]): best=(de,x)
    if best is not None:
        r=pick(best[1]); 
        if r is not None: return r,2
    return None,0
def regride(sy,Rt,la,lo,P,t,gs):
    """P: pings em ordem de tempo (x,y m). Devolve indice do ponto de parada (mais proximo do stand no caminho de volta) e distancias usadas."""
    g=sy.g; sp=np.array(g.proj(la,lo)); opts,_=Rt.attach(la,lo); n=len(P)
    def dist(i):
        e=sy.snap(P[i:i+1])[0]
        if e[0]>=0:
            L=sy.L[e[0]]; best=np.inf
            for nd,acc in opts: best=min(best,acc+e[1]*L+sy.D[nd,sy.ea[e[0]]],acc+(1-e[1])*L+sy.D[nd,sy.eb[e[0]]])
            return best,1
        return float(np.hypot(*(P[i]-sp))),0
    best_i=n-1; best_d,_=dist(n-1); D=[None]*n; D[n-1]=best_d
    for i in range(n-2,-1,-1):
        if t[i+1]-t[i]>CHAIN_GAP: break                   # lacuna maior que 15 min: a cadeia quebra
        d,_=dist(i); D[i]=d
        if d<best_d: best_d,best_i=d,i
        elif d>best_d+HIST and best_d<d-HIST: break       # distancia ao stand subiu: ja passou do stand
    return best_i,best_d,D
def dia(args):
    apt,day=args; out=f'{OUT}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    if apt not in _S:
        sy=Sync(apt); _S[apt]=(sy,T.Router(sy.g),T.build_stand_index(sy.g.A))
    sy,Rt,idx=_S[apt]; g=sy.g
    tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt'])
    tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
    tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
    t0=pd.Timestamp(day,tz='UTC').value//10**9; tr=tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)]
    p=f'data/processed/adsb_surface/{day}.parquet'
    a=pd.read_parquet(f'{p}',filters=[('apt','==',apt)]).sort_values('t') if os.path.exists(p) else pd.DataFrame(columns=['callsign','t','alt','icao24','lat','lon','gs'])
    a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}; gn={}
    for k,v in grp.items(): gn.setdefault(norm(k),[]).append(v)
    R=[]
    for r in tr.itertuples():
        rec=dict(id=r.MVT_ID_mvt,day=day,mvt=r.mvt,blk_rotulo=r.blk,tem=0,stand=r.STAND_mvt)
        la,lo,lv=T.map_stand(r.STAND_mvt,idx); rec['stand_nivel']=lv
        q,lig=pings(grp,gn,r.cs,r.mvt); rec['lig']=lig
        if q is not None and lv<9:
            P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; t=q.t.to_numpy(); gs=q.gs.to_numpy()*KT
            i,d,D=regride(sy,Rt,la,lo,P,t,gs)
            seg=slice(i,len(P)); dt=np.diff(t[seg]); dd=np.hypot(*np.diff(P[seg],axis=0).T); mov=(dt>0)&(dt<=30)&(gs[seg][1:]>1)&(gs[seg][1:]<=18)
            rec.update(tem=1,n_total=len(P),n_usados=len(P)-i,i_stop=i,t_star=t[i],D_star=d,gs_star=gs[i],mvt_star=r.mvt-t[i],t_first=t[0],mvt_prim=r.mvt-t[0],
                       parou=int(i>0),v_bar=float(dd[mov].sum()/dt[mov].sum()) if mov.sum()>=2 else np.nan,t_mov=float(dt[mov].sum()),
                       esp_visto=float(dt[(gs[seg][1:]<1)&(dt<=600)].sum()),d_stand_eucl=float(np.hypot(*(P[i]-np.array(g.proj(la,lo))))))
        R.append(rec)
    os.makedirs(OUT,exist_ok=True); pd.DataFrame(R).to_parquet(out); return out
if __name__=='__main__':
    apt=sys.argv[1]; months=sys.argv[2].split(',')
    days=[d.strftime('%Y-%m-%d') for m in months for d in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    with Pool(int(sys.argv[3]) if len(sys.argv)>3 else 8) as pool:
        for o in pool.imap_unordered(dia,[(apt,d) for d in days]): print(o,flush=True)
