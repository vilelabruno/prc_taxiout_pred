"""Liga cada partida ao icao24 PELA FISICA: avião em cima da pista, com gs>=50 kt, no instante do MVT (e nao pelo indicativo)."""
import sys, os, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src/physical'); import taxigraph as T
from scipy.spatial import cKDTree
KT=0.514444; norm=lambda c: re.sub(r'(?<=\d)[A-Z]{1,2}$','',c)
def prepara(apt):
    g=T.TaxiGraph(apt); seg={}
    for r in g.A.runways:
        p1=np.array(g.proj(r['lat1'],r['lon1'])); p2=np.array(g.proj(r['lat2'],r['lon2']))
        seg[r['end1']]=(p1,p2); seg[r['end2']]=(p1,p2)
    return g,seg
def dseg(X,p1,p2):
    ab=p2-p1; t=np.clip(((X-p1)@ab)/(ab@ab),0,1); return np.hypot(*(X-(p1+t[:,None]*ab)).T)
def liga_dia(apt,day,g,seg,tr):
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    a['x']=(a.lon-g.lon0)*g.kx; a['y']=(a.lat-g.lat0)*g.ky
    icao_cs={k:v for k,v in a.groupby('cs')}; out=[]
    for r in tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)].itertuples():
        sg=seg.get(str(r.RUNWAY_mvt)); phys=None; ncand=0
        if sg is not None:
            w=a[(a.t>=r.mvt-120)&(a.t<=r.mvt+150)&(a.alt<=100)&(a.gs>=30)]
            if len(w):
                w=w[dseg(w[['x','y']].to_numpy(),*sg)<=50]
                best=None
                for ic,q in w.groupby('icao24'):
                    q=q.sort_values('t')
                    if len(q)<2 or q.gs.max()<80 or q.gs.iloc[-1]<q.gs.iloc[0]+15: continue      # tem que estar ACELERANDO e passar de 80 kt
                    gs=q.gs.to_numpy(); t=q.t.to_numpy(); c=np.where((gs[:-1]<80)&(gs[1:]>=80))[0]
                    t80=np.interp(80,gs[c[0]:c[0]+2],t[c[0]:c[0]+2]) if len(c) else t[np.argmax(gs>=80)]
                    ncand+=1; sc=abs(t80-(r.mvt+ (-20 if apt=='EGLL' else 25)))
                    if best is None or sc<best[0]: best=(sc,ic)
                phys=best[1] if best else None
        cs_ic=None; v=icao_cs.get(r.cs)
        if v is not None:
            v=v[(v.t<=r.mvt)&(v.t>r.mvt-5400)&(v.alt==0)]
            if len(v): cs_ic=v.icao24.iloc[-1]
        out.append(dict(id=r.id,day=day,cs=r.cs,mvt=r.mvt,icao_fis=phys,ncand=ncand,icao_cs=cs_ic))
    return pd.DataFrame(out)
if __name__=='__main__':
    apt=sys.argv[1]; g,rt=prepara(apt)
    tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt']).rename(columns={'MVT_ID_mvt':'id'})
    tr=tr[(tr.ADEP_mvt==apt)&(tr.PHASE_mvt=='DEP')].copy(); tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
    R=pd.concat([liga_dia(apt,d,g,rt,tr) for d in sys.argv[2].split(',')]); R['mes']=R.day.str[:7]
    R.to_parquet(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/link_{apt}.parquet'))
    for m,q in R.groupby('mes'):
        f=q.icao_fis.notna(); c=q.icao_cs.notna(); both=f&c
        print(f'{apt} {m}: partidas {len(q)} | com aviao na pista no MVT (fisica): {f.mean():.3f} | com ligacao por indicativo: {c.mean():.3f} | nos que tem os dois: MESMO aviao {(q.icao_fis[both]==q.icao_cs[both]).mean():.3f} (n={both.sum()}) | >1 candidato na pista: {(q.ncand>1).mean():.3f}')
