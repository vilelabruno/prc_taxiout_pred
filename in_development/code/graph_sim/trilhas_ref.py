"""Trilhas observadas (ADS-B) a partir da vizinhanca do stand: para cada voo cuja cadeia chega a <=150 m do stand, todos os pings do ponto de parada ate o MVT,
com tau = tempo desde que saiu da vizinhanca do stand. Tambem grava o ponto de parada (P*) de TODOS os voos com cadeia."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import regride as R, taxigraph as T
S=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); apt=sys.argv[1]
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{S}/regride/{apt}_*.parquet'))]); f=f[f.tem==1]
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','RUNWAY_mvt']).rename(columns={'MVT_ID_mvt':'id'}); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
f=f.merge(tr[['id','cs','RUNWAY_mvt']],on='id'); g=T.TaxiGraph(apt)
def dia(day):
    ff=f[f.day==day]
    if ff.empty: return None,None
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}; gn={}
    for k,v in grp.items(): gn.setdefault(R.norm(k),[]).append(v)
    pings=[];stops=[]
    for r in ff.itertuples():
        q,lig=R.pings(grp,gn,r.cs,r.mvt)
        if q is None or len(q)<=r.i_stop: continue
        P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; t=q.t.to_numpy(); i=int(r.i_stop)
        stops.append(dict(id=r.id,px=P[i,0],py=P[i,1],icao=q.icao24.iloc[-1]))
        if r.D_star<=150:
            pings.append(pd.DataFrame(dict(id=r.id,day=day,stand=r.stand,rwy=r.RUNWAY_mvt,x=P[i:,0],y=P[i:,1],tau=t[i:]-t[i])))
    return (pd.concat(pings) if pings else None),pd.DataFrame(stops)
if __name__=='__main__':
    days=sorted(f.day.unique())
    with Pool(12) as p: res=p.map(dia,days)
    A=pd.concat([r[0] for r in res if r[0] is not None]); B=pd.concat([r[1] for r in res if r[1] is not None and len(r[1])])
    A.to_parquet(f'{S}/trilhas_ref_{apt}.parquet'); B.to_parquet(f'{S}/ponto_parada_{apt}.parquet')
    print(apt,'voos de referencia (cadeia chega <=150 m do stand):',A.id.nunique(),'| pings de referencia:',len(A),'| stands com referencia:',A.stand.nunique(),'| pontos de parada gravados:',len(B))
