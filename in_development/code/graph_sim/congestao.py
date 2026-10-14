"""Congestionamento observado: por dia, faixa de 10 min e aresta de taxi, quais avioes (icao24) estavam la, pelo ADS-B de solo de TODOS os avioes."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from sincro2 import Sync2
S=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/congestao'); apt=sys.argv[1]; _s={}
def dia(day):
    out=f'{S}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    if apt not in _s: _s[apt]=Sync2(apt)
    sy=_s[apt]; g=sy.g; p=f'data/processed/adsb_surface/{day}.parquet'
    if not os.path.exists(p): return None
    a=pd.read_parquet(p,columns=['icao24','t','lat','lon','alt','apt'],filters=[('apt','==',apt)]); a=a[a.alt==0]
    X=np.c_[(a.lon.to_numpy()-g.lon0)*g.kx,(a.lat.to_numpy()-g.lat0)*g.ky]; k,_=sy.snap_many(X); a=a.assign(k=k,bin=(a.t.to_numpy()//600).astype('int64'))
    a=a[a.k>=0][['icao24','bin','k']].drop_duplicates(); os.makedirs(S,exist_ok=True); a.to_parquet(out); return out
if __name__=='__main__':
    days=[d.strftime('%Y-%m-%d') for m in sys.argv[2].split(',') for d in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    with Pool(12) as pl: r=pl.map(dia,days)
    print(apt,'dias com congestionamento:',sum(x is not None for x in r))
