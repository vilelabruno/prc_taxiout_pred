"""Extrai, para todos os dias rotulados (jan+jul 2025), voos cuja trilha cruza o stand e tem pushback + parada: ini2 (BLOCK->taxi proprio), par, taxi2 (taxi proprio->MVT)."""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from parada import setup, corridas
from regride import regua, regride
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/anual')
def dia(args):
    apt,day=args; out=f'{O}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    sy,idx,tr=setup(apt); g=sy.g; t0=pd.Timestamp(day,tz='UTC').value//10**9; tt=tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}; R=[]
    for r in tt.itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt+60)]; gd=v[v.alt==0]
        if len(gd)<8 or r.mvt-gd.t.max()>30: continue
        v=v[v.icao24==gd.icao24.iloc[-1]]; gd=v[(v.alt==0)&(v.t<=r.mvt)].reset_index(drop=True)
        if len(gd)<8: continue
        Pp=np.c_[(gd.lon-g.lon0)*g.kx,(gd.lat-g.lat0)*g.ky]; t=gd.t.to_numpy().astype(float); gs=gd.gs.fillna(0).to_numpy()
        Rg=regua(sy,g,idx,r.stand,Pp)
        if Rg is None: continue
        res=regride(t,Pp,gs,Rg,r.mvt)
        if isinstance(res,dict) or res[0].get('cat') not in ('cruzou','cruzou0') or res[0].get('gap',99)>30: continue
        rs=corridas(t,gs)
        if len(rs)<2: continue
        e1,s2=rs[0][1],rs[1][0]; ini1=t[rs[0][0]]-r.blk; par=t[s2]-t[e1]
        if not(-120<=ini1<=180) or par>900: continue
        R.append(dict(day=day,blk=r.blk,mvt=r.mvt,stand=r.stand,rwy=str(r.rwy),ini1=ini1,dur1=t[e1]-t[rs[0][0]],par=par,ini2=t[s2]-r.blk,taxi2=r.mvt-t[s2]))
    os.makedirs(O,exist_ok=True); pd.DataFrame(R).to_parquet(out); return out
if __name__=='__main__':
    apt=sys.argv[1]; days=[d.strftime('%Y-%m-%d') for m in ('2025-01','2025-07') for d in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    with Pool(12) as p:
        for o in p.imap_unordered(dia,[(apt,d) for d in days]): pass
    d=pd.concat([pd.read_parquet(f'{O}/{apt}_{x}.parquet') for x in days]); print(apt,len(d),'voos em',len(days),'dias')
