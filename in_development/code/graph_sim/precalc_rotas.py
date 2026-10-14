"""Pre-calcula, por dia, as rotas de todas as partidas e chegadas: Dijkstra stand<->pista, cadeia ADS-B desde o MVT (regredindo), e as emissoes de ADS-B (pings por aresta e tempo)."""
import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import sim_dia as SD, sim_run as SR
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/rotas_'+os.environ.get('APT','EGLL')); APT=os.environ.get('APT','EGLL')
def dia(day):
    out=f'{OUT}/{day}.pkl'
    if os.path.exists(out): return out
    d=SD.Dia(APT,day); D,A,Oobs,own=SR.prepara(d)
    keep=['id','cs','mvt','blk','rwy','stand','ok','ed','chain','ed_h','tstar','obs','t_patio','xy','F','Pstar','pings','tp','icao']
    DD=[{k:r.get(k) for k in keep} for r in D]; AA=[{k:r.get(k) for k in ('id','cs','ok','ed','obs','t_in','t_out')} for r in A]
    a=d.a[d.a.k>=0]; tt=a.t.to_numpy(); same=np.r_[a.icao24.to_numpy()[1:]==a.icao24.to_numpy()[:-1],False]; dt=np.where(same,np.minimum(np.r_[tt[1:],(tt[-1]+30 if len(tt) else 0.0)]-tt,30.0),10.0)
    ads=dict(t=tt.astype(float),k=a.k.to_numpy().astype(int),w=dt.astype(float),icao=a.icao24.astype('category').cat.codes.to_numpy(),icao_list=list(a.icao24.astype('category').cat.categories))
    import multi_u as MU
    um,umed,nn=MU.unidade(d); V=d.a.gs.to_numpy()*SD.KT; okv=(d.a.k.to_numpy()>=0)&(V>1)&(V<15); vref=float(np.quantile(V[okv],0.5)) if okv.any() else 9.0
    pickle.dump(dict(day=day,t0=d.t0,D=DD,A=AA,ads=ads,v=d.v,unit=um,vref=vref),open(out,'wb')); return out
if __name__=='__main__':
    days=[x.strftime('%Y-%m-%d') for x in pd.date_range('2025-01-01','2025-12-31')] if os.environ.get('ANO') else [x.strftime('%Y-%m-%d') for m in ('2025-01','2025-07') for x in pd.date_range(m+'-01',periods=pd.Period(m).days_in_month)]
    os.makedirs(OUT,exist_ok=True)
    with Pool(12) as p:
        for o in p.imap_unordered(dia,days): print(o,flush=True)
