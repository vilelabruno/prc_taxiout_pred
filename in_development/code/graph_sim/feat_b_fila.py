"""Fila REAL na cabeceira por partida (todo o ano, 1 aeroporto): avioes OUTROS parados (gs<2 kt) perto do no-ancora da pista do voo (grafo), do ADS-B de solo, nos minutos antes do MVT.
Colunas (R em {600,1000} m): n_par_R (avioes distintos, 15 min), t_par_R (s parados somados, 15 min), n_par5_R (5 min), t_par30_R (s parados, 30 min). NaN quando o dia nao tem cobertura de solo perto da cabeceira (cob_cab_dia < 20 pings).
Usa so ADS-B dos outros voos ate o MVT do proprio voo; nunca BLOCK, TAXITIME, y."""
import os,sys,glob,pickle,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import sim_dia as SD
from multiprocessing import Pool
APT=os.environ.get('APT','EGLL'); R0=os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/rotas_{APT}'); OUT=os.path.expanduser('~/Projetos/prc_data_challenge/work/ablation/b'); RS=(600.,1000.)
def dia(day):
    d=SD.Dia(APT,day); g=d.g; a=d.a; Dia=pickle.load(open(f'{R0}/{day}.pkl','rb')); rows=[]; cache={}
    if len(a):
        a=a.sort_values('t'); tt=a.t.to_numpy(); nx=np.r_[tt[1:],tt[-1]+30]; same=np.r_[a.icao24.to_numpy()[1:]==a.icao24.to_numpy()[:-1],False]; w=np.where(same,np.minimum(nx-tt,30.0),10.0)
        X=a.x.to_numpy(); Y=a.y.to_numpy(); gs=a.gs.to_numpy(); ic=a.icao24.to_numpy()
    for r in Dia['D']:
        row=dict(MVT_ID_mvt=r['id'],day=day)
        an=d.anchor(r['rwy']) if r['ok'] else None
        if an is None or not len(a): rows.append(row); continue
        if r['rwy'] not in cache:
            xy=g.xy[an]; dist=np.hypot(X-xy[0],Y-xy[1]); cache[r['rwy']]={R:np.where((gs<2)&(dist<=R))[0] for R in RS}; cache[r['rwy']]['cob']=int((dist<=RS[1]).sum())
        c=cache[r['rwy']]; row['cob_cab_dia']=c['cob']; own=r.get('icao') or r.get('obs')
        for R in RS:
            ix=c[R]; t=tt[ix]; k=int(R)
            if c['cob']<20: row.update({f'n_par_{k}':np.nan,f't_par_{k}':np.nan,f'n_par5_{k}':np.nan,f't_par30_{k}':np.nan}); continue
            def jan(sec): 
                lo=np.searchsorted(t,r['mvt']-sec); hi=np.searchsorted(t,r['mvt']); j=ix[lo:hi]; return j[ic[j]!=own] if own is not None else j
            j15=jan(900); j5=jan(300); j30=jan(1800)
            row[f'n_par_{k}']=len(set(ic[j15])); row[f't_par_{k}']=float(w[j15].sum()); row[f'n_par5_{k}']=len(set(ic[j5])); row[f't_par30_{k}']=float(w[j30].sum())
        rows.append(row)
    return pd.DataFrame(rows)
if __name__=='__main__':
    days=sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{R0}/20*.pkl'))
    with Pool(int(os.environ.get('NP','6'))) as p: rs=[x for x in p.imap(dia,days) if len(x)]
    df=pd.concat(rs,ignore_index=True); df['MVT_ID_mvt']=df.MVT_ID_mvt.astype('int64'); df.to_parquet(f'{OUT}/feat_b_fila_{APT}.parquet'); print(APT,len(days),'dias',len(df),'voos | NaN n_par_600: %.2f'%df.n_par_600.isna().mean(),flush=True)
