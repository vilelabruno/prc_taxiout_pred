"""BLOCK -> inicio do taxi em A (BLOCK->1o movimento), B (pushback ate parar), C (espera ate andar para frente). Voos cuja trilha comeca parada no stand."""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import taxigraph as T
KT=0.514444
apt=sys.argv[1]; days=sys.argv[2].split(','); g=T.TaxiGraph(apt); idx=T.build_stand_index(g.A)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
R=[]
for day in days:
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    for r in tr[(tr.mvt>=t0+3600)&(tr.mvt<t0+23*3600)].itertuples():
        v=grp.get(r.cs); la,lo,lv=T.map_stand(r.STAND_mvt,idx)
        if v is None or lv>2: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt)&(v.alt==0)]
        if len(v)<10: continue
        v=v[v.icao24==v.icao24.iloc[-1]].reset_index(drop=True)
        P=np.c_[(v.lon-g.lon0)*g.kx,(v.lat-g.lat0)*g.ky]; t=v.t.to_numpy(); gs=v.gs.to_numpy()*KT
        if np.hypot(*(P[0]-np.array(g.proj(la,lo))))>20 or gs[0]>0.5: continue      # comeca parado no stand
        dis=np.hypot(*(P-P[0]).T); j25=np.where(dis>=25)[0]
        if not len(j25): continue
        pk=np.where(dis[:j25[0]]<=4)[0]
        if not len(pk): continue
        io=pk[-1]+1; onset=t[io]; gap=t[io]-t[io-1] if io>0 else np.nan
        fast=np.where((gs>=3)&(np.r_[gs[1:],0]>=3)&(np.r_[gs[2:],0,0]>=3))[0]
        if not len(fast) or fast[0]<=io: continue
        k=fast[0]
        stop=np.where(gs[io:k]<0.5)[0]; ist=io+stop[0] if len(stop) else io
        # tempo parado apos o pushback: do primeiro ping parado depois do onset ate k
        R.append(dict(blk=r.blk,mvt=r.mvt,t_prev=t[io-1] if io>0 else np.nan,t_io=t[io],t_first=t[0],blk_first=t[0]-r.blk,gap_onset=gap,A=onset-r.blk,B=t[ist]-onset,C=t[k]-t[ist],total=t[k]-r.blk,onset_gap=gap,n_pings_pb=ist-io+1,mvt_taxi=r.mvt-t[k]))
d=pd.DataFrame(R); pd.set_option('display.width',200)
print(apt,'voos que comecam parados no stand:',len(d))
q=d[(d.onset_gap<=12)&(d.A.abs()<900)&(d.C<900)]; print('com o inicio do pushback medido a +-12 s:',len(q))
print(q[['A','B','C','total']].describe(percentiles=[.1,.25,.5,.75,.9]).T[['mean','std','10%','25%','50%','75%','90%']].round(1))
print('variancia: A',round(q.A.var()),'B',round(q.B.var()),'C',round(q.C.var()),'total',round(q.total.var()),'| 2cov(A,B+C)',round(2*np.cov(q.A,q.B+q.C)[0,1]))
print('corr A~(B+C)',round(q.A.corr(q.B+q.C),2),' corr B~C',round(q.B.corr(q.C),2))
print('A dentro de +-15 s:',round((q.A.abs()<=15).mean(),2),'+-30 s:',round((q.A.abs()<=30).mean(),2),'+-60 s:',round((q.A.abs()<=60).mean(),2))
d.to_csv(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/componentes_{apt}.csv'),index=False)
