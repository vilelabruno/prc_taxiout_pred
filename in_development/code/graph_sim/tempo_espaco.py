import sys, glob, numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import regride as R, taxigraph as T
apt='EGLL'; S=os.path.expanduser('~/Projetos/oof_store'); g=T.TaxiGraph(apt); Rt=T.Router(g); idx=T.build_stand_index(g.A); KT=0.514444
casos=[('2025-07-30','SHT14L'),('2025-01-07','SHT18Z'),('2025-01-17','BAW48HY'),('2025-01-18','BAW4MJ'),('2025-01-20','AUA45XD'),('2025-01-05','SHT18E')]
ref=pd.concat([pd.read_parquet(p) for p in glob.glob(f'{S}/sincronia_grafo/regride/{apt}_*.parquet')]); VBAR=float(ref.v_bar.median())
full=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt'])
full['mvt']=pd.to_datetime(full.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); full['blk']=pd.to_datetime(full.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); full['cs']=full.CALLSIGN_flt.fillna(full.FLIGHT_mvt).str.strip()
dep=full[(full.ADEP_mvt==apt)&(full.PHASE_mvt=='DEP')]; arr=full[(full.ADES_mvt==apt)&(full.PHASE_mvt=='ARR')]
fig,ax=plt.subplots(2,3,figsize=(22,13)); ax=ax.ravel()
for A,(day,cs) in zip(ax,casos):
    t0=pd.Timestamp(day,tz='UTC').value//10**9; r=dep[(dep.cs==cs)&(dep.mvt>=t0)&(dep.mvt<t0+86400)].iloc[0]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip(); grp={k:v for k,v in a.groupby('cs')}; gn={}
    for k,v in grp.items(): gn.setdefault(R.norm(k),[]).append(v)
    R.END_GAP=1e9; q,lig=R.pings(grp,gn,r.cs,r.mvt); R.END_GAP=600.0
    la,lo,lv=T.map_stand(r.STAND_mvt,idx); sp=np.array(g.proj(la,lo)); rt=Rt.route(la,lo,str(r.RUNWAY_mvt)); L=rt['len_full']; end=g.xy[rt['path'][-1]]; d_run=float(np.hypot(*(end-sp)))
    # chegada anterior no mesmo stand
    ap=arr[(arr.STAND_mvt==r.STAND_mvt)&(arr.blk<=r.mvt-120)&(arr.blk>=r.mvt-36*3600)].sort_values('blk'); ap=ap.iloc[-1] if len(ap) else None
    if q is not None:
        X=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; dist=np.hypot(*(X-sp).T); tm=(q.t.to_numpy()-r.mvt)/60; gs=q.gs.to_numpy()*KT
        sc=A.scatter(tm,dist,c=np.clip(gs,0,15),cmap='viridis',s=20,zorder=4,edgecolors='k',linewidths=.2); cb=plt.colorbar(sc,ax=A,fraction=.035); cb.set_label('velocidade no ping (m/s)')
    A.axvline(0,color='#c0392b',lw=2,label='MVT (decolagem)'); A.axvline((r.blk-r.mvt)/60,color='#2c3e50',lw=2,ls='--',label='BLOCK (rotulo)')
    if ap is not None and (r.mvt-ap.blk)<=6*3600: A.axvline((ap.blk-r.mvt)/60,color='#16a085',lw=2,ls=':',label=f'on-block da chegada anterior ({ap.cs})')
    A.axvspan(-10,0,color='#f1c40f',alpha=.18,label='janela de 10 min antes do MVT (cadeia valida exige ping aqui)')
    A.axhline(d_run,color='#7f8c8d',lw=1,ls='-.',label='distancia stand -> pista (reta)')
    bh=(r.mvt-(L/VBAR+807))
    A.scatter([(bh-r.mvt)/60],[0],marker='v',s=140,color='#e67e22',zorder=6,label=f'BLOCK simulado (ancora na pista): y={(L/VBAR+807)/60:.0f} min')
    A.set_xlim(min(-300,(r.blk-r.mvt)/60-20),12); A.set_xlabel('minutos em relacao ao MVT'); A.set_ylabel('distancia reta ao stand (m)')
    A.set_title(f"{day} {cs} stand {r.STAND_mvt} pista {r.RUNWAY_mvt} | y real {(r.mvt-r.blk)/60:.0f} min",fontsize=10); A.legend(fontsize=7,loc='upper left')
plt.tight_layout(); plt.savefig(f'{S}/sincronia_grafo/egll_tempo_espaco.png',dpi=100)
