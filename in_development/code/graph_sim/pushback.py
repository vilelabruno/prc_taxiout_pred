"""Posicao de saida do pushback no referencial do stand (x ao longo do nariz, y lateral), por stand."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import taxigraph as T
KT=0.514444
apt=sys.argv[1]; days=sys.argv[2].split(','); g=T.TaxiGraph(apt); A=g.A
sinfo={}
for s in A.stands:
    if s['kind']=='tie_down' and 'jets' not in s['equip']: continue
    sinfo.setdefault(T.norm_name(s['name'],apt),(np.array(g.proj(s['lat'],s['lon'])),s['hdg']))
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','AOBT_3_flt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt','RUNWAY_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['aobt']=pd.to_datetime(tr.AOBT_3_flt,utc=True,errors='coerce').dt.as_unit('s').astype('int64').where(tr.AOBT_3_flt.notna());
tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
R=[]
for day in days:
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    for r in tr[(tr.mvt>=t0+3600)&(tr.mvt<t0+23*3600)].itertuples():
        v=grp.get(r.cs); si=sinfo.get(T.norm_name(str(r.STAND_mvt),apt))
        if v is None or si is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt)&(v.alt==0)]
        if len(v)<10: continue
        v=v[v.icao24==v.icao24.iloc[-1]].reset_index(drop=True)
        P=np.c_[(v.lon-g.lon0)*g.kx,(v.lat-g.lat0)*g.ky]; t=v.t.to_numpy(); gs=v.gs.to_numpy()*KT
        sp,hd=si; h=np.radians(90-hd); u=np.array([np.cos(h),np.sin(h)]); w=np.array([-u[1],u[0]])   # u = direcao do nariz (leste=0 graus matematico)
        d=np.hypot(*(P-sp).T)
        if d[0]>150: continue                          # trilha nao comeca no pateio do stand
        # inicio do taxi para frente: primeiro ping com gs>=3 m/s sustentado (3 pings)
        fast=np.where((gs>=3)&(np.r_[gs[1:],0]>=3)&(np.r_[gs[2:],0,0]>=3))[0]
        if not len(fast): continue
        k=fast[0]; lo=max(0,k-1)
        # ultimo ping lento antes do taxi
        slow=np.where(gs[:k]<1.0)[0]; ip=slow[-1] if len(slow) else 0
        rel=lambda p:((p-sp)@u,(p-sp)@w)
        x0,y0=rel(P[0]); xp,yp=rel(P[ip])
        # direcao de saida do taxi (primeiros 5 pings apos k)
        dv=P[min(k+4,len(P)-1)]-P[k]; ang=np.degrees(np.arctan2(dv@w,dv@u)) if np.hypot(*dv)>3 else np.nan
        R.append(dict(mvt=r.mvt,blk=r.blk,cs=r.cs,acft=r.AIRCRAFT_TYPE_mvt,oper=r.AIRCRAFT_OPERATOR_flt,rwy=r.RUNWAY_mvt,t_taxi=t[k],aobt_blk=r.aobt-r.blk,taxi_aobt=t[k]-r.aobt,day=day,stand=r.STAND_mvt,hdg=hd,d0=d[0],x0=x0,y0=y0,dp=np.hypot(xp,yp),xp=xp,yp=yp,ang_saida=ang,
            blk_p0=t[0]-r.blk,blk_pp=t[ip]-r.blk,blk_taxi=t[k]-r.blk,p0_taxi=t[k]-t[0],n_parado=int((gs[:k]<1).sum()),mvt_taxi=r.mvt-t[k]))
d=pd.DataFrame(R); d.to_csv(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/pushback_{apt}.csv'),index=False)
pd.set_option('display.width',200)
print(apt,'voos com trilha no patio',len(d),'stands',d.stand.nunique())
print(d[['d0','x0','y0','dp','xp','yp','ang_saida','blk_p0','blk_taxi','p0_taxi','mvt_taxi']].describe(percentiles=[.1,.25,.5,.75,.9]).T[['count','10%','25%','50%','75%','90%']].round(1))
# repetibilidade por stand (>=6 voos): dispersao da posicao de saida do pushback vs dispersao global
s=d.groupby('stand').filter(lambda q:len(q)>=6)
w=s.groupby('stand').agg(n=('xp','size'),xp=('xp','median'),yp=('yp','median'),sx=('xp',lambda z:(z-z.median()).abs().median()),sy=('yp',lambda z:(z-z.median()).abs().median()),dp=('dp','median'))
print('stands com >=6 voos:',len(w),'| MAD da posicao dentro do stand (m): x',round(w.sx.median(),1),'y',round(w.sy.median(),1),'| MAD global x',round((d.xp-d.xp.median()).abs().median(),1),'y',round((d.yp-d.yp.median()).abs().median(),1))
print(w.sort_values('n',ascending=False).head(12).round(1))
