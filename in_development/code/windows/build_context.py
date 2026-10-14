"""Contexto por decolagem (TODOS os voos): METAR sincronizado ao MVT (+ altura AGL pelo QNH ao longo da subida típica da pista, teto, vento relativo à pista), carga por pista (decolagens e chegadas em janelas), hora/dia, setor de destino."""
import glob,re,numpy as np,pandas as pd
from metar_sync import *
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/data'; AP=['EGLL','EDDF','EDDM','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
apx=pd.read_csv(R+'/external/ourairports/airports.csv',usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident')
C={'EGLL':(51.4706,-0.4619),'EDDF':(50.0333,8.5706),'EDDM':(48.3538,11.7861),'EHAM':(52.3086,4.7639),'LEBL':(41.2971,2.0785),'LEMD':(40.4719,-3.5626),'LFPG':(49.0097,2.5479),'LIRF':(41.8003,12.2389),'LSZH':(47.4647,8.5492),'LTFM':(41.2753,28.7519)}
T=lambda s: pd.to_datetime(s,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
parts=[]
for m in range(1,13):
    f=glob.glob(R+'/raw/training_2025-%02d-01_*.parquet'%m)[0]
    r=pd.read_parquet(f,columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']); r['mvt']=T(r.MVT_TIME_UTC_mvt); parts.append(r.drop(columns='MVT_TIME_UTC_mvt'))
r=pd.concat(parts,ignore_index=True); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64'); out=[]
for a in AP:
    dep=r[(r.PHASE_mvt=='DEP')&(r.ADEP_mvt==a)].sort_values('mvt').reset_index(drop=True); arr=r[(r.PHASE_mvt=='ARR')&(r.ADES_mvt==a)].sort_values('mvt')
    M=load_metar(R+f'/external/metar_iem/{a}.csv'); w=metar_at(M,dep.mvt.values); la1,lo1=np.radians(C[a])
    def brg(z):
        if z not in apx.index: return np.nan
        la2,lo2=np.radians(apx.latitude_deg[z]),np.radians(apx.longitude_deg[z]); return (np.degrees(np.arctan2(np.sin(lo2-lo1)*np.cos(la2),np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(lo2-lo1)))+360)%360
    dep['sec']=np.floor(((dep.ADES_mvt.map(brg)+22.5)%360)/45); hd=dep.RUNWAY_mvt.map(lambda s:(int(re.findall(r'\d+',str(s))[0])*10)%360 if re.findall(r'\d+',str(s)) else np.nan).values; th=np.radians(np.where(np.isfinite(hd),hd,0.0))
    d=pd.DataFrame({'MVT_ID_mvt':dep.MVT_ID_mvt,'apt':a,'rwy':dep.RUNWAY_mvt,'sec':dep.sec,'mvt':dep.mvt})
    for c in ('qnh','temp','dew','wspd','gust','ceil','vis','ra','sn','fz','fg','ts','lvp','age_min'): d['wx_'+c]=w[c]
    d['wx_hw_rwy']=-(w['wu']*np.sin(th)+w['wv']*np.cos(th)); d['wx_xw_rwy']=w['wu']*np.cos(th)-w['wv']*np.sin(th); d['wx_spread']=w['temp']-w['dew']
    # altura AGL ao longo da subida típica (perfil mediano barométrico da pista, nominal) com o QNH da hora; relação com o teto
    prof={60:600,180:4200,300:7400}   # AGL típico de referência (aprox. medido no EGLL); ajustado pelo QNH abaixo
    for s,agl_ref in prof.items():
        alt_b=agl_ref+ELEV[a]-(1013.25-1013.25)*27.0; d[f'agl_{s}']=alt_b+(1013.25-w['qnh'])*27.0-ELEV[a]; d[f'incloud_{s}']=(d[f'agl_{s}']>=w['ceil']).astype(float)
    d['frac_nuvem_subida']=d[['incloud_60','incloud_180','incloud_300']].mean(1)
    # carga por pista (decolagens e chegadas) e do aeroporto, janelas centradas no MVT
    tdep=dep.mvt.values; tarr=arr.mvt.values
    for W in (300,900,1800):
        d[f'ndep_apt_{W}']=np.searchsorted(tdep,tdep+W,'right')-np.searchsorted(tdep,tdep-W,'left')-1; d[f'narr_apt_{W}']=np.searchsorted(tarr,tdep+W,'right')-np.searchsorted(tarr,tdep-W,'left')
    for rw,g in dep.groupby('RUNWAY_mvt'):
        ix=g.index.values; tr=np.sort(g.mvt.values)
        for W in (300,900,1800): d.loc[ix,f'ndep_rwy_{W}']=np.searchsorted(tr,g.mvt.values+W,'right')-np.searchsorted(tr,g.mvt.values-W,'left')-1
        pv=np.searchsorted(tr,g.mvt.values,'left')-1; nx=np.searchsorted(tr,g.mvt.values,'right')
        d.loc[ix,'hw_prev']=np.where(pv>=0,g.mvt.values-tr[np.clip(pv,0,None)],np.nan); d.loc[ix,'hw_next']=np.where(nx<len(tr),tr[np.clip(nx,0,len(tr)-1)]-g.mvt.values,np.nan)
        ar=arr[arr.RUNWAY_mvt==rw].mvt.values
        for W in (300,900): d.loc[ix,f'narr_rwy_{W}']=np.searchsorted(ar,g.mvt.values+W,'right')-np.searchsorted(ar,g.mvt.values-W,'left')
    d['hr']=((d.mvt%86400)//3600).astype(int); d['dow']=(((d.mvt//86400)+3)%7).astype(int); out.append(d)
X=pd.concat(out,ignore_index=True); X.to_parquet('context_all.parquet'); print(X.shape, X.isna().mean().round(2).to_dict())
