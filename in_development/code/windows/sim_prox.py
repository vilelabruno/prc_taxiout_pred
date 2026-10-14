"""Simulação de proximidade na origem. Para uma decolagem (apt, MVT): acha a trilha ADS-B do próprio voo, e desloca essa mesma trajetória no tempo (d = -30..+30 min, passo 30 s);
para cada d mede, contra TODAS as outras aeronaves no ar (trilhas do dia), a menor distância horizontal (NM) com separação vertical < 1000 ft e sem filtro vertical, e quantas aeronaves chegam a < R NM.
Uso: python sim_prox.py APT AAAA-MM-DD MVT_EPOCH [DIR_TRILHAS]"""
import sys, os, numpy as np, pandas as pd
APT,DAY,MVT=sys.argv[1],sys.argv[2],float(sys.argv[3]); DIR=os.path.expanduser(sys.argv[4] if len(sys.argv)>4 else '~/tracks_full')
ap=pd.read_csv(os.path.expanduser('~/yearabl/airports.csv'),usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident'); la0,lo0=ap.latitude_deg[APT],ap.longitude_deg[APT]
o=pd.read_parquet(f'{DIR}/trf_{DAY}.parquet'); o=o[o.apt==APT].sort_values(['icao','t']).reset_index(drop=True)
KX=111.32*np.cos(np.radians(la0)); KY=110.57; NM=1.852
o['x']=((o.lon-lo0)*KX).astype('float32'); o['y']=((o.lat-la0)*KY).astype('float32')
DT=5; T0=np.floor(o.t.min()/DT)*DT; NB=int((o.t.max()-T0)//DT)+2
ids=o.icao.unique(); P=np.full((NB,len(ids),3),np.nan,np.float32); cs={}
for j,(ic,g) in enumerate(o.groupby('icao',sort=False)):
    t=g.t.values; b0=int((t[0]-T0)//DT); b1=int((t[-1]-T0)//DT)+1; tb=T0+np.arange(b0,b1+1)*DT
    for c,col in enumerate(('x','y','alt')): P[b0:b1+1,j,c]=np.interp(tb,t,g[col].values)
    gap=np.diff(t)>30; 
    if gap.any():   # buracos > 30 s ficam NaN
        for a,b in zip(t[:-1][gap],t[1:][gap]): P[int((a-T0)//DT)+1:int((b-T0)//DT)+1,j,:]=np.nan
    s=g.callsign.dropna(); cs[j]=s.iloc[0].strip() if len(s) else g.icao.iloc[0]
icl=list(o.groupby('icao',sort=False).groups.keys()); assert len(icl)==len(ids)
# trilha do próprio voo: ponto a <5 km do aeroporto, alt<2500 ft, |t-MVT|<180 s, subindo; o mais próximo em tempo
c=o[(np.hypot(o.x,o.y)<5)&(o.alt<2500)&((o.t-MVT).abs()<180)]
if len(c)==0: print('SEM trilha própria perto de MVT'); sys.exit(2)
own_ic=c.iloc[(c.t-MVT).abs().argsort().iloc[0]].icao; oj=icl.index(own_ic); print('próprio:',own_ic,cs[oj],'ponto mais próximo em MVT %+.0f s'%((c[c.icao==own_ic].t-MVT).abs().min()))
K=int(900/DT)+1; b_mvt=int((MVT-T0)//DT); own=P[b_mvt:b_mvt+K,oj,:]
print('trajetória própria (15 min após MVT): pontos válidos',int(np.isfinite(own[:,0]).sum()),'de',K,'| alcança %.1f km da origem'%np.nanmax(np.hypot(own[:,0],own[:,1])))
mask=np.ones(len(ids),bool); mask[oj]=False; Oth=P[:,mask,:]; ocs=[cs[j] for j in range(len(ids)) if mask[j]]
rows=[]
for d in np.arange(float(os.environ.get('DMIN',-30)),float(os.environ.get('DMAX',30))+1e-6,0.5):
    bd=int(round((MVT+d*60-T0)/DT)); ks=np.arange(K); tb=bd+ks; ok=(tb>=0)&(tb<NB)&np.isfinite(own[:,0]); 
    if ok.sum()<20: rows.append((d,)+(np.nan,)*5); continue
    Ot=Oth[tb[ok]]; dx=Ot[:,:,0]-own[ok,0][:,None]; dy=Ot[:,:,1]-own[ok,1][:,None]; h=np.hypot(dx,dy)/NM; v=np.abs(Ot[:,:,2]-own[ok,2][:,None])
    h=np.where(np.isfinite(h),h,np.inf); hv=np.where(v<1000,h,np.inf)
    ia=np.unravel_index(np.argmin(h),h.shape); iv=np.unravel_index(np.argmin(hv),hv.shape)
    rows.append((d,h.min(),hv.min(),int((hv.min(axis=0)<5).sum()),int((h.min(axis=0)<5).sum()),ocs[iv[1]] if np.isfinite(hv.min()) else ''))
r=pd.DataFrame(rows,columns=['d','dmin_NM','dmin_NM_vert1000','n_acft_<5NM_vert1000','n_acft_<5NM','mais_proximo_v1000']); r.to_csv(os.path.expanduser(f'~/yearabl/prox_{APT}_{DAY}_{int(MVT)}.csv'),index=False)
print(r[(r.d%2==0)|(r.d.abs()<=2)].round(2).to_string(index=False))
