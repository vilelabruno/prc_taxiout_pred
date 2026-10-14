"""Indicador de pista fechado normalizado por (aeroporto x pista x setor de direção x hora x dia da semana) -> percentil mapeado em [0,1;0,9] = peso dos 7 min.
3 passos para trás (MVT, MVT-15, MVT-30 min): w_j; só soma se o voo já estava pronto (ETOT <= s_j + 5 min). taxi_adj = EXOT + 420 * sum_j ready_j * w_j."""
import sys,glob,numpy as np,pandas as pd
MODE=sys.argv[1] if len(sys.argv)>1 else '2025'
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/data'; T=lambda s: pd.to_datetime(s,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
cal=pd.read_csv('rwy_calib.csv'); CALD={(x.apt,x.rwy):(min(x.b_dep,120.0),min(x.b_arr,120.0)) for x in cal.itertuples()}; APD=cal.groupby('apt')[['b_dep','b_arr']].median().clip(upper=120.0)
if MODE=='2025':
    parts=[pd.read_parquet(glob.glob(R+'/raw/training_2025-%02d-01_*.parquet'%m)[0],columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']) for m in range(1,13)]; r=pd.concat(parts,ignore_index=True)
else: r=pd.read_parquet(R+'/raw/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt'])
r['mvt']=T(r.MVT_TIME_UTC_mvt); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64'); r['apt']=np.where(r.PHASE_mvt=='DEP',r.ADEP_mvt,r.ADES_mvt)
K=np.arange(1,113)   # minutos antes do MVT: 1..112 (8 janelas de 7 min a cada 15 min)
ids=[];cl=[]
for (a,rw),g in r.groupby(['apt','RUNWAY_mvt']):
    d=g[g.PHASE_mvt=='DEP'].sort_values('mvt'); ar=g[g.PHASE_mvt=='ARR'].sort_values('mvt')
    if len(d)<20: continue
    bd,ba=CALD.get((a,rw),(float(APD.loc[a,'b_dep']),float(APD.loc[a,'b_arr']))) if a in APD.index else (51.0,60.0); tD=d.mvt.values; tA=ar.mvt.values
    def near(times,q,own=None):
        n=len(times)
        if n==0: return np.full(q.shape,np.inf)
        idx=np.searchsorted(times,q); best=np.full(q.shape,np.inf)
        for off in (-2,-1,0,1):
            j=np.clip(idx+off,0,n-1); dd=np.abs(times[j]-q)
            if own is not None: dd=np.where(j==own,np.inf,dd)
            best=np.minimum(best,dd)
        return best
    S=tD[:,None]-K[None,:]*60.0; own=np.arange(len(tD))[:,None]*np.ones((1,len(K)),int)
    ids.append(d.MVT_ID_mvt.values); cl.append(((near(tD,S,own)<bd)|(near(tA,S)<ba)))
ids=np.concatenate(ids); C=np.concatenate(cl)
# indicador: fração de minutos fechados nos 7 min que antecedem s_j (s_0=MVT, s_1=MVT-15, s_2=MVT-30)
Ij=[C[:,15*j:15*j+7].mean(1) for j in range(8)]    # colunas 0..6 = minutos 1..7 ; 15..21 = minutos 16..22 ; 30..36 = minutos 31..37
F=pd.DataFrame({'MVT_ID_mvt':ids,**{f'I{j}':Ij[j] for j in range(8)}})
# contexto de normalização: aeroporto x pista x setor x hora x dia da semana
X=pd.read_parquet('context_all.parquet',columns=['MVT_ID_mvt','apt','rwy','sec','hr','dow','mvt']); X['MVT_ID_mvt']=X.MVT_ID_mvt.astype('int64')
if MODE=='2026':
    r2=pd.read_parquet(R+'/raw/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']); r2=r2[r2.PHASE_mvt=='DEP'].copy(); r2['MVT_ID_mvt']=r2.MVT_ID_mvt.astype('int64'); r2['mvt']=T(r2.MVT_TIME_UTC_mvt); r2['apt']=r2.ADEP_mvt; r2['rwy']=r2.RUNWAY_mvt
    rk=pd.read_parquet('rank_table.parquet',columns=['MVT_ID_mvt','ADES_mvt']); 
    ap=pd.read_csv(R+'/external/ourairports/airports.csv',usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident'); C0={'EGLL':(51.4706,-0.4619),'EDDF':(50.0333,8.5706),'EDDM':(48.3538,11.7861),'EHAM':(52.3086,4.7639),'LEBL':(41.2971,2.0785),'LEMD':(40.4719,-3.5626),'LFPG':(49.0097,2.5479),'LIRF':(41.8003,12.2389),'LSZH':(47.4647,8.5492),'LTFM':(41.2753,28.7519)}
    def brg(a,z):
        if z not in ap.index: return np.nan
        la1,lo1=np.radians(C0[a]); la2,lo2=np.radians(ap.latitude_deg[z]),np.radians(ap.longitude_deg[z]); return (np.degrees(np.arctan2(np.sin(lo2-lo1)*np.cos(la2),np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(lo2-lo1)))+360)%360
    r2['sec']=np.floor(((np.array([brg(a,z) for a,z in zip(r2.apt,r2.ADES_mvt)])+22.5)%360)/45); r2['hr']=((r2.mvt%86400)//3600).astype(int); r2['dow']=(((r2.mvt//86400)+3)%7).astype(int); X=r2[['MVT_ID_mvt','apt','rwy','sec','hr','dow','mvt']]
F=F.merge(X,on='MVT_ID_mvt',how='left'); F['sec']=F.sec.fillna(-1)
KEYS=[['apt','rwy','sec','hr','dow'],['apt','rwy','hr','dow'],['apt','rwy','hr'],['apt','hr'],['apt']]
# ECDF de referência de 2025 por célula (mesma para 2025 e 2026)
if MODE=='2025': REF=F
else:
    F25=pd.read_parquet('ctotnorm8_ref25.parquet'); REF=F25
def pct(col):
    out=np.full(len(F),np.nan)
    for keys in KEYS:
        gref={k:np.sort(g[col].values) for k,g in REF.groupby(keys) if len(g)>=40}
        m=np.isnan(out)
        if not m.any(): break
        idx=F[m].groupby(keys).indices; sub=F[m]
        for k,ix in idx.items():
            if k in gref: v=sub[col].values[ix]; ref=gref[k]; lo=np.searchsorted(ref,v,'left'); hi=np.searchsorted(ref,v,'right'); out[np.where(m)[0][ix]]=(lo+hi)/2/len(ref)
    return out
for j in range(8): F[f'p{j}']=pct(f'I{j}'); F[f'w{j}']=0.1+0.8*F[f'p{j}'].fillna(0.5)
if MODE=='2025': F[['MVT_ID_mvt','apt','rwy','sec','hr','dow']+[f'I{j}' for j in range(8)]].to_parquet('ctotnorm8_ref25.parquet')
cw=pd.read_parquet(f'ctotwin_{MODE}.parquet',columns=['MVT_ID_mvt','cw_exot','cw_E_wait_min']); cw['MVT_ID_mvt']=cw.MVT_ID_mvt.astype('int64'); F=F.merge(cw,on='MVT_ID_mvt',how='left')
E=F.cw_E_wait_min.values*60.0
for j in range(8): F[f'ready{j}']=(E>=900*j-300).astype(float)
def two_windows(thr,tag):
    S=np.zeros(len(F)); n=np.zeros(len(F)); found=np.zeros(len(F)); steps=np.zeros(len(F)); active=np.ones(len(F))
    for j in range(1,8):
        rj=F[f'ready{j}'].values; fj=(F[f'I{j}'].values<=thr).astype(float); wj=F[f'w{j}'].values
        S+=active*rj*(1-fj)*wj; n+=active*rj*(1-fj); found+=active*rj*fj; steps+=active*rj; active=active*rj*(1-fj)
    F[f'cn2_S{tag}']=S; F[f'cn2_nblocked{tag}']=n; F[f'cn2_found{tag}']=np.minimum(found,1.0); F[f'cn2_steps{tag}']=steps
    for M_ in (7,10,15): F[f'cn2_taxi_m{M_}{tag}']=F.cw_exot+60.0*M_*(F.w0*F.ready0+S)
for thr,tag in ((0.0,'_strict'),(0.15,'_lax')): two_windows(thr,tag)
F['cn2_w0']=F.w0; F['cn2_p0']=F.p0; F['cn2_I0']=F.I0; F['cn2_ready0']=F.ready0
OUT=['MVT_ID_mvt']+[c for c in F.columns if c.startswith('cn2_')]; F[OUT].to_parquet(f'ctotnorm2w_{MODE}.parquet'); print(F[OUT].shape); print(F[OUT].describe().loc[['mean','25%','50%','75%']].round(2).T.to_string())
