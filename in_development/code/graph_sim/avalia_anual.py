"""Tabela condicional de ini2 (BLOCK->taxi proprio) por encolhimento hierarquico + taxi por L/v; avalia y=MVT-BLOCK em todos os voos."""
import os,sys,glob,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from simula import rotas, med
apt='EHAM'; K=20; GAP=2
S=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); rm=lambda a,b:np.sqrt(np.nanmean((np.asarray(a)-np.asarray(b))**2)); mae=lambda a,b:np.nanmean(abs(np.asarray(a)-np.asarray(b)))
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
for c,n in (('MVT_TIME_UTC_mvt','mvt'),('BLOCK_TIME_UTC_mvt','blk')): tr[n]=pd.to_datetime(tr[c],utc=True).dt.as_unit('s').astype('int64')
tr=tr.drop_duplicates(['blk','mvt','STAND_mvt']).rename(columns={'AIRCRAFT_TYPE_mvt':'acft','AIRCRAFT_OPERATOR_flt':'op'})[['blk','mvt','STAND_mvt','acft','op']]
me=pd.read_parquet('data/processed/metar_2025.parquet'); me=me[me.ADEP_mvt==apt]; me['t']=pd.to_datetime(me.metar_time_utc,utc=True).dt.as_unit('s').astype('int64'); me=me.sort_values('t'); me['tc']=(me.tmpf-32)/1.8
def prep(d):
    d=d.merge(tr,left_on=['blk','mvt','stand'],right_on=['blk','mvt','STAND_mvt'],how='left').sort_values('blk')
    d=pd.merge_asof(d,me[['t','tc']],left_on='blk',right_on='t',direction='backward',tolerance=5400)
    dt=pd.to_datetime(d.blk,unit='s'); d['hb']=(dt.dt.hour//4).astype(str); d['dow']=dt.dt.dayofweek.astype(str); d['cold']=(d.tc<=3).astype(str)
    d['didx']=(pd.to_datetime(d.day)-pd.Timestamp('2025-01-01')).dt.days; return d.reset_index(drop=True)
FACT=['acft','op','stand','hb','dow','cold']
def fit(train,y,factors):
    m0=train[y].median(); res=train[y]-m0; eff={}
    for _ in range(2):
        for f in factors:
            r=res+train[f].map(eff.get(f,{})).fillna(0); g=r.groupby(train[f]).agg(['median','size'])
            e=(g['median']*g['size']/(g['size']+K)).to_dict(); eff[f]=e; res=r-train[f].map(e).fillna(0)
    return m0,eff
def pred(m0,eff,df): 
    p=np.full(len(df),m0,dtype=float)
    for f,e in eff.items(): p=p+df[f].map(e).fillna(0).to_numpy()
    return p
A=prep(pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{S}/anual/{apt}_*.parquet'))],ignore_index=True)); A['rwy']=A.rwy.astype(str)
print(f'{apt}: {len(A)} voos de referencia (62 dias); ini2 mediana {A.ini2.median():.0f} sd {A.ini2.std():.0f}')
# 1) ini2: validacao fora dos dias vizinhos
def cv(df,y,factors):
    out=np.full(len(df),np.nan)
    for di in df.didx.unique():
        te=df.didx==di; trn=df[(df.didx-di).abs()>GAP]; m0,eff=fit(trn,y,factors); out[te.to_numpy()]=pred(m0,eff,df[te])
    return out
print('\nini2 (BLOCK->taxi proprio) previsto sem os dias vizinhos (+-2 d):'); print(f'  {"fatores":34s} RMSE   MAE')
base=cv(A,'ini2',[]); print(f'  {"mediana":34s} {rm(A.ini2,base):4.0f}  {mae(A.ini2,base):4.0f}')
for k in range(1,len(FACT)+1):
    p=cv(A,'ini2',FACT[:k]); print(f'  {"+ ".join(FACT[:k])[:34]:34s} {rm(A.ini2,p):4.0f}  {mae(A.ini2,p):4.0f}')
# 2) y total
sim=pd.read_parquet(f'{S}/sim/{apt}.parquet'); sim['rwy']=sim.rwy.astype(str); sim=prep(sim[['day','blk','mvt','stand','rwy','y','L','adsb','y_base']].copy())
sim=sim[pd.to_datetime(sim.day).dt.day>=8].reset_index(drop=True)
A['L']=rotas(apt,A); R=A[A.L.notna()&(A.taxi2>30)].sort_values('mvt'); R=R.assign(v=R.L/R.taxi2)
sim['ini2_med']=np.nan; sim['ini2_c']=np.nan
for di in sim.didx.unique():
    te=(sim.didx==di).to_numpy(); trn=A[(A.didx-di).abs()>GAP]
    m0,eff=fit(trn,'ini2',[]); sim.loc[te,'ini2_med']=m0
    m0,eff=fit(trn,'ini2',FACT); sim.loc[te,'ini2_c']=pred(m0,eff,sim[te])
sim['v_hat']=np.nan; sim['taxi2_med']=np.nan
for rw,gi in sim.groupby('rwy').groups.items():
    r=R[R.rwy==rw]
    if len(r)<20: continue
    b=sim.loc[gi,'blk'].to_numpy(); t=r.mvt.to_numpy(); vv=r.v.to_numpy()
    vh,ch=med(b,t,vv,3600); vm,cm=med(b,t,vv,30*86400); vm=np.where(cm>=10,vm,np.nan); w=ch/(ch+5); sim.loc[gi,'v_hat']=np.where(ch>0,w*vh+(1-w)*vm,vm)
    sim.loc[gi,'taxi2_med']=med(b,t,r.taxi2.to_numpy(),30*86400)[0]
sim['L_ok']=sim.L.notna()
sim['y_a']=sim.ini2_c+sim.L/sim.v_hat; sim['y_b']=sim.ini2_med+sim.L/sim.v_hat; sim['y_c']=sim.ini2_c+sim.taxi2_med
e=sim.dropna(subset=['y_a','y_b','y_c','y_base']); print(f'\ny = MVT-BLOCK, todos os voos do EHAM com dados (dias >= 8; n={len(e)}); RMSE / MAE em s:')
for nome,c in (('mediana da pista, 30 d (base)','y_base'),('ini2 global + L/v','y_b'),('ini2 condicional + L/v','y_a'),('ini2 condicional + taxi mediano da pista','y_c')): print(f'  {nome:44s} {rm(e.y,e[c]):4.0f} / {mae(e.y,e[c]):4.0f}')
for nome,m in (('sem trilha',~e.adsb),('com trilha',e.adsb)):
    h=e[m]; print(f'  [{nome}, n={len(h)}] base {rm(h.y,h.y_base):.0f} | ini2 global+L/v {rm(h.y,h.y_b):.0f} | condicional+L/v {rm(h.y,h.y_a):.0f} | condicional+taxi mediano {rm(h.y,h.y_c):.0f}')
rng=np.random.default_rng(0); days=e.day.unique(); dif=[]
for _ in range(500):
    s=e[e.day.isin(rng.choice(days,len(days)))]; dif.append(rm(s.y,s.y_a)-rm(s.y,s.y_base))
print(f'  condicional+L/v menos base (RMSE): {rm(e.y,e.y_a)-rm(e.y,e.y_base):+.0f} s, IC95% por dia {np.percentile(dif,2.5):+.0f} a {np.percentile(dif,97.5):+.0f}')
sim.to_parquet(f'{S}/sim/{apt}_anual.parquet')
