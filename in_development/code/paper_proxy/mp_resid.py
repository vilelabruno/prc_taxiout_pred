import numpy as np, pandas as pd, lightgbm as lgb
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/'; S='/private/tmp/claude-502/-Users-bruno-quirino-latam-com-Desktop-bruno-Projetos-prc-data-challenge/5e41970f-79c3-434a-8d98-364545fe860d/scratchpad/'
f=pd.read_parquet(R+'work/daily_layer/frame_result_best.parquet',columns=['MVT_ID_mvt','apt','mes','y','en','pn']); f['MVT_ID_mvt']=f.MVT_ID_mvt.astype('int64')
raw=pd.concat([pd.read_parquet(R+'data/raw/training_2025-0%d-01_2025-0%d-01.parquet'%(a,b),columns=['MVT_ID_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','SCHED_TIME_UTC_mvt']) for a,b in [(1,2),(7,8)]]).reset_index(drop=True); raw=raw[raw.PHASE_mvt=='DEP']; raw['MVT_ID_mvt']=raw.MVT_ID_mvt.astype('int64')
mt=pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True)
for c,n in [('AOBT_3_flt','ma'),('EOBT_1_flt','me'),('SCHED_TIME_UTC_mvt','ms')]: raw[n]=(mt-pd.to_datetime(raw[c],utc=True)).dt.total_seconds()
raw['hr']=mt.dt.hour; raw['dow']=mt.dt.dayofweek; raw['day']=mt.dt.strftime('%Y-%m-%d')
mp=pd.read_parquet(S+'mp_feats.parquet'); mp['MVT_ID_mvt']=mp.MVT_ID_mvt.astype('int64')
d=f.merge(raw[['MVT_ID_mvt','ma','me','ms','hr','dow','day']],on='MVT_ID_mvt').merge(mp.drop(columns=['L15']),on='MVT_ID_mvt',how='left').reset_index(drop=True)
d['aptc']=d.apt.astype('category'); d['ma_m_pn']=d.ma-d.pn; N=len(f); T=(f.en**2).sum(); R0=np.sqrt(T/N); e0=d.en.values
CLK=['pn','ma','ms','me','hr','dow','ma_m_pn','aptc']; NEW=['dep_rel','arr_rel','oc_avg_rel_max','oc_max_rel_max','oc_raw_max']
days=d.day.unique(); fold=d.day.map(dict(zip(days,np.random.default_rng(0).integers(0,5,len(days))))).values; y=d.en.clip(-3000,3000).values
def cv(cols):
    c=np.zeros(len(d))
    for k in range(5):
        tr=fold!=k; c[~tr]=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1).fit(d.loc[tr,cols],y[tr]).predict(d.loc[~tr,cols])
    return c
res={'relógios':cv(CLK),'+ desenho Mas-Pujol (rel. à história)':cv(CLK+NEW)}
print('RMSE base %.3f'%R0)
for nm,c in res.items(): print(f'  {nm:40s}',' | '.join(f'dose {s}: {np.sqrt(((e0-s*c)**2).mean())-R0:+.3f}' for s in (0.5,1.0)))
a=e0-0.5*res['relógios']; b=e0-0.5*res['+ desenho Mas-Pujol (rel. à história)']; g=pd.DataFrame({'day':d.day,'a':a**2,'b':b**2}).groupby('day').sum(); rng=np.random.default_rng(1)
bs=[np.sqrt(x.b.sum()/N)-np.sqrt(x.a.sum()/N) for x in (g.sample(len(g),replace=True,random_state=rng.integers(1e9)) for _ in range(2000))]
print('  incremento: %+.3f IC90 [%+.3f; %+.3f] dias %d/%d'%(np.sqrt(g.b.sum()/N)-np.sqrt(g.a.sum()/N),np.percentile(bs,5),np.percentile(bs,95),(g.b<g.a).sum(),len(g)))
print('  por aeroporto (incremento):',{ap:round(float(np.sqrt((T-(a[d.apt==ap]**2).sum()+(b[d.apt==ap]**2).sum())/N)-np.sqrt((T-(a[d.apt==ap]**2).sum()+(a[d.apt==ap]**2).sum())/N)),3) for ap in sorted(d.apt.unique())})
sl=(d.ma.notna())&((d.y-d.ma)>600); print('  nas linhas de espera (AOBT−BLOCK>10 min) RMSE: base %.1f | relógios %.1f | +MP %.1f'%(np.sqrt((e0[sl]**2).mean()),np.sqrt(((e0-0.5*res["relógios"])[sl]**2).mean()),np.sqrt(((e0-0.5*res["+ desenho Mas-Pujol (rel. à história)"])[sl]**2).mean())))
