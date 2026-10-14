import json,hashlib,numpy as np,pandas as pd,lightgbm as lgb
A='../ablmac/'; G=json.load(open(A+'groups_year.json')); GR=json.load(open('rank_groups.json'))
a=pd.read_parquet(A+'year_table.parquet'); a['MVT_ID_mvt']=a.MVT_ID_mvt.astype('int64')
for f,drop in (('rwy_windows.parquet',['apt','rwy','rf_dep_gap_prev','rf_dep_gap_next']),('rwy_ref.parquet',[])):
    x=pd.read_parquet(f).drop(columns=drop); x['MVT_ID_mvt']=x.MVT_ID_mvt.astype('int64'); a=a.merge(x.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left')
r=pd.read_parquet('rank_table.parquet'); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64')
for f,drop in (('rwy_windows_2026.parquet',['apt','rwy','rf_dep_gap_prev','rf_dep_gap_next']),('rwy_ref_2026.parquet',[])):
    x=pd.read_parquet(f).drop(columns=drop); x['MVT_ID_mvt']=x.MVT_ID_mvt.astype('int64'); r=r.merge(x.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left')
RW=[c for c in pd.read_parquet('rwy_windows.parquet').columns if c not in ('MVT_ID_mvt','apt','rwy','rf_dep_gap_prev','rf_dep_gap_next')]; RF=[c for c in pd.read_parquet('rwy_ref.parquet').columns if c!='MVT_ID_mvt']
groups=['CLK','Pista_fila','Pista_reorg','CTOT_proxy_trafego','Dijkstra_stand_pista','ADSB_antigo_basico','Clima_origem','Demanda_aeroporto']
feats=[c for g in groups for c in G[g]]+RW+RF; miss=[c for c in feats if c not in r.columns]; print('colunas ausentes em 2026:',miss); assert not miss
print('treino (2025 inteiro):',len(a),'linhas | features',len(feats))
y=a.en.clip(-3000,3000).values
P=dict(n_estimators=250,learning_rate=0.07,num_leaves=63,min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,max_bin=127,verbose=-1,n_jobs=14,random_state=0)
m=lgb.LGBMRegressor(**P).fit(a[feats],y); c=m.predict(r[feats])
v=pd.read_parquet('../../../../../../../../Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/submissions/upstanding-insect_v98.parquet') if False else pd.read_parquet('/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/submissions/upstanding-insect_v98.parquet'); v['MVT_ID_mvt']=v.MVT_ID_mvt.astype('int64')
r['corr']=c; m2=v.merge(r[['MVT_ID_mvt','corr','apt']],on='MVT_ID_mvt',how='left'); assert m2['corr'].notna().all() and len(m2)==len(v)
DOSE=0.25; new=np.maximum(m2.TAXITIME_SEC_mvt.values+DOSE*m2['corr'].values,60.0); out=pd.DataFrame({'MVT_ID_mvt':m2.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':new.astype('float64')})
dl=new-m2.TAXITIME_SEC_mvt.values; print('delta sobre a v98: média %.2f s | rms %.2f s | p1/p99 %.1f/%.1f | |delta|max %.1f'%(dl.mean(),np.sqrt((dl**2).mean()),np.percentile(dl,1),np.percentile(dl,99),np.abs(dl).max()))
print('delta médio por aeroporto:',pd.Series(dl).groupby(m2.apt.values).mean().round(2).to_dict()); print('rms por aeroporto:',pd.Series(dl).groupby(m2.apt.values).apply(lambda s:np.sqrt((s**2).mean())).round(1).to_dict())
assert np.isfinite(out.TAXITIME_SEC_mvt).all() and out.MVT_ID_mvt.is_unique and (out.MVT_ID_mvt.values==v.MVT_ID_mvt.values).all()
p='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/submissions/pendente_envio/candidate_corretor_janelas_dose025_sobre_v98.parquet'; out.to_parquet(p,index=False)
print('salvo',p.split('/')[-1],'| linhas',len(out),'| sha256',hashlib.sha256(open(p,'rb').read()).hexdigest()); print('dtypes',out.dtypes.astype(str).to_dict(),'| v98 dtypes',v.dtypes.astype(str).to_dict())
