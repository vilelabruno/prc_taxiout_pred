import json,hashlib,numpy as np,pandas as pd,lightgbm as lgb
A='../ablmac/'; G=json.load(open(A+'groups_year.json')); GR=json.load(open('rank_groups.json'))
a=pd.read_parquet(A+'year_table.parquet'); a['MVT_ID_mvt']=a.MVT_ID_mvt.astype('int64')
def mrg(df,f,drop=()):
    x=pd.read_parquet(f).drop(columns=list(drop)); x['MVT_ID_mvt']=x.MVT_ID_mvt.astype('int64'); return df.merge(x.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left')
a=mrg(a,'rwy_windows.parquet',['apt','rwy','rf_dep_gap_prev','rf_dep_gap_next']); a=mrg(a,'rwy_ref.parquet'); a=mrg(a,'ctotwin_2025.parquet')
r=pd.read_parquet('rank_table.parquet'); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64')
r=mrg(r,'rwy_windows_2026.parquet',['apt','rwy','rf_dep_gap_prev','rf_dep_gap_next']); r=mrg(r,'rwy_ref_2026.parquet'); r=mrg(r,'ctotwin_2026.parquet')
# pesos por aeroporto da espera no stand (alvo hold=y-ma), refeitos com os 12 meses de 2025
FEA=['cw_E_wait_eobt_min','cw_folga_eobt_lo','cw_n_blocked','cw_n_lost','cw_n_avail','cw_frac_avail','ma_min']
for t in (a,r): t['ma_min']=t.ma/60; 
def prep(t): t[FEA]=t[FEA].fillna(0.0); return t
hold=a.y-a.ma; ok=a.ma.notna()&(hold>-300)&(hold<3600); a['hold']=hold.clip(0,3600)
def X(t): return np.c_[np.ones(len(t)),t[FEA].fillna(0.0).values]
bp,*_=np.linalg.lstsq(X(a[ok]),a.loc[ok,'hold'].values,rcond=None); B={}
for ap,g in a[ok].groupby('apt'): B[ap],*_=np.linalg.lstsq(X(g),g.hold.values,rcond=None)
FE_=FEA
def ca2(t):
    out=np.zeros(len(t)); outp=np.zeros(len(t)); bl=np.zeros(len(t)); bb=np.zeros(len(t)); be=np.zeros(len(t)); Xt=X(t)
    for ap,idx in t.groupby('apt').indices.items():
        b=B[ap]; out[idx]=Xt[idx]@b; outp[idx]=Xt[idx]@bp; bl[idx]=b[FE_.index('cw_n_lost')+1]; bb[idx]=b[FE_.index('cw_n_blocked')+1]; be[idx]=b[1]
    t['ca2_hold_fit']=out; t['ca2_hold_fit_pooled']=outp; t['ca2_b_lost_s']=bl; t['ca2_b_blocked_s']=bb; t['ca2_b_Ee_s_per_min']=be; t['ca2_diff_apt_pooled']=out-outp; return t
a=ca2(a); r=ca2(r)
print('betas por aeroporto (espera por min de E_wait EOBT / janela perdida / min bloqueado, s):',{k:(round(v[1]),round(v[FE_.index('cw_n_lost')+1],1),round(v[FE_.index('cw_n_blocked')+1],1)) for k,v in B.items()})
RW=[c for c in pd.read_parquet('rwy_windows.parquet').columns if c not in ('MVT_ID_mvt','apt','rwy','rf_dep_gap_prev','rf_dep_gap_next')]; RF=[c for c in pd.read_parquet('rwy_ref.parquet').columns if c!='MVT_ID_mvt']
CT=[c for c in pd.read_parquet('ctotwin_2025.parquet').columns if c!='MVT_ID_mvt']; CA=['ca2_hold_fit','ca2_hold_fit_pooled','ca2_b_lost_s','ca2_b_blocked_s','ca2_b_Ee_s_per_min','ca2_diff_apt_pooled']
groups=['CLK','Pista_fila','Pista_reorg','CTOT_proxy_trafego','Dijkstra_stand_pista','ADSB_antigo_basico','Clima_origem','Demanda_aeroporto']
feats=[c for g in groups for c in G[g]]+RW+RF+CT+CA; miss=[c for c in feats if c not in r.columns]; print('colunas ausentes em 2026:',miss); assert not miss
print('treino (2025 inteiro):',len(a),'linhas | features',len(feats),'| NaN em ca2 (2026):',int(r[CA].isna().sum().sum()))
y=a.en.clip(-3000,3000).values
P=dict(n_estimators=250,learning_rate=0.07,num_leaves=63,min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,max_bin=127,verbose=-1,n_jobs=14,random_state=0)
m=lgb.LGBMRegressor(**P).fit(a[feats],y); c=m.predict(r[feats])
v=pd.read_parquet('/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/submissions/upstanding-insect_v98.parquet'); v['MVT_ID_mvt']=v.MVT_ID_mvt.astype('int64')
r['corr']=c; m2=v.merge(r[['MVT_ID_mvt','corr','apt']],on='MVT_ID_mvt',how='left'); assert m2['corr'].notna().all() and len(m2)==len(v)
DOSE=0.25; new=np.maximum(m2.TAXITIME_SEC_mvt.values+DOSE*m2['corr'].values,60.0); out=pd.DataFrame({'MVT_ID_mvt':m2.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':new.astype('float64')})
dl=new-m2.TAXITIME_SEC_mvt.values; print('delta sobre a v98: média %.2f s | rms %.2f s | p1/p99 %.1f/%.1f | |delta|max %.1f'%(dl.mean(),np.sqrt((dl**2).mean()),np.percentile(dl,1),np.percentile(dl,99),np.abs(dl).max()))
v99=pd.read_parquet('/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/submissions/upstanding-insect_v99.parquet'); v99['MVT_ID_mvt']=v99.MVT_ID_mvt.astype('int64'); d99=out.merge(v99,on='MVT_ID_mvt',suffixes=('','_99')); dd=d99.TAXITIME_SEC_mvt-d99.TAXITIME_SEC_mvt_99
print('diferença para a v99 (enviada): média %.2f | rms %.2f s | corr(delta_v100, delta_v99)=%.3f'%(dd.mean(),np.sqrt((dd**2).mean()),np.corrcoef(d99.TAXITIME_SEC_mvt-m2.TAXITIME_SEC_mvt.values,d99.TAXITIME_SEC_mvt_99-m2.TAXITIME_SEC_mvt.values)[0,1]))
print('delta médio por aeroporto:',pd.Series(dl).groupby(m2.apt.values).mean().round(2).to_dict())
ok_=np.isfinite(out.TAXITIME_SEC_mvt).all() and out.MVT_ID_mvt.is_unique and (out.MVT_ID_mvt.values==v.MVT_ID_mvt.values).all() and len(out)==344841
print('VALIDAÇÃO:',ok_); assert ok_
p='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/submissions/pendente_envio/candidate_corretor_ctotjanela_apt_dose025_sobre_v98.parquet'; out.to_parquet(p,index=False)
print('salvo',p.split('/')[-1],'| linhas',len(out),'| sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
