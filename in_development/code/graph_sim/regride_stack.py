import sys, glob, os, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store')
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{S}/sincronia_grafo/regride/{APT}_*.parquet'))]).drop(columns=['mvt','day','blk_rotulo','stand'])
d=pd.read_parquet(f'{S}/grafo_por_aeroporto/{APT}/cenario_b_oof.parquet').merge(f,on='id',how='left'); d['d']=pd.to_datetime(d.day); d['tem']=d.tem.fillna(0)
d['dow']=d.d.dt.dayofweek; d['reached']=(d.D_star<=150).astype(float).where(d.tem==1)
v=d.v_bar.fillna(d.v_bar.median()).clip(1.5,12); d['rota_s']=d.D_star/v; d['y_fis_bruto']=d.mvt_star+d.rota_s       # y fisico sem a constante de patio/espera
feats=['tem','mvt_star','D_star','rota_s','y_fis_bruto','v_bar','reached','esp_visto','n_usados','parou','gs_star','d_stand_eucl','mvt_prim','lig']
rm=lambda e: float(np.sqrt(np.mean(np.square(e)))); base=d.y-d.champ
days=np.array(sorted(d.d.unique())); d['blk']=d.d.map({x:i//5 for i,x in enumerate(days)})
P=dict(n_estimators=250,learning_rate=.03,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1)
def cv(cols,sh):
    pred=np.zeros(len(d))
    for b in d.blk.unique():
        te=(d.blk==b).to_numpy(); lo=d.d[te].min()-pd.Timedelta(days=1); hi=d.d[te].max()+pd.Timedelta(days=1); tr=((d.d<lo)|(d.d>hi)).to_numpy()
        m=lgb.LGBMRegressor(**P).fit(d.loc[tr,cols],(d.y-d.champ)[tr]); pred[te]=d.champ[te]+sh*m.predict(d.loc[te,cols])
    return pred
print(APT,'RMSE campeao: todas',round(rm(base),1),'| com cadeia',round(rm(base[d.tem==1]),1))
rng=np.random.default_rng(0); idx={k:np.where(d.d.to_numpy()==k)[0] for k in days}
for nome,cols,sh in [('controle (campeao + dia da semana)',['champ','dow'],1.0),('+ regressao desde o MVT',['champ','dow']+feats,1.0),('+ regressao desde o MVT, encolhido 0.5',['champ','dow']+feats,0.5)]:
    e=d.y-cv(cols,sh); out=[]
    for _ in range(300):
        ii=np.concatenate([idx[k] for k in rng.choice(days,len(days))]); out.append(rm(e.to_numpy()[ii])-rm(base.to_numpy()[ii]))
    t=d.tem==1; print(f'{nome:42s} RMSE todas {rm(e):6.1f} (delta {rm(e)-rm(base):+6.2f} IC90 [{np.percentile(out,5):+.2f},{np.percentile(out,95):+.2f}]) | com cadeia {rm(e[t]):6.1f} contra {rm(base[t]):6.1f} | jan {rm(e[d.mes=="2025-01"])-rm(base[d.mes=="2025-01"]):+.2f} jul {rm(e[d.mes=="2025-07"])-rm(base[d.mes=="2025-07"]):+.2f}')
