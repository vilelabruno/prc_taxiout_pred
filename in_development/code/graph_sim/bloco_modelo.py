"""Quanto da duracao BLOCK -> inicio do taxi para frente (blk_taxi) se explica?  Alvo: erro na estimativa do BLOCK.
Validacao: deixa um dia fora, purgando +-1 dia do treino."""
import numpy as np, pandas as pd, lightgbm as lgb, os
d=pd.read_csv(os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/pushback_EHAM.csv'))
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt'])
tr['apt']=np.where(tr.PHASE_mvt=='DEP',tr.ADEP_mvt,tr.ADES_mvt); tr=tr[tr.apt=='EHAM']
mv=np.sort(pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64').to_numpy())
d['carga30']=np.searchsorted(mv,d.mvt+1800)-np.searchsorted(mv,d.mvt-1800)
d['hora']=(pd.to_datetime(d.mvt,unit='s').dt.hour+pd.to_datetime(d.mvt,unit='s').dt.minute/60)
d['pier']=d.stand.astype(str).str[0]
d['dow']=pd.to_datetime(d.mvt,unit='s').dt.dayofweek
for c in ['acft','oper','pier','rwy']: d[c]=d[c].astype('category')
d['y']=d.blk_taxi; d['day']=pd.to_datetime(d.day)
d=d[(d.y>-200)&(d.y<900)]
def cv(cols,params=dict(n_estimators=150,learning_rate=.05,num_leaves=8,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1)):
    e=np.zeros(len(d)); 
    for dd in d.day.unique():
        te=d.day==dd; trn=(d.day<dd-pd.Timedelta(days=1))|(d.day>dd+pd.Timedelta(days=1))
        if cols is None: p=d.y[trn].mean()
        else: p=lgb.LGBMRegressor(**params).fit(d.loc[trn,cols],d.y[trn]).predict(d.loc[te,cols])
        e[te.to_numpy()]=d.y[te]-p
    return np.sqrt((e**2).mean()),np.median(np.abs(e))
print('n',len(d),'| desvio de blk_taxi',round(d.y.std(),1))
for nome,c in [('constante',None),('stand (dp,xp,yp,hdg)',['dp','xp','yp','hdg']),('+ pier',['dp','xp','yp','hdg','pier']),
  ('+ hora, dia da semana',['dp','xp','yp','hdg','pier','hora','dow']),('+ carga +-30 min',['dp','xp','yp','hdg','pier','hora','dow','carga30']),
  ('+ tipo, operador, pista',['dp','xp','yp','hdg','pier','hora','dow','carga30','acft','oper','rwy'])]:
    r,m=cv(c); print(f'{nome:28s} RMSE do BLOCK {r:6.1f} s   erro absoluto mediano {m:5.1f} s')
