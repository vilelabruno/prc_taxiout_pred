"""A duracao BLOCK->taxi proprio (ini2) e a parada (par) dependem de tipo de aeronave, companhia, stand, hora, dia da semana, atraso e frio?"""
import os,glob,sys,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/parada'); rm=lambda a,b:np.sqrt(np.mean((a-b)**2))
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt','SCHED_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt'])
tr=tr[tr.PHASE_mvt=='DEP'].copy()
for c,n in (('MVT_TIME_UTC_mvt','mvt'),('BLOCK_TIME_UTC_mvt','blk'),('SCHED_TIME_UTC_mvt','sch')): tr[n]=pd.to_datetime(tr[c],utc=True).dt.as_unit('s').astype('int64')
me=pd.read_parquet('data/processed/metar_2025.parquet'); me['t']=pd.to_datetime(me.metar_time_utc,utc=True).dt.as_unit('s').astype('int64'); me['tc']=(me.tmpf-32)/1.8; me['spread']=(me.tmpf-me.dwpf)/1.8
def top(s,k=25): v=s.value_counts().head(k).index; return s.where(s.isin(v),'outros').astype('category')
for apt in sys.argv[1].split(','):
    d=pd.concat([pd.read_parquet(f) for f in glob.glob(f'{O}/{apt}_2025*.parquet')],ignore_index=True)
    t=tr[tr.ADEP_mvt==apt].drop_duplicates(['blk','mvt','STAND_mvt']); d=d.merge(t,left_on=['blk','mvt','stand'],right_on=['blk','mvt','STAND_mvt'],how='left')
    m=me[me.ADEP_mvt==apt].sort_values('t'); d=pd.merge_asof(d.sort_values('blk'),m[['t','tc','spread','vsby']],left_on='blk',right_on='t',direction='backward',tolerance=5400)
    d['dow']=pd.to_datetime(d.blk,unit='s').dt.dayofweek; d['atraso']=(d.blk-d.sch)/60; d['acft']=top(d.AIRCRAFT_TYPE_mvt); d['op']=top(d.AIRCRAFT_OPERATOR_flt); d['stnd']=top(d.stand,40)
    d['frio']=(d.tc<=3).astype(float); d['month']=pd.to_datetime(d.blk,unit='s').dt.month
    print(f'\n== {apt} n={len(d)} (com tipo {d.acft.notna().mean():.0%}, metar {d.tc.notna().mean():.0%}); frio(<=3C): {d.frio.mean():.0%}')
    print(' ini2 (BLOCK->taxi proprio) mediana por frio:',d.groupby('frio').ini2.median().round(0).to_dict(),'| por dia da semana:',d.groupby('dow').ini2.median().round(0).tolist())
    print(' ini2 por tipo (n>=25):',{k:round(v.ini2.median()) for k,v in d.groupby('AIRCRAFT_TYPE_mvt') if len(v)>=25})
    gk=GroupKFold(5); cat=lambda X:[c in('acft','op','stnd') for c in X.columns]
    steps=[('hora',['hour']),('+ dia da semana',['dow']),('+ tipo de aeronave',['acft']),('+ companhia',['op']),('+ stand',['stnd']),('+ atraso vs programado',['atraso']),('+ temperatura/umidade/visib.',['tc','spread','vsby','frio'])]
    for alvo in ('ini2','par'):
        cols=[]; print(f' alvo {alvo}: mediana RMSE {rm(d[alvo],d[alvo].median()):.0f} (sd {d[alvo].std():.0f})')
        for nome,add in steps:
            cols+=add; X=d[cols]
            p=cross_val_predict(HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=150,categorical_features=cat(X),random_state=0),X,d[alvo],cv=gk,groups=d.day)
            print(f'   {nome:32s} RMSE {rm(d[alvo],p):.0f}')
