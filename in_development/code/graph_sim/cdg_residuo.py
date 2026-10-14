"""LFPG: o residuo do campeao (stack OOF) nas linhas normais depende do comprimento da rota (grafo), stand, tipo, companhia, hora, frio?
Features SEM o BLOCK (que e o alvo): relogio de MVT, SCHED, METAR em MVT."""
import os,sys,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from simula import rotas
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
apt='LFPG'; R='/Users/bruno.quirino@latam.com/Projetos/oof_store/clean_oof/chain/'; C='stack linear por aeroporto [componentes+hand]'; rm=lambda a:np.sqrt(np.mean(np.asarray(a)**2))
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)]
d=pd.read_parquet(R+'stack_oof_preds.parquet',columns=['MVT_ID_mvt','y',C]).merge(tr,on='MVT_ID_mvt'); d['e']=d.y-d[C]
d['mvt']=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); d['sch']=pd.to_datetime(d.SCHED_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
d=d.rename(columns={'STAND_mvt':'stand','RUNWAY_mvt':'rwy','AIRCRAFT_TYPE_mvt':'acft','AIRCRAFT_OPERATOR_flt':'op'}); d['rwy']=d.rwy.astype(str)
me=pd.read_parquet('data/processed/metar_2025.parquet'); me=me[me.ADEP_mvt==apt].copy(); me['t']=pd.to_datetime(me.metar_time_utc,utc=True).dt.as_unit('s').astype('int64'); me['tc']=(me.tmpf-32)/1.8
d=pd.merge_asof(d.sort_values('mvt'),me.sort_values('t')[['t','tc','vsby','sknt']],left_on='mvt',right_on='t',direction='backward',tolerance=5400)
dt=pd.to_datetime(d.mvt,unit='s'); d['hour']=dt.dt.hour; d['dow']=dt.dt.dayofweek; d['dia']=dt.dt.strftime('%Y-%m-%d'); d['atraso']=(d.mvt-d.sch)/60
d['L']=rotas(apt,d); print(f'{apt}: n={len(d)}  L disponivel {d.L.notna().mean():.0%}  L mediana {d.L.median():.0f} m (q10 {d.L.quantile(.1):.0f}, q90 {d.L.quantile(.9):.0f})')
def top(s,k=30): v=s.value_counts().head(k).index; return s.where(s.isin(v),'outros').astype('category')
d['acft']=top(d.acft); d['op']=top(d.op); d['stnd']=top(d.stand,60); d['rwyc']=d.rwy.astype('category')
pior=set((d.e**2).sort_values(ascending=False).index[:20]); n=d[~d.index.isin(pior)].reset_index(drop=True)
print(f'linhas normais (sem as 20 piores, escolha ex-post): n={len(n)} RMSE do campeao {rm(n.e):.1f}, MAE {abs(n.e).mean():.0f}')
print('correlacao do residuo com L: %.3f | de y com L: %.3f | do residuo com atraso: %.3f'%(n[['e','L']].corr().iloc[0,1],n[['y','L']].corr().iloc[0,1],n[['e','atraso']].corr().iloc[0,1]))
gk=GroupKFold(5); mk=lambda cats:HistGradientBoostingRegressor(max_depth=4,learning_rate=.05,max_iter=200,min_samples_leaf=50,categorical_features=cats,random_state=0)
conj=[('L',['L']),('+ rota (pista)',['L','rwyc']),('+ stand',['L','rwyc','stnd']),('+ tipo, companhia',['L','rwyc','stnd','acft','op']),('+ hora, dia da semana, atraso',['L','rwyc','stnd','acft','op','hour','dow','atraso']),('+ METAR (frio, vento, visib.)',['L','rwyc','stnd','acft','op','hour','dow','atraso','tc','vsby','sknt'])]
print('residuo do campeao previsto com GBM (validacao cruzada por dia):  RMSE do residuo  (campeao = %.1f)'%rm(n.e))
for nome,cols in conj:
    X=n[cols]; cats=[c in('rwyc','stnd','acft','op') for c in cols]; p=cross_val_predict(mk(cats),X,n.e,cv=gk,groups=n.dia); print(f'  {nome:34s} {rm(n.e-p):6.1f}  (delta {rm(n.e-p)-rm(n.e):+.2f})')
# y direto sem o campeao, so com essas features (quao longe fica do campeao)
cols=conj[-1][1]; X=n[cols]; cats=[c in('rwyc','stnd','acft','op') for c in cols]; p=cross_val_predict(mk(cats),X,n.y,cv=gk,groups=n.dia)
print(f'y previsto so com essas features (sem o campeao): RMSE {rm(n.y-p):.0f}  | mediana global {rm(n.y-n.y.median()):.0f} | campeao {rm(n.e):.0f}')
