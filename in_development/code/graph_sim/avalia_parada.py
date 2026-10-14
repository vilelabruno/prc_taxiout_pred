import os,glob,sys,numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/parada')
for apt in sys.argv[1].split(','):
    d=pd.concat([pd.read_parquet(f) for f in glob.glob(f'{O}/{apt}_2025*.parquet')],ignore_index=True)
    h=pd.concat([pd.read_parquet(f) for f in glob.glob(f'{O}/hod_{apt}_2025*.parquet')]); hm=h.groupby('hour').agg(mm=('n_mov','mean'),mg=('n_ground','mean'))
    d=d.join(hm,on='hour'); 
    for suf in 'bs': d[f'ratio_mov_{suf}']=d[f'n_mov_{suf}']/d.mm; d[f'ratio_ground_{suf}']=d[f'n_ground_{suf}']/d.mg
    print(f'\n== {apt} n={len(d)}  par q10/50/90 {d.par.quantile(.1):.0f}/{d.par.median():.0f}/{d.par.quantile(.9):.0f} s, sd {d.par.std():.0f}')
    feats={'a frente (na rota)':['n_ahead_b','n_ahead_s','n_ahead_stop_b','n_ahead_stop_s','n_ahead_1k_b','n_ahead_1k_s'],
           'ao redor (300 m do stand)':['n_near_b','n_near_s','n_near_mov_b','n_near_mov_s'],
           'global':['n_mov_b','n_mov_s','n_ground_b','ratio_mov_b','ratio_mov_s','ratio_ground_b'],
           'controle':['hour','dur1','s_e1']}
    for g,cols in feats.items():
        print(f' {g}:',{c:f'{spearmanr(d[c],d.par)[0]:+.2f}' for c in cols})
    # quartis do melhor sinal
    for c in ('n_ahead_s','n_near_s','ratio_mov_b'):
        try: q=pd.qcut(d[c].rank(method='first'),4,labels=False); print(f' par mediana por quartil de {c}:',d.groupby(q).par.median().round(0).tolist())
        except Exception as e: pass
    # CV por dia
    X0=d[['hour','dur1','s_e1']]; X1=d[['hour','dur1','s_e1','n_ahead_b','n_ahead_stop_b','n_ahead_1k_b','n_near_b','n_near_mov_b','n_mov_b','ratio_mov_b','ratio_ground_b']]
    X2=X1.join(d[['n_ahead_s','n_ahead_stop_s','n_ahead_1k_s','n_near_s','n_near_mov_s','n_mov_s','ratio_mov_s']])
    gk=GroupKFold(n_splits=5); rm=lambda a,b:np.sqrt(np.mean((a-b)**2))
    print(f' baseline: mediana  RMSE {rm(d.par,d.par.median()):.0f} s')
    for nome,X in (('controle',X0),('+ estado no BLOCK',X1),('+ estado no inicio da parada',X2)):
        p=cross_val_predict(GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,random_state=0),X,d.par,cv=gk,groups=d.day)
        print(f'   GBM {nome}: RMSE {rm(d.par,p):.0f}  R2 {1-np.var(d.par-p)/np.var(d.par):.2f}')
    # y total: MVT-BLOCK? efeito do estado no y
    p0=cross_val_predict(GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,random_state=0),X0,d.y,cv=gk,groups=d.day)
    p1=cross_val_predict(GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,random_state=0),X1,d.y,cv=gk,groups=d.day)
    print(f' y=MVT-BLOCK (so nesses voos): mediana {rm(d.y,d.y.median()):.0f} | controle {rm(d.y,p0):.0f} | +estado no BLOCK {rm(d.y,p1):.0f}')
