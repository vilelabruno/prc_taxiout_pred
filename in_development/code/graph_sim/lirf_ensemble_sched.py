"""LIRF jan+jul 2025: o grafo (camada relogio+grafo, ou grafo sozinho) ajuda um ensemble com o campeao nas linhas classificadas como normais?"""
import numpy as np, pandas as pd, warnings, os; warnings.filterwarnings('ignore')
S=os.path.expanduser('~/Projetos/oof_store')
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/LIRF/cenario_b_oof.parquet')
pa=pd.read_parquet(f'{S}/defeitos_prio/pi_A.parquet')
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'])
o=o.merge(tr.rename(columns={'MVT_ID_mvt':'id'}),on='id',how='left')
o['hora']=pd.to_datetime(o.MVT_TIME_UTC_mvt,utc=True).dt.hour
flags=[c for c in pa.columns if c.startswith('flag90_')]
o=o.merge(pa[['MVT_ID_mvt','pi_eco','pi_dia1','pi_sao','pi_inc']+flags].rename(columns={'MVT_ID_mvt':'id'}),on='id',how='left')
o[['pi_eco','pi_dia1','pi_sao','pi_inc']]=o[['pi_eco','pi_dia1','pi_sao','pi_inc']].fillna(0); o[flags]=o[flags].fillna(False).astype(bool)
o['any90']=o[flags].any(axis=1); o['d']=pd.to_datetime(o.day)
print('linhas',len(o),'| sinalizadas (flag90 de qualquer tipo):',int(o.any90.sum()),'| eco:',int(o.flag90_eco.sum()),'| madrugada 00-05 UTC:',int((o.hora<6).sum()),'| y>=21600:',int((o.y>=21600).sum()))
def rm(e): return float(np.sqrt(np.mean(np.square(e))))
def best_w(d,g):
    ws=np.linspace(0,1,21); return ws[int(np.argmin([rm(d.y-(w*d.champ+(1-w)*d[g])) for w in ws]))]
def boot(d,g,w,n=400,seed=0):
    rng=np.random.default_rng(seed); days=d.d.unique(); grp={k:v for k,v in d.groupby('d')}; out=[]
    for _ in range(n):
        s=pd.concat([grp[k] for k in rng.choice(days,len(days))]); out.append(rm(s.y-(w*s.champ+(1-w)*s[g]))-rm(s.y-s.champ))
    return np.percentile(out,[5,95])
def avalia(nome,d):
    print(f'\n== {nome}  n={len(d)}  (jan {int((d.mes=="2025-01").sum())} / jul {int((d.mes=="2025-07").sum())})  RMSE campeao {rm(d.y-d.champ):.1f}')
    for g in ('b_relogio_mais_grafo','b_so_relogio','sim_a'):
        # peso aprendido no OUTRO mes e aplicado no mes de teste
        e=[];ws=[]
        for m in d.mes.unique():
            w=best_w(d[d.mes!=m],g); t=d[d.mes==m]; e.append(t.y-(w*t.champ+(1-w)*t[g])); ws.append(w)
        e=pd.concat(e); wf=best_w(d,g); lo,hi=boot(d,g,wf)
        print(f'  {g:22s} sozinho {rm(d.y-d[g]):7.1f} | ensemble peso do campeao (mes fora: {ws}) RMSE {rm(e):7.1f} (delta {rm(e)-rm(d.y-d.champ):+6.2f}) | peso global {wf:.2f} delta {rm(d.y-(wf*d.champ+(1-wf)*d[g]))-rm(d.y-d.champ):+6.2f} IC90 [{lo:+.2f},{hi:+.2f}]')

o['ms']=(pd.to_datetime(o.MVT_TIME_UTC_mvt,utc=True)-pd.to_datetime(o.SCHED_TIME_UTC_mvt,utc=True)).dt.total_seconds()
print('MVT-SCHED quantis 1/5/25/50/75/95/99:',o.ms.quantile([.01,.05,.25,.5,.75,.95,.99]).round(0).tolist(),'| nulos',int(o.ms.isna().sum()))
for T in (1800,3600):
    f=o[o.ms<T]; print(f'MVT-SCHED < {T//60} min: {len(f)} linhas ({len(f)/len(o):.1%}) | dessas, y>=30 min: {(f.y>=1800).sum()} ({(f.y>=1800).mean():.1%}) | y>=6 h: {(f.y>=21600).sum()} | fora do filtro: y>=30 min {(o[o.ms>=T].y>=1800).mean():.1%}, n={int((o.ms>=T).sum())}')
avalia('MVT-SCHED < 30 min (sem usar o rotulo)', o[o.ms<1800])
avalia('MVT-SCHED < 30 min + sem sinalizadas + sem madrugada', o[(o.ms<1800)&(~o.any90)&(o.hora>=6)])
avalia('MVT-SCHED < 60 min', o[o.ms<3600])
x=o[(o.ms<1800)&(~o.any90)&(o.hora>=6)]
print('\nsubconjunto final: RMSE por modelo; |erro|>1800 s conta',int(((x.y-x.b_relogio_mais_grafo).abs()>1800).sum()),'linhas e',round(float((((x.y-x.b_relogio_mais_grafo).abs()>1800)*(x.y-x.b_relogio_mais_grafo)**2).sum()/((x.y-x.b_relogio_mais_grafo)**2).sum()),3),'do erro quadratico do grafo')
