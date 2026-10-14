"""LIRF jan+jul 2025: o grafo (camada relogio+grafo, ou grafo sozinho) ajuda um ensemble com o campeao nas linhas classificadas como normais?"""
import numpy as np, pandas as pd, warnings, os; warnings.filterwarnings('ignore')
S=os.path.expanduser('~/Projetos/oof_store')
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/LIRF/cenario_b_oof.parquet')
pa=pd.read_parquet(f'{S}/defeitos_prio/pi_A.parquet')
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt'])
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

avalia('y < 30 min (MVT-BLOCK), todas as linhas', o[o.y<1800])
avalia('y < 30 min + sem sinalizadas + sem madrugada', o[(o.y<1800)&(~o.any90)&(o.hora>=6)])
avalia('y < 30 min + sem sinalizadas (com madrugada)', o[(o.y<1800)&(~o.any90)])
x=o[(o.y<1800)&(~o.any90)&(o.hora>=6)].copy()
print('\nerro por faixa do y real (subconjunto y<30 min, sem sinalizadas, sem madrugada)')
for g in ('b_relogio_mais_grafo','champ'):
    e=x.y-x[g]; print(g,'viés',round(e.mean(),1),'RMSE',round(float(np.sqrt((e**2).mean())),1),'corr(previsto,y)',round(float(np.corrcoef(x[g],x.y)[0,1]),3))
x['faixa']=pd.cut(x.y,[0,600,900,1200,1500,1800])
print(x.groupby('faixa').apply(lambda q:pd.Series({'n':len(q),'grafo_rmse':np.sqrt(((q.y-q.b_relogio_mais_grafo)**2).mean()),'grafo_vies':(q.y-q.b_relogio_mais_grafo).mean(),'champ_rmse':np.sqrt(((q.y-q.champ)**2).mean()),'champ_vies':(q.y-q.champ).mean()})).round(1))
