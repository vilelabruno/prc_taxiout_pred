"""LIRF jan+jul 2025: o grafo (camada relogio+grafo, ou grafo sozinho) ajuda um ensemble com o campeao nas linhas classificadas como normais?"""
import sys, numpy as np, pandas as pd, warnings, os; warnings.filterwarnings('ignore')
S=os.path.expanduser('~/Projetos/oof_store'); APT=sys.argv[1]
o=pd.read_parquet(S+'/grafo_por_aeroporto/'+APT+'/cenario_b_oof.parquet')
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
o['ms']=(pd.to_datetime(o.MVT_TIME_UTC_mvt,utc=True)-pd.to_datetime(o.SCHED_TIME_UTC_mvt,utc=True)).dt.total_seconds()
o['pmax']=o[['pi_eco','pi_dia1','pi_sao','pi_inc']].max(axis=1)
print('pmax quantis 50/90/95/99/99.5:',o.pmax.quantile([.5,.9,.95,.99,.995]).round(3).tolist(),'| >0.5:',int((o.pmax>.5).sum()),'| >0.1:',int((o.pmax>.1).sum()))
TAUS=[0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.01]; WS=np.linspace(0,1,21); G='b_relogio_mais_grafo'
def sistema(d,keep,w):
    p=np.where(keep,w*d.champ+(1-w)*d[G],d.champ); return d.y-p
def procura(d,cut):
    best=None
    for t in TAUS:
        keep=((d.pmax<t)&cut).to_numpy()
        for w in WS:
            r=rm(sistema(d,keep,w))
            if best is None or r<best[0]: best=(r,t,w)
    return best
def roda(nome,cutcol,lim):
    cut=(o[cutcol]<lim)
    print(f'\n=== {nome}: retem linhas com {cutcol} < {lim} s e pmax < tau ===  (campeao em todas as linhas: {rm(o.y-o.champ):.1f})')
    tot=[];
    for m in o.mes.unique():
        tr=o[o.mes!=m]; te=o[o.mes==m]; r,t,w=procura(tr,(tr[cutcol]<lim))
        keep=((te.pmax<t)&(te[cutcol]<lim)).to_numpy(); e=sistema(te,keep,w); tot.append(e)
        print(f'  teste {m}: tau={t} w_campeao={w:.2f} (escolhidos no outro mes) | RMSE sistema {rm(e):.1f} contra campeao {rm(te.y-te.champ):.1f} | retidas {keep.mean():.1%}')
    e=pd.concat(tot); print(f'  mes fora, junto: sistema {rm(e):.1f} contra campeao {rm(o.y-o.champ):.1f}  (delta {rm(e)-rm(o.y-o.champ):+.2f})')
    r,t,w=procura(o,cut); keep=((o.pmax<t)&cut).to_numpy(); e=sistema(o,keep,w)
    rng=np.random.default_rng(0); days=o.d.unique(); idx={k:np.where(o.d.to_numpy()==k)[0] for k in days}; out=[]
    for _ in range(300):
        ii=np.concatenate([idx[k] for k in rng.choice(days,len(days))]); out.append(rm(e.to_numpy()[ii])-rm((o.y-o.champ).to_numpy()[ii]))
    k=o[keep]; print(f'  global (mesmos dados): tau={t} w={w:.2f} RMSE {r:.1f} delta {r-rm(o.y-o.champ):+.2f} IC90 [{np.percentile(out,5):+.2f},{np.percentile(out,95):+.2f}] | retidas {keep.mean():.1%} (n={len(k)}) | dentro das retidas: campeao {rm(k.y-k.champ):.1f}, grafo {rm(k.y-k[G]):.1f}, mistura {rm(k.y-(w*k.champ+(1-w)*k[G])):.1f}')
    print('  curva (tau -> melhor RMSE do sistema com w otimo, global):',' '.join(f'{t_}:{min(rm(sistema(o,((o.pmax<t_)&cut).to_numpy(),w_)) for w_ in WS):.1f}' for t_ in TAUS))

print('APT',APT,'| linhas',len(o),'| pmax>0.5:',int((o.pmax>.5).sum()),'| MVT-SCHED quantis 5/25/50/75/95:',o.ms.quantile([.05,.25,.5,.75,.95]).round(0).tolist())
for g in ('b_relogio_mais_grafo','sim_a'):
    G=g
    print(f'\n--- grafo = {g}: sozinho {rm(o.y-o[g]):.1f}, campeao {rm(o.y-o.champ):.1f}, corr(grafo,y) {np.corrcoef(o[g],o.y)[0,1]:.3f}, corr(champ,y) {np.corrcoef(o.champ,o.y)[0,1]:.3f}')
    globals()['G']=g
    roda('sem corte (todas as linhas)','ms',1e9)
    roda('MVT-SCHED < 20 min','ms',1200)
import lightgbm as lgb
print('\n--- empilhamento: LightGBM sobre o residuo do campeao, mes fora ---')
o['hora']=pd.to_datetime(o.MVT_TIME_UTC_mvt,utc=True).dt.hour
for nome,cols in [('so campeao + hora',['champ','hora']),('+ grafo (3 camadas)',['champ','hora','sim_a','b_so_relogio','b_relogio_mais_grafo']),('+ grafo + MVT-SCHED',['champ','hora','sim_a','b_so_relogio','b_relogio_mais_grafo','ms'])]:
    e=pd.Series(0.,index=o.index)
    for m in o.mes.unique():
        tr=o[o.mes!=m]; te=o[o.mes==m]
        mdl=lgb.LGBMRegressor(n_estimators=200,learning_rate=.03,num_leaves=15,min_child_samples=100,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1).fit(tr[cols],tr.y-tr.champ)
        e[te.index]=te.y-(te.champ+0.5*mdl.predict(te[cols]))
    print(f'{nome:26s} RMSE {rm(e):.1f}  (campeao {rm(o.y-o.champ):.1f}, delta {rm(e)-rm(o.y-o.champ):+.2f})')
