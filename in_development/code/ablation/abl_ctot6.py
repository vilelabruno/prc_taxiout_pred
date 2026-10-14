"""Ablação com TREINO NO ANO INTEIRO: treina nos 10 meses de treino (resíduo = y − proxy do campeão) e avalia no OOF padrão do projeto (jan+jul 2025) sobre o resíduo do campeão atual (frame_result_best, proxy da v92).
 Configurações: CLK; TODOS; CLK+cada grupo (add-one); TODOS menos cada grupo (leave-one-out); e, para CLK e TODOS, um modelo por aeroporto.
 Métricas no holdout (342.930 linhas): ΔRMSE total em doses 0,25/0,5/0,75/1,0 e RMSE nas esperas (AOBT−BLOCK > 10 min; diagnóstico), por aeroporto; IC por bootstrap de dias nas comparações-chave."""
import json, os, re, time, numpy as np, pandas as pd, lightgbm as lgb
D='/private/tmp/claude-502/-Users-bruno-quirino-latam-com-Desktop-bruno-Projetos-prc-data-challenge/5e41970f-79c3-434a-8d98-364545fe860d/scratchpad/ablmac/'; a=pd.read_parquet(D+'year_table.parquet'); G=json.load(open(D+'groups_year.json'))
W=pd.read_parquet('../prox/winfeat_all.parquet'); W['MVT_ID_mvt']=W.MVT_ID_mvt.astype('int64'); GW=json.load(open('../prox/groups_win.json'))
RW=pd.read_parquet('../prox/rwy_windows.parquet').drop(columns=['apt','rwy','rf_dep_gap_prev','rf_dep_gap_next']); RW['MVT_ID_mvt']=RW.MVT_ID_mvt.astype('int64'); GW['Janela_pista']=[c for c in RW.columns if c!='MVT_ID_mvt']
a=a.merge(RW.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left')
RF=pd.read_parquet('../prox/rwy_ref.parquet'); RF['MVT_ID_mvt']=RF.MVT_ID_mvt.astype('int64'); GW['Janela_ref']=[c for c in RF.columns if c!='MVT_ID_mvt']; a=a.merge(RF.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); CW=pd.read_parquet('../prox/ctotwin_2025.parquet'); CW['MVT_ID_mvt']=CW.MVT_ID_mvt.astype('int64'); GW['CTOT_janela']=[c for c in CW.columns if c!='MVT_ID_mvt']; a=a.merge(CW.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); CN=pd.read_parquet('../prox/ctotnorm_tails_2025.parquet'); CN['MVT_ID_mvt']=CN.MVT_ID_mvt.astype('int64'); GW['CTOT_norm']=[c for c in CN.columns if c!='MVT_ID_mvt']; a=a.merge(CN.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); GW={k:GW[k] for k in ('Janela_pista','Janela_ref','CTOT_janela','CTOT_norm')}
newc=[c for k,v in GW.items() if k not in ('Janela_pista','Janela_ref','CTOT_janela','CTOT_norm') for c in v if c in W.columns and c not in a.columns]; a=a.merge(W[['MVT_ID_mvt']+newc].drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); G.update({k:v for k,v in GW.items()}); NEW=list(GW); print('grupos novos:',{k:len(G[k]) for k in NEW},'| cobertura Janela_obs (voos com trilha própria): %.1f%%'%(100*a['rg_wait_ready'].notna().mean()),flush=True)
tr=(a.holdout==0).values; te=(a.holdout==1).values; e_te=a.en.values[te]; N=int(te.sum()); R0=np.sqrt((e_te**2).mean()); y=a.en.clip(-3000,3000).values
hold=(((a.y-a.ma)>600)&a.ma.notna()).values[te]; apt_te=a.apt.values[te]; day_te=a.day.values[te]
P=dict(n_estimators=250,learning_rate=0.07,num_leaves=63,min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,max_bin=127,verbose=-1,n_jobs=14)
def fit_pred(cols,mask_tr,mask_te):
    m=lgb.LGBMRegressor(**P).fit(a.loc[mask_tr,cols],y[mask_tr]); return m.predict(a.loc[mask_te,cols])
def run(cols,name,per_apt=False):
    t0=time.time()
    if per_apt:
        c=np.zeros(N)
        for ap in sorted(a.apt.unique()):
            mt=tr&(a.apt==ap).values; me=te&(a.apt==ap).values; c[(a.apt.values[te]==ap)]=fit_pred(cols,mt,me)
    else: c=fit_pred(cols,tr,te)
    np.save(D+f'oofy_ctot6_{name}.npy',c)
    r={f'd{s}':np.sqrt(((e_te-s*c)**2).mean())-R0 for s in (0.25,0.5,0.75,1.0)}; r['esperas0.5']=np.sqrt(((e_te-0.5*c)[hold]**2).mean())-np.sqrt((e_te[hold]**2).mean()); r['n']=len(cols)
    print(f'{name:46s} n={len(cols):3d} | Δ dose0,25 {r["d0.25"]:+.3f} | 0,5 {r["d0.5"]:+.3f} | 0,75 {r["d0.75"]:+.3f} | 1,0 {r["d1.0"]:+.3f} | esperas(0,5) {r["esperas0.5"]:+.2f} ({time.time()-t0:.0f}s)',flush=True); return r,c
print(f'treino {tr.sum()} linhas (10 meses) | holdout {N} | RMSE base (campeão) {R0:.3f} | RMSE do resíduo de treino {np.sqrt((a.en[tr]**2).mean()):.1f}',flush=True)
OLD=[g for g in G if g not in NEW and g!='CLK' and not g.startswith('B_')]; CLK=G['CLK']; BASE=CLK+[c for g in OLD for c in G[g]]; BJ=BASE+G['Janela_pista']+G['Janela_ref']; CT=G['CTOT_janela']
CN_=G['CTOT_norm']; rows={}; oofs={}; BJC=BJ+CT
rows['BASE'],oofs['BASE']=run(BASE,'T_BASE'); rows['BASE+Janela (v99)'],oofs['BJ']=run(BJ,'T_BJ'); rows['BJ+CTOT_janela'],oofs['bjc']=run(BJC,'T_bjc')
VAR={k:[c for c in CN_ if c.startswith('cn3_'+k+'_')] for k in ('pow3','tailhi','tail2','exp')}
VAR['TODAS as caudas']=CN_
for k,v in VAR.items(): rows['BJC+'+k],oofs['bjc+'+k]=run(BJC+v,'T_'+re.sub(r'[^A-Za-z0-9]','',k)[:18])
days=pd.Series(day_te).unique(); idx={d:np.where(day_te==d)[0] for d in days}
def boot(c1,c0,s=0.5,B=1000):
    rng=np.random.default_rng(1); n1=(e_te-s*c1)**2; n0=(e_te-s*c0)**2; out=[]
    for _ in range(B):
        dd=rng.choice(days,len(days)); ii=np.concatenate([idx[d] for d in dd]); out.append(np.sqrt(n1[ii].mean())-np.sqrt(n0[ii].mean()))
    return np.mean(out),np.percentile(out,5),np.percentile(out,95)
for s_ in (0.25,0.5):
    print(f'\n== comparações com IC90 por bootstrap de dias (dose {s_}; Δ negativo = melhor o 1º)',flush=True)
    def show(nm,c1,c0): m,l,h=boot(c1,c0,s_); print(f'  {nm:62s} {m:+.3f} [{l:+.3f}; {h:+.3f}]',flush=True)
    show('BJ (v99) vs BASE',oofs['BJ'],oofs['BASE']); show('BJ+CTOT_janela vs BJ (v99)',oofs['bjc'],oofs['BJ'])
    for k in VAR: show(f'BJC+{k} vs BJ+CTOT_janela',oofs['bjc+'+k],oofs['bjc'])
    for k in VAR: show(f'BJC+{k} vs BJ (v99)',oofs['bjc+'+k],oofs['BJ'])
print('\n== por aeroporto (dose 0,5): (BJ+CT+CN_v2) − (BJ+CT)',flush=True)
tot=(e_te**2).sum(); out={}
for ap in sorted(set(apt_te)):
    m=apt_te==ap; f=lambda c: np.sqrt((tot-(e_te[m]**2).sum()+((e_te-0.5*c)[m]**2).sum())/N); out[ap]=round(float(f(oofs['bjc+TODAS as caudas'])-f(oofs['bjc'])),3)
print(out,flush=True); print('FIM',flush=True)
