"""Ablação com TREINO NO ANO INTEIRO: treina nos 10 meses de treino (resíduo = y − proxy do campeão) e avalia no OOF padrão do projeto (jan+jul 2025) sobre o resíduo do campeão atual (frame_result_best, proxy da v92).
 Configurações: CLK; TODOS; CLK+cada grupo (add-one); TODOS menos cada grupo (leave-one-out); e, para CLK e TODOS, um modelo por aeroporto.
 Métricas no holdout (342.930 linhas): ΔRMSE total em doses 0,25/0,5/0,75/1,0 e RMSE nas esperas (AOBT−BLOCK > 10 min; diagnóstico), por aeroporto; IC por bootstrap de dias nas comparações-chave."""
import json, os, time, numpy as np, pandas as pd, lightgbm as lgb
D='/private/tmp/claude-502/-Users-bruno-quirino-latam-com-Desktop-bruno-Projetos-prc-data-challenge/5e41970f-79c3-434a-8d98-364545fe860d/scratchpad/ablmac/'; a=pd.read_parquet(D+'year_table.parquet'); G=json.load(open(D+'groups_year.json'))
W=pd.read_parquet('../prox/winfeat_all.parquet'); W['MVT_ID_mvt']=W.MVT_ID_mvt.astype('int64'); GW=json.load(open('../prox/groups_win.json'))
RW=pd.read_parquet('../prox/rwy_windows.parquet').drop(columns=['apt','rwy','rf_dep_gap_prev','rf_dep_gap_next']); RW['MVT_ID_mvt']=RW.MVT_ID_mvt.astype('int64'); GW['Janela_pista']=[c for c in RW.columns if c!='MVT_ID_mvt']
a=a.merge(RW.drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left')
newc=[c for k,v in GW.items() if k!='Janela_pista' for c in v if c in W.columns and c not in a.columns]; a=a.merge(W[['MVT_ID_mvt']+newc].drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); G.update({k:([c for c in v if c in newc] if k!='Janela_pista' else v) for k,v in GW.items()}); NEW=list(GW); print('grupos novos:',{k:len(G[k]) for k in NEW},'| cobertura Janela_obs (voos com trilha própria): %.1f%%'%(100*a[G['Janela_obs'][0]].notna().mean()),flush=True)
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
    np.save(D+f'oofy_rwy_{name}.npy',c)
    r={f'd{s}':np.sqrt(((e_te-s*c)**2).mean())-R0 for s in (0.25,0.5,0.75,1.0)}; r['esperas0.5']=np.sqrt(((e_te-0.5*c)[hold]**2).mean())-np.sqrt((e_te[hold]**2).mean()); r['n']=len(cols)
    print(f'{name:46s} n={len(cols):3d} | Δ dose0,25 {r["d0.25"]:+.3f} | 0,5 {r["d0.5"]:+.3f} | 0,75 {r["d0.75"]:+.3f} | 1,0 {r["d1.0"]:+.3f} | esperas(0,5) {r["esperas0.5"]:+.2f} ({time.time()-t0:.0f}s)',flush=True); return r,c
print(f'treino {tr.sum()} linhas (10 meses) | holdout {N} | RMSE base (campeão) {R0:.3f} | RMSE do resíduo de treino {np.sqrt((a.en[tr]**2).mean()):.1f}',flush=True)
OLD=[g for g in G if g not in NEW and g!='CLK' and not g.startswith('B_')]; CLK=G['CLK']; BASE=CLK+[c for g in OLD for c in G[g]]; ALLN=BASE+[c for g in NEW for c in G[g]]; rows={}; oofs={}
rows['CLK'],oofs['CLK']=run(CLK,'W_CLK'); rows['BASE (todos os grupos antigos)'],oofs['BASE']=run(BASE,'W_BASE'); rows['BASE+novos'],oofs['ALLN']=run(ALLN,'W_ALLN')
for g in NEW: rows['CLK+'+g],oofs['so+'+g]=run(CLK+G[g],'R_so_'+g)
for g in NEW: rows['BASE+'+g],oofs['b+'+g]=run(BASE+G[g],'R_base_'+g)
for g in NEW: rows['BASE+novos sem '+g],oofs['sem '+g]=run([c for c in ALLN if c not in G[g]],'R_sem_'+g)
res=pd.DataFrame(rows).T; res.to_csv(D+'ablation_rwy_result.csv')
days=pd.Series(day_te).unique(); idx={d:np.where(day_te==d)[0] for d in days}
def boot(c1,c0,s=0.5,B=1000):
    rng=np.random.default_rng(1); n1=(e_te-s*c1)**2; n0=(e_te-s*c0)**2; out=[]
    for _ in range(B):
        dd=rng.choice(days,len(days)); ii=np.concatenate([idx[d] for d in dd]); out.append(np.sqrt(n1[ii].mean())-np.sqrt(n0[ii].mean()))
    return np.mean(out),np.percentile(out,5),np.percentile(out,95)
for s_ in (0.25,0.5):
    print(f'\n== comparações com IC90 por bootstrap de dias (dose {s_}; Δ negativo = melhor o 1º)',flush=True)
    def show(nm,c1,c0): m,l,h=boot(c1,c0,s_); print(f'  {nm:62s} {m:+.3f} [{l:+.3f}; {h:+.3f}]',flush=True)
    show('BASE vs CLK',oofs['BASE'],oofs['CLK']); show('BASE+novos vs BASE',oofs['ALLN'],oofs['BASE'])
    for g in NEW: show(f'CLK+{g} vs CLK',oofs['so+'+g],oofs['CLK'])
    for g in NEW: show(f'BASE+{g} vs BASE',oofs['b+'+g],oofs['BASE'])
    for g in NEW: show(f'BASE+novos sem {g} vs BASE+novos',oofs['sem '+g],oofs['ALLN'])
print('\n== por aeroporto (Δ RMSE atribuível, dose 0,5): BASE+novos − BASE',flush=True)
tot=(e_te**2).sum(); out={}
for ap in sorted(set(apt_te)):
    m=apt_te==ap; f=lambda c: np.sqrt((tot-(e_te[m]**2).sum()+((e_te-0.5*c)[m]**2).sum())/N)
    out[ap]=round(float(f(oofs['ALLN'])-f(oofs['BASE'])),3)
print(out,flush=True)
# por subconjunto: voos com trilha própria (onde Janela_obs existe) vs sem
own=a.loc[te,G['Janela_obs'][0]].notna().values
for nm,c in (('BASE',oofs['BASE']),('BASE+novos',oofs['ALLN'])): print(f'  RMSE do resíduo (dose 0,5) nos voos COM trilha própria ({own.sum()}): {np.sqrt(((e_te-0.5*c)[own]**2).mean()):.3f} | base sem correção {np.sqrt((e_te[own]**2).mean()):.3f}',flush=True)
print('FIM',flush=True)
