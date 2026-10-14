"""Ablação com TREINO NO ANO INTEIRO: treina nos 10 meses de treino (resíduo = y − proxy do campeão) e avalia no OOF padrão do projeto (jan+jul 2025) sobre o resíduo do campeão atual (frame_result_best, proxy da v92).
 Configurações: CLK; TODOS; CLK+cada grupo (add-one); TODOS menos cada grupo (leave-one-out); e, para CLK e TODOS, um modelo por aeroporto.
 Métricas no holdout (342.930 linhas): ΔRMSE total em doses 0,25/0,5/0,75/1,0 e RMSE nas esperas (AOBT−BLOCK > 10 min; diagnóstico), por aeroporto; IC por bootstrap de dias nas comparações-chave."""
import json, os, time, numpy as np, pandas as pd, lightgbm as lgb
D=os.path.expanduser('~/yearabl/'); a=pd.read_parquet(D+'year_table.parquet'); G=json.load(open(D+'groups_year.json'))
bf=os.path.expanduser('~/ablation/b/feat_b_v1.parquet')
if os.path.exists(bf) and os.path.exists(os.path.expanduser('~/ablation/b/groups_b.json')):
    b=pd.read_parquet(bf); b['MVT_ID_mvt']=b.MVT_ID_mvt.astype('int64'); gb=json.load(open(os.path.expanduser('~/ablation/b/groups_b.json'))); keep=[c for v in gb.values() for c in v if c in b.columns and c not in a.columns]
    a=a.merge(b[['MVT_ID_mvt']+keep].drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); G.update({'B_'+k:[c for c in v if c in keep] for k,v in gb.items()}); print('grupos B:',{k:len(v) for k,v in G.items() if k.startswith('B_')},flush=True)
tr=(a.holdout==0).values; te=(a.holdout==1).values; e_te=a.en.values[te]; N=int(te.sum()); R0=np.sqrt((e_te**2).mean()); y=a.en.clip(-3000,3000).values
hold=(((a.y-a.ma)>600)&a.ma.notna()).values[te]; apt_te=a.apt.values[te]; day_te=a.day.values[te]
P=dict(n_estimators=250,learning_rate=0.07,num_leaves=63,min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,max_bin=127,verbose=-1,n_jobs=32)
def fit_pred(cols,mask_tr,mask_te):
    m=lgb.LGBMRegressor(**P).fit(a.loc[mask_tr,cols],y[mask_tr]); return m.predict(a.loc[mask_te,cols])
def run(cols,name,per_apt=False):
    t0=time.time()
    if per_apt:
        c=np.zeros(N)
        for ap in sorted(a.apt.unique()):
            mt=tr&(a.apt==ap).values; me=te&(a.apt==ap).values; c[(a.apt.values[te]==ap)]=fit_pred(cols,mt,me)
    else: c=fit_pred(cols,tr,te)
    np.save(D+f'oofy_v1_{name}.npy',c)
    r={f'd{s}':np.sqrt(((e_te-s*c)**2).mean())-R0 for s in (0.25,0.5,0.75,1.0)}; r['esperas0.5']=np.sqrt(((e_te-0.5*c)[hold]**2).mean())-np.sqrt((e_te[hold]**2).mean()); r['n']=len(cols)
    print(f'{name:46s} n={len(cols):3d} | Δ dose0,25 {r["d0.25"]:+.3f} | 0,5 {r["d0.5"]:+.3f} | 0,75 {r["d0.75"]:+.3f} | 1,0 {r["d1.0"]:+.3f} | esperas(0,5) {r["esperas0.5"]:+.2f} ({time.time()-t0:.0f}s)',flush=True); return r,c
print(f'treino {tr.sum()} linhas (10 meses) | holdout {N} | RMSE base (campeão) {R0:.3f} | RMSE do resíduo de treino {np.sqrt((a.en[tr]**2).mean()):.1f}',flush=True)
GR=[g for g in G if g!='CLK']; CLK=G['CLK']; ALL=CLK+[c for g in GR for c in G[g]]; rows={}; oofs={}
rows['CLK'],oofs['CLK']=run(CLK,'CLK'); rows['TODOS'],oofs['TODOS']=run(ALL,'TODOS')
for g in GR: rows['so+'+g],oofs['so+'+g]=run(CLK+G[g],'so_'+g)
for g in GR: rows['sem '+g],oofs['sem '+g]=run([c for c in ALL if c not in G[g]],'sem_'+g)
rows['CLK|1 modelo por aeroporto'],oofs['CLK|apt']=run(CLK,'CLK_porapt',True); rows['TODOS|1 modelo por aeroporto'],oofs['TODOS|apt']=run(ALL,'TODOS_porapt',True)
res=pd.DataFrame(rows).T; res.to_csv(D+'ablation_year_result_v1.csv')
# bootstrap por dias: TODOS vs CLK, e cada grupo (so+ vs CLK; sem vs TODOS), dose 0,5
days=pd.Series(day_te).unique(); idx={d:np.where(day_te==d)[0] for d in days}
def boot(c1,c0,s=0.5,B=1000):
    rng=np.random.default_rng(1); n1=(e_te-s*c1)**2; n0=(e_te-s*c0)**2; out=[]
    for _ in range(B):
        dd=rng.choice(days,len(days)); ii=np.concatenate([idx[d] for d in dd]); out.append(np.sqrt(n1[ii].mean())-np.sqrt(n0[ii].mean()))
    return np.mean(out),np.percentile(out,5),np.percentile(out,95)
print('\n== comparações com IC90 por bootstrap de dias (dose 0,5; Δ negativo = melhor o 1º)',flush=True)
def show(nm,c1,c0): m,l,h=boot(c1,c0); print(f'  {nm:62s} {m:+.3f} [{l:+.3f}; {h:+.3f}]',flush=True)
show('TODOS vs CLK',oofs['TODOS'],oofs['CLK']); show('TODOS (1 modelo por aeroporto) vs TODOS (um só)',oofs['TODOS|apt'],oofs['TODOS']); show('CLK (por aeroporto) vs CLK (um só)',oofs['CLK|apt'],oofs['CLK'])
for g in GR: show(f'só+{g} vs CLK',oofs['so+'+g],oofs['CLK'])
for g in GR: show(f'TODOS sem {g} vs TODOS',oofs['sem '+g],oofs['TODOS'])
# por aeroporto: TODOS vs CLK
print('\n== por aeroporto (Δ do RMSE total atribuível ao aeroporto, dose 0,5): TODOS − CLK',flush=True)
tot=(e_te**2).sum(); out={}
for ap in sorted(set(apt_te)):
    m=apt_te==ap; out[ap]=round(float(np.sqrt((tot-(e_te[m]**2).sum()+((e_te-0.5*oofs['TODOS'])[m]**2).sum())/N)-np.sqrt((tot-(e_te[m]**2).sum()+((e_te-0.5*oofs['CLK'])[m]**2).sum())/N)),3)
print(out,flush=True); print('FIM',flush=True)
