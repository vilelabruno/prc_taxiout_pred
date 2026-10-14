import sys, glob, numpy as np, pandas as pd, lightgbm as lgb, warnings, os; warnings.filterwarnings('ignore')
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store')
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/{APT}/cenario_b_oof.parquet')
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{S}/sincronia_grafo/fisico/{APT}_*.parquet'))])
d=o.merge(f.drop(columns=['day','mvt']),on='id',how='left'); d['d']=pd.to_datetime(d.day); d['tem']=d.tem.fillna(0)
d['hora']=pd.to_datetime(d.d).dt.dayofweek
print(APT,len(d),'| com trilha de solo:',round(d.tem.mean(),3),'| jan',round(d[d.mes=='2025-01'].tem.mean(),3),'jul',round(d[d.mes=='2025-07'].tem.mean(),3),'| com inicio do taxi (t_k):',round(d.t_k.notna().mean(),3),'| com onset:',round(d.onset.notna().mean(),3),'| ligacao 2:',int((d.lig==2).sum()))
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
dm=d.copy(); tk=dm[dm.mvt_k.notna()]; print('voos com t_k:',len(tk),'| y - mvt_k: mediana',round((tk.y-tk.mvt_k).median(),0),'desvio',round((tk.y-tk.mvt_k).std(),0),'| RMSE do campeao neles',round(rm(tk.y-tk.champ),1),'| RMSE de (mvt_k + constante)',round(rm(tk.y-tk.mvt_k-(tk.y-tk.mvt_k).median()),1))
fp=dm[dm.mvt_prim.notna()]; r=fp.y-fp.mvt_prim-fp.tau_acc; print('voos com 1o ping:',len(fp),'| RMSE do campeao',round(rm(fp.y-fp.champ),1),'| RMSE de (mvt_prim + tau_acc + c)',round(rm(r-r.median()),1),'c=',round(r.median(),0))
feats=['tem','lig','mvt_prim','mvt_ult','mvt_k','mvt_onset','d0','esp_fora','esp_pat','D_rota','D_tipo','tau_acc','gs_first','gs_last','n','gap_onset']
days=np.array(sorted(d.d.unique())); blk={x:i//5 for i,x in enumerate(days)}; d['blk']=d.d.map(blk)
P=dict(n_estimators=250,learning_rate=.03,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1)
def cv(cols,alvo='res',shrink=1.0):
    pred=np.zeros(len(d))
    for b in d.blk.unique():
        te=(d.blk==b).to_numpy(); lo=d.d[te].min()-pd.Timedelta(days=1); hi=d.d[te].max()+pd.Timedelta(days=1)
        tr=((d.d<lo)|(d.d>hi)).to_numpy()
        yv=(d.y-d.champ) if alvo=='res' else d.y
        m=lgb.LGBMRegressor(**P).fit(d.loc[tr,cols],yv[tr]); p=m.predict(d.loc[te,cols])
        pred[te]=shrink*p+(d.champ[te] if alvo=='res' else 0) if alvo=='res' else p
    return pred
base=d.y-d.champ; print('\nRMSE do campeao: todas',round(rm(base),1),'| com trilha',round(rm(base[d.tem==1]),1),'| sem trilha',round(rm(base[d.tem==0]),1))
for nome,cols,alvo,sh in [('residuo: campeao+hora (controle)',['champ','hora'],'res',1.0),('residuo: + features fisicas',['champ','hora']+feats,'res',1.0),('residuo: + features fisicas, encolhido 0.5',['champ','hora']+feats,'res',0.5),('so fisico (sem campeao), alvo y',feats+['hora'],'y',1.0)]:
    p=cv(cols,alvo,sh); e=d.y-p
    # IC por dia
    rng=np.random.default_rng(0); idx={k:np.where(d.d.to_numpy()==k)[0] for k in days}; out=[]
    for _ in range(300):
        ii=np.concatenate([idx[k] for k in rng.choice(days,len(days))]); out.append(rm(e.to_numpy()[ii])-rm(base.to_numpy()[ii]))
    t=d.tem==1; print(f'{nome:44s} RMSE todas {rm(e):6.1f} (delta {rm(e)-rm(base):+6.2f} IC90 [{np.percentile(out,5):+.2f},{np.percentile(out,95):+.2f}]) | com trilha {rm(e[t]):6.1f} contra campeao {rm(base[t]):6.1f} | jan {rm(e[d.mes=="2025-01"])-rm(base[d.mes=="2025-01"]):+.2f} jul {rm(e[d.mes=="2025-07"])-rm(base[d.mes=="2025-07"]):+.2f}')
