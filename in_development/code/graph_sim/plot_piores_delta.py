import numpy as np, pandas as pd, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
S=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); APT='EGLL'
d=pd.read_parquet(f'{S}/grafo_completo_{APT}.parquet')
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','CALLSIGN_flt','FLIGHT_mvt','STAND_mvt']).rename(columns={'MVT_ID_mvt':'id'}); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
d=d.merge(tr[['id','cs','STAND_mvt']],on='id'); d['eg']=d.y-d.yh; d['ec']=d.y-d.champ; d['delta']=d.eg.abs()-d.ec.abs()
tot=(d.eg**2).sum(); print('RMSE grafo',round(np.sqrt((d.eg**2).mean()),1),'| campeao',round(np.sqrt((d.ec**2).mean()),1))
for k in (10,50,100,500,1000): 
    ix=d.eg.abs().nlargest(k).index; print(f'os {k} piores do grafo = {(d.eg[ix]**2).sum()/tot:.1%} do erro quadratico do grafo | no campeao esses mesmos {k} = {(d.ec[ix]**2).sum()/(d.ec**2).sum():.1%} do dele')
print('grafo erra mais que o campeao em',round((d.delta>0).mean(),3),'dos voos | delta mediano (s)',round(d.delta.median(),0),'| delta medio',round(d.delta.mean(),0))
w=d.reindex(d.eg.abs().sort_values(ascending=False).index).head(20)[['day','cs','STAND_mvt','y','yh','champ','eg','ec','delta','tem']]; pd.set_option('display.width',200); print(w.round(0).to_string(index=False))
fig,ax=plt.subplots(1,3,figsize=(21,6.5)); c1='#c0392b'; c2='#2a6f97'
top=w.iloc[::-1]; yy=np.arange(len(top)); ax[0].barh(yy+.2,top.eg.abs()/60,.4,color=c1,label='|erro| do grafo'); ax[0].barh(yy-.2,top.ec.abs()/60,.4,color=c2,label='|erro| do campeao')
ax[0].set_yticks(yy); ax[0].set_yticklabels([f"{r.day[5:]} {r.cs}" for r in top.itertuples()],fontsize=8); ax[0].set_xlabel('minutos'); ax[0].set_title('20 piores erros do grafo (e o erro do campeao nos mesmos voos)'); ax[0].legend()
ax[1].scatter(d.ec/60,d.eg/60,s=3,alpha=.25,color='#7f8c8d'); ax[1].scatter(w.ec/60,w.eg/60,s=30,color=c1,zorder=3,label='20 piores do grafo'); lim=max(abs(d.eg).max(),abs(d.ec).max())/60
ax[1].plot([-lim,lim],[-lim,lim],color='k',lw=.8); ax[1].axhline(0,color='k',lw=.4); ax[1].axvline(0,color='k',lw=.4); ax[1].set_xlabel('erro do campeao (min)  [real - previsto]'); ax[1].set_ylabel('erro do grafo (min)'); ax[1].set_title('erro do grafo x erro do campeao, voo a voo'); ax[1].legend()
ax[2].hist((d.delta/60).clip(-30,60),bins=140,color=c2); ax[2].set_yscale('log'); ax[2].axvline(0,color='k',lw=.8); ax[2].set_xlabel('delta = |erro grafo| - |erro campeao| (min), cortado em -30/+60'); ax[2].set_title('delta (positivo: o grafo erra mais)')
plt.tight_layout(); plt.savefig(f'{S}/grafo_piores_delta.png',dpi=105)
sel=d.reindex(d.eg.abs().sort_values(ascending=False).index).head(6); sel[['day','cs','yh','champ']].to_csv(f'{S}/casos_piores.csv',index=False)
