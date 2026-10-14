import numpy as np, pandas as pd, warnings, os, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=os.path.expanduser('~/Projetos/oof_store')
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/LIRF/cenario_b_oof.parquet')
pa=pd.read_parquet(f'{S}/defeitos_prio/pi_A.parquet'); flags=[c for c in pa.columns if c.startswith('flag90_')]
pa['any90']=pa[flags].any(axis=1); o=o.merge(pa[['MVT_ID_mvt','any90']].rename(columns={'MVT_ID_mvt':'id'}),on='id',how='left'); o['any90']=o.any90.fillna(False)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','SCHED_TIME_UTC_mvt'])
o=o.merge(tr.rename(columns={'MVT_ID_mvt':'id'}),on='id',how='left')
o['t']=pd.to_datetime(o.MVT_TIME_UTC_mvt,utc=True); o['hora']=o.t.dt.hour
d=o[(~o.any90)&(o.hora>=6)&(o.y<21600)].copy(); d['pred']=d.b_relogio_mais_grafo; d['e']=d.y-d.pred; d['ec']=d.y-d.champ
print('n',len(d),'RMSE grafo',round(np.sqrt((d.e**2).mean()),1),'campeao',round(np.sqrt((d.ec**2).mean()),1))
q=d.e.abs().sort_values(ascending=False); tot=(d.e**2).sum()
for k in (10,50,100,250,500,1000): print(f'os {k} maiores erros = {(d.e.loc[q.index[:k]]**2).sum()/tot:.1%} do erro quadratico')
print('subestimado (y>pred):',round(((d.e>0)*d.e**2).sum()/tot,3),'| superestimado:',round(((d.e<0)*d.e**2).sum()/tot,3))
w=d.loc[q.index[:25],['day','t','STAND_mvt','AIRCRAFT_OPERATOR_flt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','y','pred','champ','e']].copy(); w['t']=w.t.dt.strftime('%m-%d %H:%M'); pd.set_option('display.width',220); print(w.round(0).to_string(index=False))
# por faixa de y
d['faixa']=pd.cut(d.y,[0,600,900,1200,1800,3000,6000,21600]); g=d.groupby('faixa').agg(n=('y','size'),vies=('e','mean'),rmse=('e',lambda s:np.sqrt((s**2).mean())),parte=('e',lambda s:(s**2).sum()/tot)); print(g.round(2))
# por dia e por hora
dd=d.groupby('day').agg(n=('y','size'),rmse=('e',lambda s:np.sqrt((s**2).mean())),vies=('e','mean'),parte=('e',lambda s:(s**2).sum()/tot)).sort_values('parte',ascending=False); print(dd.head(8).round(2))
fig,ax=plt.subplots(2,2,figsize=(15,11)); A=ax.ravel(); c='#2a6f97'; r='#c0392b'
A[0].scatter(d.pred,d.y,s=3,alpha=.25,color=c); m=max(d.y.max(),d.pred.max()); A[0].plot([0,m],[0,m],color='k',lw=.8)
top=d.loc[q.index[:60]]; A[0].scatter(top.pred,top.y,s=22,color=r,zorder=3,label='60 piores erros'); A[0].set_xlabel('previsto pelo grafo (s)'); A[0].set_ylabel('real y (s)'); A[0].set_title('previsto x real (LIRF, jan+jul 2025)'); A[0].legend(); A[0].set_xlim(0,3500)
A[1].hist(d.e.clip(-3000,6000),bins=120,color=c); A[1].set_yscale('log'); A[1].axvline(0,color='k',lw=.8); A[1].set_xlabel('erro = real − grafo (s), cortado em −3000/6000'); A[1].set_title('distribuição do erro (escala log)')
h=d.groupby('hora').agg(vies=('e','mean'),rmse=('e',lambda s:np.sqrt((s**2).mean()))); A[2].bar(h.index-.2,h.rmse,.4,color=c,label='RMSE'); A[2].bar(h.index+.2,h.vies,.4,color=r,label='viés (real − previsto)'); A[2].axhline(0,color='k',lw=.6); A[2].set_xlabel('hora UTC do MVT'); A[2].set_title('erro por hora do dia'); A[2].legend()
A[3].barh(range(len(g)),g.parte.values*100,color=c); A[3].set_yticks(range(len(g))); A[3].set_yticklabels([str(i) for i in g.index]); A[3].set_xlabel('% do erro quadrático'); A[3].set_title('de onde vem o erro, por faixa do y real (s)')
for i,(p,n,v) in enumerate(zip(g.parte,g.n,g.vies)): A[3].text(p*100+.5,i,f'n={n}  viés {v:+.0f} s',va='center',fontsize=9)
plt.tight_layout(); plt.savefig(f'{S}/sincronia_grafo/lirf_piores_erros.png',dpi=110)
