"""Plota, para o aeroporto APT, a simulacao do voo de MAIOR ERRO (y central 5-95%; erro = y - (exot_graf + 45)) com a calibracao v1: aeroporto completo, rota simulada, ADS-B do voo, ancora, stand, campo de rastro e demais avioes no instante do meio do trecho."""
import sys,os,json,pickle,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt,warnings; warnings.filterwarnings('ignore')
os.environ['SEM_GUIA']='1'
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_campo as SC, multi as M
APT=SC.APT; B='work/ablation/b'; os.makedirs(f'{B}/plots',exist_ok=True)
F=pd.read_parquet(f'{B}/feat_b_v1_{APT}.parquet'); T=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt'])
F=F.merge(T,on='MVT_ID_mvt'); lo,hi=F.TAXITIME_SEC_mvt.quantile([.05,.95]); F=F[(F.TAXITIME_SEC_mvt>=lo)&(F.TAXITIME_SEC_mvt<=hi)].copy(); F['pred']=F.exot_graf+45; F['err']=F.TAXITIME_SEC_mvt-F.pred
r0=F.loc[F.err.abs().idxmax()]; day=r0.day; mid=int(r0.MVT_ID_mvt)
G=SC.graf(); sy=G['sy']; g=sy.g; Dia=pickle.load(open(f'{SC.R0}/{day}.pkl','rb')); i=[k for k,x in enumerate(Dia['D']) if x['id']==mid][0]; r=Dia['D'][i]
PR=dict(X=3995.,theta=10.55,tau=318.9,wads=.3026,p=1.7576,q=1.4117); PR.update(json.load(open(f'{B}/params_{APT}.json'))['params']); PR.update(rev=True,stats=True,gm={},em={})
fim=r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-SC.ROLL_DEP; dur=r0.dur_sim_oculto; rec=dict(alvo_i=i,t_snap=fim-dur/2); PR['rec']=rec
res=SC.simula_dia((day,PR)); ed=r['ed_h'] if (r['chain'] and r['ed_h']) else r['ed']
fig,A=plt.subplots(figsize=(14,14)); I=rec.get('I'); ea,eb=sy.ea,sy.eb
for k_,e in enumerate(g.edges):
    pass
for k in range(len(ea)):
    a_,b_=g.xy[ea[k]],g.xy[eb[k]]; v=0 if I is None else I[k]; c=plt.cm.Reds(min(1,np.log1p(v)/np.log1p(max(1e-9,np.nanmax(I)))) ) if (I is not None and v>0) else '#e3e5e8'
    A.plot([a_[0],b_[0]],[a_[1],b_[1]],color=c,lw=2 if v>0 else 1,zorder=1)
for e in g.edges:
    if e['kind']!='taxi': a_,b_=g.xy[e['a']],g.xy[e['b']]; A.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#b5babf',lw=4,zorder=0)
for q,k in rec.get('agentes',[]): m=(g.xy[ea[k]]+g.xy[eb[k]])/2 if k>=0 else None; (A.scatter(*m,s=22,color='#555',zorder=3) if m is not None else None)
xy=M.poly_ed(type('N',(),dict(sy=sy,g=g))(),ed,r['xy']); A.plot(xy[:,0],xy[:,1],color='#1f618d',lw=6,zorder=6,label='rota simulada (%.0f m)'%np.sum([e[1] for e in ed]))
if r['chain'] and r.get('pings') is not None: A.plot(np.asarray(r['pings'])[:,0],np.asarray(r['pings'])[:,1],'-o',color='k',lw=2,ms=5,zorder=7,label='ADS-B do voo (ancora -> MVT)')
A.scatter(*r['xy'],marker='*',s=900,color='#f1c40f',edgecolors='k',zorder=9,label='stand %s'%r['stand'])
if r['chain']: A.scatter(*r['Pstar'],marker='s',s=300,color='k',zorder=9,label='ancora ADS-B')
A.scatter(*xy[-1],marker='D',s=250,color='#c0392b',zorder=9,label='fim do trecho simulado')
mx=150; A.set_xlim(g.xy[:,0].min()-mx,g.xy[:,0].max()+mx); A.set_ylim(g.xy[:,1].min()-mx,g.xy[:,1].max()+mx); A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([]); A.legend(loc='lower left',fontsize=11)
A.set_title('%s %s %s | y real %.0f s | previsto %.0f s (sim %.0f + ADS %.0f + 45) | erro %+.0f s | tem_ads=%d'%(APT,day,r['cs'],r0.TAXITIME_SEC_mvt,r0.pred,r0.dur_sim_oculto,0 if r0.t_ads_obs!=r0.t_ads_obs else r0.t_ads_obs,r0.err,r0.tem_ads),fontsize=12); plt.tight_layout(); plt.savefig(f'{B}/plots/maior_erro_{APT}.png',dpi=90)
print(APT,day,r['cs'],'stand',r['stand'],'y %.0f'%r0.TAXITIME_SEC_mvt,'prev %.0f'%r0.pred,'erro %+.0f'%r0.err,'cadeia',int(r['chain']),flush=True)
