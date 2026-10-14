import sys, os, pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_campo as SC
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); DAY=sys.argv[1]; S=pickle.load(open(f'{OUT}/solver_rev_{DAY}.pkl','rb')); T=S['T'].copy(); T['dur']=S['dur']; AL=S['AL']; PG=S['PG']; rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
FACT=['tipo','hb','clima','pier']; folds=np.array_split(np.arange(len(T)),5)
def onehot(tr_,te_):
    A=[];B=[]
    for k in FACT:
        cats=pd.Categorical(tr_[k]); ix={c:i for i,c in enumerate(cats.categories)}
        def oh(df):
            ii=df[k].map(ix); m=ii.notna().to_numpy(); r=np.where(m)[0]; return sparse.csr_matrix((np.ones(m.sum()),(r,ii[m].astype(int).to_numpy())),shape=(len(df),len(ix)))
        A.append(oh(tr_)); B.append(oh(te_))
    A.append(sparse.csr_matrix(tr_[['chain']].to_numpy(float))); B.append(sparse.csr_matrix(te_[['chain']].to_numpy(float))); return sparse.hstack(A).tocsr(),sparse.hstack(B).tocsr()
T['res']=T.y-T.base-T.dur; pred=np.zeros(len(T))
for f in folds:
    te=np.zeros(len(T),bool); te[f]=True; Xtr,Xte=onehot(T[~te],T[te]); pred[te]=Ridge(alpha=AL).fit(Xtr,T.res[~te]).predict(Xte)
T['yh']=T.base+T.dur+pred; T['e']=T.y-T.yh
oc=pd.read_parquet(os.path.expanduser('~/Projetos/oof_store/grafo_por_aeroporto/EGLL/cenario_b_oof.parquet'),columns=['id','champ']); T=T.merge(oc,on='id',how='left')
print('RMSE do dia (CV): %.1f | campeao %.1f'%(rm(T.e),rm((T.y-T.champ).dropna())))
print('duracao simulada do trecho escondido: media %.0f s (sem rastros, 146 s) | mediana %.0f | p90 %.0f | max %.0f | acima de 600 s: %.1f%% dos voos'%(T.dur.mean(),T.dur.median(),T.dur.quantile(.9),T.dur.max(),100*(T.dur>600).mean()))
print('RMSE por tem-cadeia:',{k:round(rm(g.e),1) for k,g in T.groupby('chain')})
w=T.reindex(T.e.abs().sort_values(ascending=False).index).head(8); print(w[['cs','stand','tipo','chain','y','yh','champ','dur','e']].round(0).to_string(index=False))
r=w.iloc[0]; i=int(r.i); print('\nMAIOR ERRO:',r.cs,'stand',r.stand,'| y real %.0f s | previsto %.0f s | campeao %.0f s | duracao simulada do trecho %.0f s'%(r.y,r.yh,r.champ,r.dur))
# roda de novo com gravacao para esse voo
pr=dict(X=PG['X'],gamma=0.0,tau=PG['tau'],wads=PG['wads'],theta=PG['theta'],p=PG['p'],q=PG['q'],rev=True,gm={},em={},rec=dict(alvo_i=i,t_snap=None)); R=SC.simula_dia((DAY,pr)); rec=pr['rec']; traj=np.array(rec['traj']); ag=rec['ag']; a=ag[rec['alvo']]
parado=(traj[:,3]<0.2); print('trajetoria regredindo do MVT: %.0f s no total | %.0f s com velocidade < 20%% da livre | campo maximo %.1f (theta=%.2f) | fracao do caminho percorrida quando parou mais: %.0f%%'%(a['fim']-traj[-1,0],parado.sum()*5/ (1) *0+parado.sum()*(SC.UNIT/SC.VREF),traj[:,4].max(),PG['theta'],100*traj[np.argmin(traj[:,3]),1]/a['tot']))
tpk=float(traj[np.argmax(traj[:,4]),0]); pr['rec']=dict(alvo_i=i,t_snap=tpk); R=SC.simula_dia((DAY,pr)); rec=pr['rec']; I=rec['I']; ag=rec['ag']
G=SC.graf(); sy=G['sy']; g=sy.g; Dia=pickle.load(open(f'{SC.R0}/{DAY}.pkl','rb')); rr=Dia['D'][i]
import multi as M
class N: pass
n=N(); n.sy=sy; n.g=g; edges=[(k,1.0) for k in a['ks']]; PL=M.poly_ed(n,edges,rr['xy'])
fig,A_=plt.subplots(figsize=(15,15))
for e in g.edges: a_,b_=g.xy[e['a']],g.xy[e['b']]; A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color='#e3e5e8' if e['kind']=='taxi' else '#b5babf',lw=1 if e['kind']=='taxi' else 4,zorder=1)
imax=max(I.max(),1e-6)
for k in np.argsort(I):
    if I[k]<0.04*imax: continue
    a_,b_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; qq=min(I[k]/imax,1); A_.plot([a_[0],b_[0]],[a_[1],b_[1]],color=plt.cm.YlOrRd(0.15+0.85*qq),lw=2+6*qq,alpha=.85,zorder=2,solid_capstyle='round')
if rr['chain'] and rr['pings'] is not None: A_.plot(rr['pings'][:,0],rr['pings'][:,1],color='k',lw=1.6,ls=':',zorder=4,label='ADS-B real do voo (ancora)')
A_.plot(PL[:,0],PL[:,1],color='#1f618d',lw=5,zorder=5,solid_capstyle='round',label='rota simulada (regredindo do MVT ate o stand)'); A_.scatter(*PL[0],marker='*',s=800,color='#f1c40f',edgecolors='k',zorder=8,label='stand'); A_.scatter(*PL[-1],marker='s',s=300,color='k',zorder=8,label='ancora no MVT (P* do ADS ou entrada da pista)')
for q,e in rec['agentes']:
    if q==rec['alvo']: continue
    A_.scatter(*((g.xy[sy.ea[e]]+g.xy[sy.eb[e]])/2),s=60,color='#222',zorder=7)
A_.scatter([],[],s=60,color='#222',label='outros avioes taxiando agora (origem dos rastros)'); A_.plot([],[],color='#e74c3c',lw=5,label='campo de rastro no instante de maior bloqueio (mais escuro = mais forte)'); A_.legend(loc='lower right',fontsize=11)
mx=150; A_.set_xlim(g.xy[:,0].min()-mx,g.xy[:,0].max()+mx); A_.set_ylim(g.xy[:,1].min()-mx,g.xy[:,1].max()+mx); A_.set_aspect('equal'); A_.set_xticks([]); A_.set_yticks([])
A_.set_title(f'{DAY} {r.cs} stand {r.stand}: MAIOR ERRO | y real {r.y:.0f} s, previsto {r.yh:.0f} s, campeao {r.champ:.0f} s | duracao simulada do trecho {r.dur:.0f} s | instante {(tpk-a["fim"]):.0f} s do MVT',fontsize=12); plt.tight_layout(); plt.savefig(f'{OUT}/rev_maior_erro.png',dpi=80); print('plot salvo')
