import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_campo as SC
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); DAY=sys.argv[1]; CS=sys.argv[2]; S=pickle.load(open(f'{OUT}/solver_rev_{DAY}.pkl','rb')); T=S['T'].copy(); T['dur']=S['dur']; AL=S['AL']; PG=S['PG']; rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
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
T['W']=pred; T['yh']=T.base+T.dur+pred; T['e']=T.y-T.yh
oc=pd.read_parquet(os.path.expanduser('~/Projetos/oof_store/grafo_por_aeroporto/EGLL/cenario_b_oof.parquet'),columns=['id','champ']); T=T.merge(oc,on='id',how='left'); T['ec']=T.y-T.champ
# ---- 1) esse voo esta na cauda de y?
r=T[T.cs==CS].iloc[0]; allo=pd.read_parquet(os.path.expanduser('~/Projetos/oof_store/grafo_por_aeroporto/EGLL/cenario_b_oof.parquet'),columns=['y','day'])
print('VOO %s: y = %.0f s (%.1f min) | percentil de y no dia: %.1f%% | em jan+jul de Londres: %.1f%% | (p50=%.0f, p90=%.0f, p95=%.0f)'%(CS,r.y,r.y/60,100*(T.y<r.y).mean(),100*(allo.y<r.y).mean(),T.y.quantile(.5),T.y.quantile(.9),T.y.quantile(.95)))
T['faixa']=pd.cut(T.y,[0,900,1200,1500,1800,2100,2400],labels=['<15 min','15-20','20-25','25-30','30-35','35-40'])
g=T.groupby('faixa').agg(n=('y','size'),vies_solver=('e','mean'),rmse_solver=('e',lambda s:rm(s)),rmse_campeao=('ec',lambda s:rm(s.dropna())),parte_erro_solver=('e',lambda s:(s**2).sum()/(T.e**2).sum())); print(g.round(2).to_string())
print('os 10%% de maior y (n=%d) = %.0f%% do erro quadratico do solver'%(int(.1*len(T)),100*(T.e[T.y>=T.y.quantile(.9)]**2).sum()/(T.e**2).sum()))
# ---- 2) por que a velocidade nao caiu: campo ao longo da rota, em numeros absolutos
i=int(r.i); pr=dict(X=PG['X'],gamma=0.0,tau=PG['tau'],wads=PG['wads'],theta=PG['theta'],p=PG['p'],q=PG['q'],rev=True,gm={},em={},rec=dict(alvo_i=i,t_snap=None)); SC.simula_dia((DAY,pr)); rec=pr['rec']; tr_=np.array(rec['traj']); a=rec['ag'][rec['alvo']]
print('\nDIAGNOSTICO DA SIMULACAO DESSE VOO (regredindo do MVT):')
print('  caminho simulado: %.0f m | duracao livre (a %.1f m/s) = %.0f s | duracao simulada com rastros = %.0f s'%(a['tot'],SC.VREF,a['tot']/SC.VREF,a['fim']-tr_[-1,0]))
print('  campo de rastro NA POSICAO do aviao ao longo do taxi: medio %.2f | maximo %.2f | limiar theta = %.1f (expoente p = %.2f)'%(tr_[:,4].mean(),tr_[:,4].max(),PG['theta'],PG['p']))
print('  fator de velocidade f = 1/(1+(campo/theta)^p): medio %.3f | minimo %.3f  -> reducao maxima de %.0f%%'%(tr_[:,3].mean(),tr_[:,3].min(),100*(1-tr_[:,3].min())))
G=SC.graf(); Dia=pickle.load(open(f'{SC.R0}/{DAY}.pkl','rb')); ads=Dia['ads']; dEE=G['dEE']; ks=np.unique(a['ks'][a['ks']>=0]); t1=a['fim']; t0_=a['fim']-(a['fim']-tr_[-1,0])
perto=(dEE[:,ks].min(axis=1)<=1000); sel=(ads['t']>=t0_)&(ads['t']<=t1)&perto[ads['k']]; print('  avioes com ADS-B a ate 1 km (pelo grafo) da rota durante o taxi: %d distintos | pings desses avioes: %d'%(len(np.unique(ads['icao'][sel])),int(sel.sum())))
print('  por aviao, o nucleo (1-d/X)^q com X=%.1f km e q=%.2f vale %.2f a 500 m, %.2f a 1 km, %.2f a 2 km'%(PG['X']/1000,PG['q'],(1-500/PG['X'])**PG['q'],(1-1000/PG['X'])**PG['q'],(1-2000/PG['X'])**PG['q']))
print('  campo necessario para reduzir a velocidade pela metade: %.1f = ~%.0f avioes colados; parametros: tau=%.0f s, peso ADS-B=%.2f'%(PG['theta'],PG['theta'],PG['tau'],PG['wads']))
print('  composicao da previsao: base(MVT-ancora+patio) %.0f + movimento simulado %.0f + espera por contexto (ridge) %.0f = %.0f | y real %.0f'%(r.base,r.dur,r.W,r.yh,r.y))
