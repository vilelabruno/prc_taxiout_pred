"""Calibracao do simulador por aeroporto: busca (aleatoria + Nelder-Mead) dos parametros X,tau,theta,p,q,wads que minimizam o RMSE de y ~ a + b*exot_graf (OLS) em dias de TREINO; avalia em dias de TESTE.
y central (833..2108 s em Londres; para os outros, 5%-95% do proprio aeroporto). Nunca usa BLOCK nem y como feature; y so entra no alvo da calibracao."""
import os,sys,json,glob,time,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
os.environ['SEM_GUIA']='1'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import sim_campo as SC, feat_b_v0 as F
from multiprocessing import Pool
from scipy.optimize import minimize
APT=SC.APT; OUT=os.path.expanduser('~/Projetos/prc_data_challenge/work/ablation/b')
T=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','ADEP_mvt','PHASE_mvt']); T=T[(T.ADEP_mvt==APT)&(T.PHASE_mvt=='DEP')]
lo,hi=T.TAXITIME_SEC_mvt.quantile([0.05,0.95]); Y=dict(zip(T.MVT_ID_mvt,T.TAXITIME_SEC_mvt))
LOG=['X','tau','theta','p','q','wads']; LB=dict(X=(300,6000),tau=(20,900),theta=(0.05,60),p=(0.5,4),q=(0.5,3),wads=(0.05,5))
def mk(v): return dict(rev=True,stats=True,gm={},em={},**{k:float(np.clip(np.exp(x),*LB[k])) for k,x in zip(LOG,v)})
def avalia(pr,days):
    F.PR=pr; return pd.concat([F.dia(d) for d in days],ignore_index=True)
def _ev(a): pr,d=a; F.PR=pr; return F.dia(d)
def rmse(pr,days,pool):
    r=pd.concat([x for x in pool.map(_ev,[(pr,d) for d in days]) if len(x)],ignore_index=True); r['y']=r.MVT_ID_mvt.map(Y); r=r.dropna(subset=['y','exot_graf']); r=r[(r.y>=lo)&(r.y<=hi)]
    A=np.c_[np.ones(len(r)),r.exot_graf.to_numpy()]; c=np.linalg.lstsq(A,r.y.to_numpy(),rcond=None)[0]; return float(np.sqrt(np.mean((r.y-A@c)**2))),c
if __name__=='__main__':
    SC.graf(); allp=sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SC.R0}/20*.pkl')); _m=os.environ.get('MONTHS'); allp=[x for x in allp if (not _m) or x[5:7] in _m.split(',')]; pick=[allp[i] for i in np.linspace(5,len(allp)-6,9).astype(int)]; tr,te=pick[0::3]+pick[1::3],pick[2::3]
    base=json.load(open(f'{OUT}/params_base.json')) if os.path.exists(f'{OUT}/params_base.json') else dict(X=3995.,theta=10.55,tau=318.9,wads=.3026,p=1.7576,q=1.4117)
    x0=np.log([base[k] for k in LOG]); t0=time.time()
    with Pool(int(os.environ.get('NP','6'))) as pool:
        b0=rmse(mk(x0),tr,pool)[0]; rng=np.random.default_rng(0); best=(b0,x0)
        for _ in range(int(os.environ.get('NR','40'))):
            v=np.log([np.exp(np.random.default_rng(rng.integers(1e9)).uniform(np.log(LB[k][0]),np.log(LB[k][1]))) for k in LOG]); s=rmse(mk(v),tr,pool)[0]
            if s<best[0]: best=(s,v)
        res=minimize(lambda v:rmse(mk(v),tr,pool)[0],best[1],method='Nelder-Mead',options=dict(maxiter=int(os.environ.get('NM','40')),xatol=0.05,fatol=0.05))
        v1=res.x if res.fun<best[0] else best[1]; pr1=mk(v1); r_tr=rmse(pr1,tr,pool)[0]
        # teste: reajusta so a OLS no treino e avalia no teste com os mesmos parametros
        r_te_base=rmse(mk(x0),te,pool)[0]; r_te=rmse(pr1,te,pool)[0]
        # referencia sem rastros: dur_livre como unico preditor (theta enorme => f=1)
        pr0=dict(mk(x0)); pr0['theta']=1e9; r_te_0=rmse(pr0,te,pool)[0]
    out=dict(apt=APT,params={k:pr1[k] for k in LOG},rmse_treino_base=b0,rmse_treino_cal=r_tr,rmse_teste_base=r_te_base,rmse_teste_cal=r_te,rmse_teste_sem_rastro=r_te_0,dias_treino=tr,dias_teste=te,seg=time.time()-t0)
    json.dump(out,open(f'{OUT}/params_{APT}{os.environ.get("SUFX","")}.json','w'),indent=1); print(APT,'teste: sem rastro %.1f | v0 %.1f | calibrado %.1f | treino %.1f->%.1f | %.0fs'%(r_te_0,r_te_base,r_te,b0,r_tr,out['seg']),flush=True)
