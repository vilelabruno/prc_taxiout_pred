import sys, os, glob, pickle, itertools, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import sim_campo as SC
R0=SC.R0; OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo')
def tabela():
    p=f'{OUT}/voos_EHAM.parquet'
    if os.path.exists(p): return pd.read_parquet(p)
    rows=[]
    for f in sorted(glob.glob(f'{R0}/20*.pkl')):
        Dia=pickle.load(open(f,'rb')); day=Dia['day']
        for i,r in enumerate(Dia['D']):
            if not r['ok']: continue
            tend=r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-SC.ROLL_DEP
            rows.append(dict(day=day,i=i,id=r['id'],cs=r['cs'],mvt=r['mvt'],blk=r['blk'],y=r['mvt']-r['blk'],chain=bool(r['chain']),tend=tend,t_patio=r['t_patio'],stand=r['stand'],rwy=r['rwy'],obs=r['obs'] is not None))
    T=pd.DataFrame(rows); T['base']=T.mvt-T.tend+T.t_patio; T.to_parquet(p); return T
def objetivo(T,R):
    x=T.merge(R[['day','i','dur']],on=['day','i'],how='left'); x=x[np.isfinite(x.dur)]; r=x.y-x.base-x.dur; e=r-r.groupby(x.chain).transform('median'); return float(np.sqrt((e**2).mean())),len(x)
if __name__=='__main__':
    T=tabela(); days=sorted(T.day.unique()); print('voos',len(T),'| com cadeia ADS',int(T.chain.sum()),'| dias',len(days),flush=True)
    pool=Pool(12)
    def roda(pr):
        R=pd.concat(pool.map(SC.simula_dia,[(d,pr) for d in days])); return objetivo(T,R)
    t0=time.time(); b0,n=roda(dict(X=1000.,gamma=0.0,tau=100.,wads=1.)); print('SEM rastros (gamma=0): RMSE do residuo (constante por classe) %.1f s | n=%d | %.0f s'%(b0,n,time.time()-t0),flush=True)
    res=[]
    for X,g_,tau,w in itertools.product((500.,1000.,2000.),(0.03,0.1,0.3),(60.,200.),(1.,3.)):
        r,n=roda(dict(X=X,gamma=g_,tau=tau,wads=w)); res.append(dict(X=X/1000,gamma=g_,tau=tau,wads=w,rmse=r,ganho=r-b0)); print('  X=%.1f km gamma=%.2f tau=%3.0f s peso ADS=%.0f: RMSE %.1f (ganho %+.1f s)'%(X/1000,g_,tau,w,r,r-b0),flush=True)
    R=pd.DataFrame(res).sort_values('rmse'); R.to_csv(f'{OUT}/calib_global.csv',index=False); print('\nMELHORES:'); print(R.head(6).round(2).to_string(index=False)); print('\nefeito medio no RMSE por parametro:'); 
    for c in ('X','gamma','tau','wads'): print(' ',c,R.groupby(c).ganho.mean().round(2).to_dict())
