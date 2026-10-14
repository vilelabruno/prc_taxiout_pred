"""SOLVER de 1 dia: otimiza o RMSE (validacao cruzada em blocos de tempo) em funcao de (a) parametros dos rastros: X, gamma, tau, peso do ADS-B (globais, Nelder-Mead) e multiplicadores
de sensibilidade/emissao por tipo de aviao (5), hora, clima (3) e stand (pier) (busca por coordenadas, com penalidade); (b) espera no stand/pushback: efeitos aditivos por contexto resolvidos em forma fechada (ridge) dentro de cada avaliacao."""
import sys, os, re, pickle, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import sim_campo as SC
DAY=sys.argv[1]; OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
# ---------- tipo de aviao (grosseiro): cargo, comercial, lowcost, jatinho, rural
CARGO={'FDX','UPS','DHL','DHK','BOX','GTI','CLX','MPH','TAY','ABW','CKS','CAO','SQC','PAC','NCA','CTJ','ICL','BCS','ASL'}
LOWC={'RYR','EZY','EJU','EZS','WZZ','VLG','TRA','TFL','TVF','NOZ','EWG','EXS','PGT','SXS','NSZ','CND','VOE','JAF','TOM','WMT','FFT','HV'+'Y'}
JET_P={'NJE','EJM','LXJ','VJT','CFE_'}; JET_T=re.compile(r'^(C25|C5|C6|C7|GLF|GL[0-9]|LJ|FA[0-9]|CL[0-9]|E55|E50|H25|PC24|F2T|F900|EA50|SF50|PRM1|GALX|BE40|C680|C68)')
RUR_T=re.compile(r'^(DH8|AT4|AT7|ATR|SF34|SB20|F50|E120|B190|JS|SH36|D228|DHC|BE20|BE9|PC12|A139|EC[0-9]|AS3|H[0-9]{3}|B06|R44|S76)')
def tipo(cs,ty,ft):
    p=(cs or '')[:3].upper(); ty=(ty or '').upper()
    if p in CARGO or ty in ('B77L','MD11') or (ty.endswith('F') and len(ty)==4): return 'cargo'
    if p in JET_P or ft=='G' or JET_T.match(ty): return 'jatinho'
    if RUR_T.match(ty): return 'rural'
    if p in LOWC: return 'lowcost'
    return 'comercial'
# ---------- clima: score global (chuva + vento + neve) em 3 niveis
def clima(mvt_s):
    m=pd.read_parquet('data/processed/metar_events_2025.parquet'); m=m[m.ADEP_mvt==os.environ.get('APT','EGLL')].copy(); m['t']=pd.to_datetime(m.metar_time_utc,utc=True).dt.as_unit('s').astype('int64'); m=m.sort_values('t')
    w=np.nan_to_num(np.maximum(m.sknt.astype(float),m.gust.astype(float)),nan=0); wx=m.wxcodes.fillna('').astype(str)
    rain=np.where((m.metar_heavy_rain==1)|(m.metar_thunderstorm==1),1.0,np.where(wx.str.contains('RA|DZ|SH'),0.5,0.0)); wind=np.where(w>=25,1.0,np.where(w>=15,0.5,0.0)); snow=np.where((m.metar_snow==1)|(m.metar_freezing_precip==1)|(m.metar_hail==1),1.0,0.0)
    sc=rain+wind+snow; m['nivel']=np.where(sc<0.5,0,np.where(sc<1.5,1,2)); m['score']=sc
    x=pd.merge_asof(pd.DataFrame({'t':np.sort(mvt_s)}),m[['t','nivel','score']],on='t',direction='backward'); mp=dict(zip(x.t,x.nivel.fillna(0).astype(int))); return mp
# ---------- tabela do dia
Dia=pickle.load(open(f'{SC.R0}/{DAY}.pkl','rb')); rows=[]
for i,r in enumerate(Dia['D']):
    if not r['ok']: continue
    tend=r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-SC.ROLL_DEP
    rows.append(dict(i=i,id=r['id'],cs=r['cs'],mvt=r['mvt'],blk=r['blk'],y=r['mvt']-r['blk'],chain=int(bool(r['chain'])),base=r['mvt']-tend+r['t_patio'],stand=str(r['stand'])))
T=pd.DataFrame(rows)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_TYPE_flt']).rename(columns={'MVT_ID_mvt':'id'}); T=T.merge(tr,on='id',how='left')
T['tipo']=[tipo(a,b,c) for a,b,c in zip(T.cs,T.AIRCRAFT_TYPE_mvt,T.FLIGHT_TYPE_flt)]; T['hb']=((pd.to_datetime(T.mvt,unit='s').dt.hour)//6).astype(int); T['pier']=T.stand.str[0]; mp=clima(T.mvt.to_numpy()); T['clima']=T.mvt.map(mp).fillna(0).astype(int)
CUT=float(os.environ.get('CUT','1e9')); CLO=float(os.environ.get('CUT_LO','0')); n0=len(T); T=T[(T.y<=CUT)&(T.y>=CLO)]; print('CORTE DE CAUDA (pelo rotulo): removidos %d de %d voos com y < %.0f s ou y > %.0f s'%(n0-len(T),n0,CLO,CUT))
T=T.sort_values('mvt').reset_index(drop=True); print('DIA',DAY,'| partidas',len(T),'| com cadeia ADS',int(T.chain.sum())); print('tipo:',T.tipo.value_counts().to_dict(),'| clima (nivel):',T.clima.value_counts().to_dict(),'| hora (blocos de 6 h):',T.hb.value_counts().sort_index().to_dict(),'| piers:',T.pier.value_counts().head(8).to_dict(),flush=True)
# ---------- espera no stand/pushback: efeitos aditivos por contexto em forma fechada (ridge) + validacao cruzada em blocos de tempo
FACT=['tipo','hb','clima','pier']; folds=np.array_split(np.arange(len(T)),5)
def onehot(tr_,te_):
    A=[];B=[]
    for k in FACT:
        cats=pd.Categorical(tr_[k]); ix={c:i for i,c in enumerate(cats.categories)}
        def oh(df):
            ii=df[k].map(ix); m=ii.notna().to_numpy(); r=np.where(m)[0]; return sparse.csr_matrix((np.ones(m.sum()),(r,ii[m].astype(int).to_numpy())),shape=(len(df),len(ix)))
        A.append(oh(tr_)); B.append(oh(te_))
    A.append(sparse.csr_matrix(tr_[['chain']].to_numpy(float))); B.append(sparse.csr_matrix(te_[['chain']].to_numpy(float)))
    return sparse.hstack(A).tocsr(),sparse.hstack(B).tocsr()
def cv_rmse(dur,alpha,usar_dur=True,so_const=False):
    x=T.copy(); x['dur']=dur if usar_dur else 0.0; x['res']=x.y-x.base-x.dur; pred=np.zeros(len(x))
    for f in folds:
        te=np.zeros(len(x),bool); te[f]=True; trn=~te
        if so_const: pred[te]=x.res[trn].mean()
        else: Xtr,Xte=onehot(x[trn],x[te]); pred[te]=Ridge(alpha=alpha).fit(Xtr,x.res[trn]).predict(Xte)
    return rm(x.res-pred),pred
def simula(P,gm,em):
    pr=dict(X=P['X'],gamma=0.0,tau=P['tau'],wads=P['wads'],theta=P['theta'],p=P['p'],q=P['q'],rev=True,gm=gm,em=em); R=SC.simula_dia((DAY,pr)); return T[['i']].merge(R[['i','dur']],on='i',how='left').dur.to_numpy()
# ---------- BASELINES
t0=time.time(); b_const=rm(np.concatenate([T.y.to_numpy()[f]-T.y.to_numpy()[np.setdiff1d(np.arange(len(T)),f)].mean() for f in folds]))
dur0=simula(dict(X=1000.,tau=100.,wads=1.,theta=1e9,p=2.0,q=1.0),{},{}); ok=np.isfinite(dur0); print('partidas sem duracao simulada (sem caminho): %d (retiradas)'%int((~ok).sum()))
T=T[ok].reset_index(drop=True); folds=np.array_split(np.arange(len(T)),5); b_const=rm(np.concatenate([T.y.to_numpy()[f]-T.y.to_numpy()[np.setdiff1d(np.arange(len(T)),f)].mean() for f in folds])); dur0=simula(dict(X=1000.,tau=100.,wads=1.,theta=1e9,p=2.0,q=1.0),{},{}); assert np.isfinite(dur0).all()
b_base,_=cv_rmse(dur0,1.0,usar_dur=False,so_const=True); b_sim0,_=cv_rmse(dur0,1.0,so_const=True)
alphas=(0.3,1,3,10,30,100,300); r_a=[cv_rmse(dur0,a)[0] for a in alphas]; AL=alphas[int(np.argmin(r_a))]; b_ctx0=min(r_a)
oc=pd.read_parquet(os.path.expanduser('~/Projetos/oof_store/grafo_por_aeroporto/EGLL/cenario_b_oof.parquet'),columns=['id','champ']); m=T[['id','y']].merge(oc,on='id',how='inner'); m['clean_v53']=m.champ; m['blend_v53_old']=m.champ
print('\nBASELINES (RMSE de y em s, validacao cruzada em 5 blocos de tempo do dia):')
print('  constante (media de y)                                   %.1f'%b_const); print('  y = base + constante   (base = MVT - ancora + patio)    %.1f'%b_base)
print('  + duracao simulada SEM rastros (gamma=0), constante       %.1f'%b_sim0); print('  + espera no stand por contexto (ridge, alpha=%s), sem rastros  %.1f'%(AL,b_ctx0))
print('  CAMPEAO de Londres (OOF cenario_b, %d voos; nao e CV deste dia) %.1f'%(len(m),rm(m.y-m.clean_v53)),flush=True)
# ---------- SOLVER, etapa 1: parametros globais dos rastros (Nelder-Mead, log)
def objetivo_global(th):
    P=dict(X=float(np.exp(th[0])),theta=float(np.exp(th[1])),tau=float(np.exp(th[2])),wads=float(np.exp(th[3])),p=float(np.exp(th[4])),q=float(np.exp(th[5])))
    if not (200<=P['X']<=4000 and 0.05<=P['theta']<=200 and 10<=P['tau']<=1500 and 0.1<=P['wads']<=30 and 0.5<=P['p']<=6 and 0.5<=P['q']<=4): return 1e3
    d=simula(P,{},{}); return cv_rmse(d,AL)[0] if np.isfinite(d).all() else 1e3
rng=np.random.default_rng(0); t1=time.time(); bestr=(1e9,None); rs=[]
for _ in range(250):
    th=np.log([rng.uniform(300,3000),np.exp(rng.uniform(np.log(1),np.log(300))),rng.uniform(20,600),np.exp(rng.uniform(np.log(0.5),np.log(20))),rng.uniform(0.7,4),rng.uniform(0.7,3)]); o=objetivo_global(th); rs.append(o)
    if o<bestr[0]: bestr=(o,th)
print('BUSCA ALEATORIA (250 pontos, %.0f s): melhor RMSE CV %.1f | quantis do RMSE entre os pontos 10/50/90: %s'%(time.time()-t1,bestr[0],np.quantile(rs,[.1,.5,.9]).round(0).tolist()),flush=True); th0=bestr[1]; hist=[]
def cb(x): hist.append(objetivo_global(x))
sol=minimize(objetivo_global,th0,method='Nelder-Mead',options=dict(maxiter=80,xatol=0.03,fatol=0.05,initial_simplex=np.vstack([th0]+[th0+np.eye(6)[k]*0.7 for k in range(6)])))
PG=dict(X=float(np.exp(sol.x[0])),theta=float(np.exp(sol.x[1])),tau=float(np.exp(sol.x[2])),wads=float(np.exp(sol.x[3])),p=float(np.exp(sol.x[4])),q=float(np.exp(sol.x[5]))); print('\nETAPA 1 (globais, %d avaliacoes, %.0f s): X=%.2f km | theta=%.2f | p=%.2f | q=%.2f | tau=%.0f s | peso ADS-B=%.2f -> RMSE CV %.1f (sem rastros: %.1f)'%(sol.nfev,time.time()-t1,PG['X']/1000,PG['theta'],PG['p'],PG['q'],PG['tau'],PG['wads'],sol.fun,b_ctx0),flush=True)
# ---------- etapa 2: multiplicadores por contexto (sensibilidade ao rastro e emissao), busca por coordenadas, com penalidade L2 em log
GRID=(0.5,2.0); LAM=2.0
cells=[('gm','tipo',v) for v in sorted(T.tipo.unique())]+[('em','tipo',v) for v in sorted(T.tipo.unique())]+[('gm','hb',v) for v in sorted(T.hb.unique())]+[('gm','clima',v) for v in sorted(T.clima.unique())]+[('gm','pier',v) for v in sorted(T.pier.unique())]
M={c:1.0 for c in cells}
def mult(M):
    gm=np.ones(len(T)); em=np.ones(len(T))
    for (k,f,v),m_ in M.items():
        sel=(T[f]==v).to_numpy()
        if k=='gm': gm[sel]*=m_
        else: em[sel]*=m_
    return {int(i):float(g) for i,g in zip(T.i,gm)},{int(i):float(e) for i,e in zip(T.i,em)}
def objetivo(M):
    gm,em=mult(M); d=simula(PG,gm,em); pen=LAM*sum(np.log(v)**2 for v in M.values()); return cv_rmse(d,AL)[0]+pen,cv_rmse(d,AL)[0]
best,_=objetivo(M); t2=time.time(); ncell=0
for sweep in range(1):
    for c in cells:
        for g_ in GRID:
            if g_==M[c]: continue
            tr_=dict(M); tr_[c]=g_; o,_=objetivo(tr_); ncell+=1
            if o<best-0.2: best=o; M=tr_
    print('  varredura %d: objetivo (RMSE CV + penalidade) %.2f'%(sweep+1,best),flush=True)
gm,em=mult(M); dfin=simula(PG,gm,em); r_cv,pred=cv_rmse(dfin,AL); print('ETAPA 2 (%d avaliacoes, %.0f s): multiplicadores diferentes de 1:'%(ncell,time.time()-t2)); 
for c,v in M.items():
    if v!=1.0: print('   %s %s=%s -> x%.2f'%(c[0],c[1],c[2],v))
# resultado final, dentro e fora da amostra
x=T.copy(); x['dur']=dfin; x['res']=x.y-x.base-x.dur; Xa,_=onehot(x,x); in_s=rm(x.res-Ridge(alpha=AL).fit(Xa,x.res).predict(Xa))
print('\n================ RESULTADO NO DIA %s (RMSE de y, s) ================'%DAY)
print('  campeao (OOF)                      %.1f'%rm(m.y-m.clean_v53)); print('  constante                          %.1f'%b_const); print('  base + constante                   %.1f'%b_base); print('  simulacao sem rastros + espera     %.1f   (CV)'%b_ctx0); print('  SOLVER (rastros + espera)          %.1f   (CV)  | %.1f dentro da amostra'%(r_cv,in_s))
T2=T.copy(); T2['pred']=T2.y-(pred-0)*0; print('  tipo de aviao: RMSE CV do solver por tipo:',{k:round(rm((x.res-pred)[x.tipo==k]),1) for k in sorted(x.tipo.unique())})
pickle.dump(dict(PG=PG,M=M,T=T,dur=dfin,AL=AL),open(f'{OUT}/solver_rev_{DAY}.pkl','wb'))
