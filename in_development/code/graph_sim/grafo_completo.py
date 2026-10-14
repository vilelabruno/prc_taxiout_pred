"""GRAFO COMPLETO, tudo ligado: ADS-B regredido desde o MVT (ou ancora na pista) + Dijkstra com velocidade media + simulacao de patio
(taxiway -> frente do stand -> recuo pelo eixo) + dinamica (pushback/espera) como posterior por contexto (stand, dia, hora, companhia, tipo, pista)."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import taxigraph as T
from sincro import Sync
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store'); D_PB=70.0; V_PATIO=2.0
d=pd.read_parquet(f'{S}/sincronia_grafo/grafo_base_{APT}.parquet')
ctx=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','AIRCRAFT_TYPE_mvt']).rename(columns={'MVT_ID_mvt':'id'})
d=d.merge(ctx,on='id',how='left'); t=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True)
d['hora']=t.dt.hour.astype(str); d['dow']=t.dt.dayofweek.astype(str); d['stand']=d.STAND_mvt.astype(str); d['oper']=d.AIRCRAFT_OPERATOR_flt.astype(str).str[:8]; d['acft']=d.AIRCRAFT_TYPE_mvt.astype(str); d['rwy']=d.rwy.astype(str)
# ---- simulacao de patio: geometria por stand (taxiway -> frente do stand F -> recuo reto pelo eixo)
sy=Sync(APT); g=sy.g; idx=T.build_stand_index(g.A); SP=np.array([g.proj(x['lat'],x['lon']) for x in g.A.stands])
def acesso(F):
    best=np.inf
    for k in range(len(sy.ea)):
        if sy.kind[k]!='taxi': continue
        A_,B_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; ab=B_-A_; den=ab@ab; f=0. if den==0 else float(np.clip((F-A_)@ab/den,0,1)); best=min(best,float(np.hypot(*(F-(A_+f*ab)))))
    return best
patio={}
for s in d.stand.dropna().unique():
    la,lo,lv=T.map_stand(s,idx)
    if lv>=9: continue
    sp=np.array(g.proj(la,lo)); j=int(np.argmin(np.hypot(*(SP-sp).T))); h=np.radians(90-g.A.stands[j]['hdg']); u=np.array([np.cos(h),np.sin(h)]); F=sp-D_PB*u
    patio[s]=(acesso(F)+D_PB)/V_PATIO                       # tempo simulado de patio: acesso da taxiway a frente do stand + recuo pelo eixo
d['t_patio']=d.stand.map(patio)
v_h=d.v_h; v_fl=d.v_bar.fillna(v_h).clip(1.5,12)
d['mov_dij']=np.where(d.tem==1,d.D_star/v_fl,d.L/v_h)       # reserva: Dijkstra / velocidade media do aviao
# ---- trecho escondido EMPIRICO: caminho observado por ADS-B dos voos do mesmo stand, do ultimo ponto ADS (ou da pista) ate a vizinhanca do stand
ref=pd.read_parquet(f'{S}/sincronia_grafo/trilhas_ref_{APT}.parquet'); ref['dn']=(pd.to_datetime(ref.day)-pd.Timestamp('2025-01-01')).dt.days
pp=pd.read_parquet(f'{S}/sincronia_grafo/ponto_parada_{APT}.parquet'); d=d.merge(pp,on='id',how='left')
Rt=T.Router(g); anc={}
for (s,r_) in set(zip(d.stand,d.rwy)):
    la,lo,lv=T.map_stand(s,idx)
    try: o_=Rt.route(la,lo,r_); anc[(s,r_)]=g.xy[o_['path'][-1]]
    except Exception: anc[(s,r_)]=None
byst={k:(v.x.to_numpy(),v.y.to_numpy(),v.tau.to_numpy(),v.id.to_numpy(),v.dn.to_numpy()) for k,v in ref.groupby('stand')}
d['dn']=(pd.to_datetime(d.day)-pd.Timestamp('2025-01-01')).dt.days
tau=np.full(len(d),np.nan); nref=np.zeros(len(d))
for j,(s,tem,dn,px,py,r_) in enumerate(zip(d.stand,d.tem,d.dn,d.px,d.py,d.rwy)):
    if s not in byst: continue
    if tem==1:
        if not np.isfinite(px): continue
        A_=np.array([px,py]); R_=60.0
    else:
        A_=anc.get((s,r_)); R_=150.0
        if A_ is None: continue
    x,y,t_,ids,dns=byst[s]; m=((x-A_[0])**2+(y-A_[1])**2<=R_**2)&(np.abs(dns-dn)>1)
    if m.sum()<2: continue
    dd=(x[m]-A_[0])**2+(y[m]-A_[1])**2; o_=np.lexsort((dd,ids[m])); idm=ids[m][o_]; first=np.unique(idm,return_index=True)[1]
    if len(first)<2: continue
    tau[j]=np.median(t_[m][o_][first]); nref[j]=len(first)
d['tau_emp']=tau; d['n_ref']=nref
d['mov_hid']=np.where(np.isfinite(d.tau_emp),d.tau_emp,d.mov_dij)
d['ref']=np.where(d.tem==1,d.t_star,d.mvt_s)                 # ancora: ultimo ponto ADS, ou o MVT na pista
d['W']=d.ref-d.blk_rotulo-d.mov_hid-d.t_patio                # dinamica (pushback+espera) = o que sobra; alvo da posterior
ok=d.t_patio.notna()&d.L.notna()&np.isfinite(d.mov_hid); d=d[ok].reset_index(drop=True)
d['d']=pd.to_datetime(d.day); days=np.array(sorted(d.d.unique())); d['blk']=d.d.map({x:i//5 for i,x in enumerate(days)})
FACT={'stand':0.1,'hora':0.4,'oper':0.1,'acft':0.2,'rwy':0.05,'dow':0.0}; NUM=['mov_hid','D_star','L','v_h','n_usados','esp_visto']
for c in NUM: d[c]=d[c].astype(float)
def design(tr,te,num):
    A=[];B=[]
    for k,s in FACT.items():
        if s<=0: continue
        cats=pd.Categorical(tr[k]); ix={c:i for i,c in enumerate(cats.categories)}
        def oh(df):
            ii=df[k].map(ix); m=ii.notna().to_numpy(); r=np.where(m)[0]; return sparse.csr_matrix((np.ones(m.sum())*s,(r,ii[m].astype(int).to_numpy())),shape=(len(df),len(ix)))
        A.append(oh(tr)); B.append(oh(te))
    mu=tr[num].mean(); sd=tr[num].std().replace(0,1); A.append(sparse.csr_matrix(((tr[num]-mu)/sd).fillna(0).to_numpy())); B.append(sparse.csr_matrix(((te[num]-mu)/sd).fillna(0).to_numpy()))
    return sparse.hstack(A).tocsr(),sparse.hstack(B).tocsr()
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
what=np.zeros(len(d))
for cls,num in ((1,['mov_hid','D_star','v_h','n_usados','esp_visto']),(0,['mov_hid','L','v_h'])):
    m_cls=(d.tem==cls).to_numpy()
    for b in d.blk.unique():
        te=(d.blk==b).to_numpy()&m_cls; lo=d.d[d.blk==b].min()-pd.Timedelta(days=1); hi=d.d[d.blk==b].max()+pd.Timedelta(days=1); trn=m_cls&((d.d<lo)|(d.d>hi)).to_numpy()
        if te.sum()==0: continue
        Xtr,Xte=design(d[trn],d[te],num); what[te]=Ridge(alpha=1.0).fit(Xtr,d.W[trn]).predict(Xte)
d['What']=what; d['blk_hat']=d.ref-d.mov_hid-d.t_patio-d.What; d['yh']=d.mvt_s-d.blk_hat
print('\n=========== EGLL jan+jul 2025, TODAS as partidas (n=%d), tudo ligado ==========='%len(d))
print('RMSE CAMPEAO: %.1f s'%rm(d.y-d.champ)); print('RMSE GRAFO:   %.1f s'%rm(d.y-d.yh))
for m,z in d.groupby('mes'): print('   %s  campeao %.1f | grafo %.1f  (n=%d)'%(m,rm(z.y-z.champ),rm(z.y-z.yh),len(z)))
print('   com pings na janela (n=%d): campeao %.1f | grafo %.1f ;  sem pings (n=%d): campeao %.1f | grafo %.1f'%((d.tem==1).sum(),rm(d.y[d.tem==1]-d.champ[d.tem==1]),rm(d.y[d.tem==1]-d.yh[d.tem==1]),(d.tem==0).sum(),rm(d.y[d.tem==0]-d.champ[d.tem==0]),rm(d.y[d.tem==0]-d.yh[d.tem==0])))
print('   trecho escondido pelo caminho observado (ADS-B de outros voos do stand): %.1f%% das partidas (com pings %.1f%%, sem pings %.1f%%); o resto cai no Dijkstra/velocidade media'%(100*d.tau_emp.notna().mean(),100*d.tau_emp[d.tem==1].notna().mean(),100*d.tau_emp[d.tem==0].notna().mean()))
print('   (referencia sem a posterior de contexto, so constante no lugar: grafo %.1f)'%rm(d.y-(d.mvt_s-(d.ref-d.mov_hid-d.t_patio-d.groupby('tem').W.transform('median')))))
d[['id','day','mes','y','champ','yh','tem']].to_parquet(f'{S}/sincronia_grafo/grafo_completo_{APT}.parquet')
