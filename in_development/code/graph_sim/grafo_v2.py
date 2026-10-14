"""GRAFO V2, tudo ligado: so taxiway (nunca pista) + sentido das arestas + desconto de velocidade por congestionamento observado (ADS-B) ao longo do caminho
+ ADS-B regredido desde o MVT (ou ancora na pista) + trecho escondido pelo caminho observado de outros voos do stand (quando ha) + patio (frente do stand, recuo pelo eixo)
+ posterior de espera por contexto + RELOGIOS como marcacoes de atraso (se o grafo chega rapido demais, forca mais lento: o tempo nao pode ser menor que o do relogio)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import taxigraph as T
from sincro2 import Sync2
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store'); D_PB=70.0; V_PATIO=2.0
d=pd.read_parquet(f'{S}/sincronia_grafo/grafo_base_{APT}.parquet')
ctx=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','AIRCRAFT_TYPE_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt']).rename(columns={'MVT_ID_mvt':'id'})
pp=pd.read_parquet(f'{S}/sincronia_grafo/ponto_parada_{APT}.parquet')
d=d.merge(ctx,on='id',how='left').merge(pp,on='id',how='left'); mv=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True)
d['hora']=mv.dt.hour.astype(str); d['dow']=mv.dt.dayofweek.astype(str); d['stand']=d.STAND_mvt.astype(str); d['oper']=d.AIRCRAFT_OPERATOR_flt.astype(str).str[:8]; d['acft']=d.AIRCRAFT_TYPE_mvt.astype(str); d['rwy']=d.rwy.astype(str)
for nm,c in (('sched','SCHED_TIME_UTC_mvt'),('aobt','AOBT_3_flt'),('eobt','EOBT_1_flt')): d['m_'+nm]=(mv-pd.to_datetime(d[c],utc=True,errors='coerce')).dt.total_seconds()    # MVT - relogio
sy=Sync2(APT); g=sy.g; Rt=T.Router(g); idx=T.build_stand_index(g.A); SPp=np.array([g.proj(x['lat'],x['lon']) for x in g.A.stands])
# ---- congestionamento observado
cong={}
for p in glob.glob(f'{S}/sincronia_grafo/congestao/{APT}_*.parquet'):
    a=pd.read_parquet(p); day=os.path.basename(p)[len(APT)+1:-8]; cong[day]={k:set(v) for k,v in a.groupby(['bin','k']).icao24}
# ---- patio por stand (taxiway -> frente do stand F -> recuo pelo eixo)
def acesso_dist(F):
    best=np.inf
    for k in range(len(sy.ea)):
        A_,B_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; ab=B_-A_; den=ab@ab; f=0. if den==0 else float(np.clip((F-A_)@ab/den,0,1)); best=min(best,float(np.hypot(*(F-(A_+f*ab)))))
    return best
info={}
for s_ in d.stand.dropna().unique():
    la,lo,lv=T.map_stand(s_,idx)
    if lv>=9: continue
    sp=np.array(g.proj(la,lo)); j=int(np.argmin(np.hypot(*(SPp-sp).T))); h=np.radians(90-g.A.stands[j]['hdg']); u=np.array([np.cos(h),np.sin(h)]); F=sp-D_PB*u
    opts,_=Rt.attach(la,lo); info[s_]=dict(t_patio=(acesso_dist(F)+D_PB)/V_PATIO,opts=opts)
anc={}
for (s_,r_) in set(zip(d.stand,d.rwy)):
    try:
        pair,cands,rl=g.targets(r_); anc[(s_,r_)]=min(cands,key=lambda c:c[1])[0] if cands else None
    except Exception: anc[(s_,r_)]=None
def caminho_no(opts,node):
    """stand -> no (anchor): distancia direcionada, sem pista; se inalcancavel, sem sentido. devolve (dist, [(aresta, comprimento usado)])"""
    best=(np.inf,None,None,True)
    for nd,acc in opts:
        c=acc+sy.D[nd,node]
        if c<best[0]: best=(c,nd,node,True)
    if not np.isfinite(best[0]):
        for nd,acc in opts:
            c=acc+sy.Du[nd,node]
            if c<best[0]: best=(c,nd,node,False)
    if not np.isfinite(best[0]): return np.nan,[]
    c,nd,via,dirc=best; P_=sy.P if dirc else sy.Pu; cur=via; path=[cur]
    while cur!=nd and P_[nd,cur]>=0: cur=int(P_[nd,cur]); path.append(cur)
    path=path[::-1]; return c,[(sy.ek[(a,b)],sy.L[sy.ek[(a,b)]]) for a,b in zip(path[:-1],path[1:]) if (a,b) in sy.ek]
def caminho_ponto(opts,k,f):
    c,nd,via=sy.para_ponto(opts,k,f); dirc=True
    if not np.isfinite(c):
        best=(np.inf,None,None)
        for nd_,acc in opts:
            for end,off in ((sy.ea[k],f*sy.L[k]),(sy.eb[k],(1-f)*sy.L[k])):
                cc=acc+sy.Du[nd_,end]+off
                if cc<best[0]: best=(cc,nd_,end)
        c,nd,via=best; dirc=False
        if not np.isfinite(c): return np.nan,[]
    P_=sy.P if dirc else sy.Pu; cur=via; path=[cur]
    while cur!=nd and P_[nd,cur]>=0: cur=int(P_[nd,cur]); path.append(cur)
    path=path[::-1]; ed=[(sy.ek[(a,b)],sy.L[sy.ek[(a,b)]]) for a,b in zip(path[:-1],path[1:]) if (a,b) in sy.ek]
    part=f*sy.L[k] if via==sy.ea[k] else (1-f)*sy.L[k]; ed.append((k,part)); return c,ed
# ---- rota nova por voo (so taxiway, direcionada)
cache_n={}; Dn=np.full(len(d),np.nan); ED=[None]*len(d); P_xy=d[['px','py']].to_numpy()
kk=np.full(len(d),-1); ff=np.zeros(len(d)); ch=np.where((d.tem==1)&np.isfinite(P_xy[:,0]))[0]
if len(ch): k_,f_=sy.snap_many(P_xy[ch]); kk[ch]=k_; ff[ch]=f_
for j,(s_,r_,tem) in enumerate(zip(d.stand,d.rwy,d.tem)):
    if s_ not in info: continue
    opts=info[s_]['opts']
    if tem==1 and kk[j]>=0: c,ed=caminho_ponto(opts,kk[j],ff[j])
    elif tem==1: c,ed=d.D_star.iloc[j],[]                      # ponto de parada no patio (fora do grafo): distancia antiga (euclidiana)
    else:
        key=(s_,r_)
        if key not in cache_n: cache_n[key]=caminho_no(opts,anc[key]) if anc.get(key) is not None else (np.nan,[])
        c,ed=cache_n[key]
    Dn[j]=c; ED[j]=ed
d['D_new']=Dn; d['t_patio']=d.stand.map({k:v['t_patio'] for k,v in info.items()})
v_h=d.v_h; v_fl=d.v_bar.fillna(v_h).clip(1.5,12); d['v']=np.where(d.tem==1,v_fl,v_h); d['ref']=np.where(d.tem==1,d.t_star,d.mvt_s)
# ---- congestionamento ao longo do caminho (avioes OUTROS na aresta, na faixa de 10 min do meio do trecho)
cg=np.zeros(len(d))
for j in range(len(d)):
    ed=ED[j]
    if not ed or not np.isfinite(Dn[j]): continue
    mid=d.ref.iloc[j]-0.5*Dn[j]/d.v.iloc[j]; b=int(mid//600); tab=cong.get(d.day.iloc[j]); own=d.icao.iloc[j] if 'icao' in d and isinstance(d.icao.iloc[j],str) else None
    if tab is None: continue
    tot=0.;wl=0.
    for k,Lk in ed:
        s_=tab.get((b,k)); n_=(len(s_)-(1 if (own is not None and s_ and own in s_) else 0)) if s_ else 0; tot+=Lk*n_; wl+=Lk
    cg[j]=tot/wl if wl>0 else 0.
d['cg']=cg
# ---- trecho escondido empirico (caminho observado por ADS-B de outros voos do stand)
ref=pd.read_parquet(f'{S}/sincronia_grafo/trilhas_ref_{APT}.parquet'); ref['dn']=(pd.to_datetime(ref.day)-pd.Timestamp('2025-01-01')).dt.days
byst={k:(v.x.to_numpy(),v.y.to_numpy(),v.tau.to_numpy(),v.id.to_numpy(),v.dn.to_numpy()) for k,v in ref.groupby('stand')}
ancxy={k:(g.xy[v] if v is not None else None) for k,v in anc.items()}; dn=(pd.to_datetime(d.day)-pd.Timestamp('2025-01-01')).dt.days.to_numpy(); tau=np.full(len(d),np.nan)
for j,(s_,tem,px,py,r_) in enumerate(zip(d.stand,d.tem,d.px,d.py,d.rwy)):
    if s_ not in byst: continue
    if tem==1:
        if not np.isfinite(px): continue
        A_=np.array([px,py]); R_=60.0
    else:
        A_=ancxy.get((s_,r_)); R_=150.0
        if A_ is None: continue
    x,y,t_,ids,dns=byst[s_]; m=((x-A_[0])**2+(y-A_[1])**2<=R_**2)&(np.abs(dns-dn[j])>1)
    if m.sum()<2: continue
    dd=(x[m]-A_[0])**2+(y[m]-A_[1])**2; o_=np.lexsort((dd,ids[m])); first=np.unique(ids[m][o_],return_index=True)[1]
    if len(first)>=2: tau[j]=np.median(t_[m][o_][first])
d['tau_emp']=tau
ok=d.t_patio.notna()&np.isfinite(d.D_new)&np.isfinite(d.v); d=d[ok].reset_index(drop=True)
d['d']=pd.to_datetime(d.day); days=np.array(sorted(d.d.unique())); d['blk']=d.d.map({x:i//5 for i,x in enumerate(days)})
FACT={'stand':0.1,'hora':0.4,'oper':0.1,'acft':0.2,'rwy':0.05,'dow':0.0}
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
def roda(beta):
    mov0=d.D_new/d.v; mov=np.where(np.isfinite(d.tau_emp),d.tau_emp,mov0*(1+beta*d.cg)); x=d.copy(); x['mov']=mov; x['W']=x.ref-x.blk_rotulo-x.mov-x.t_patio
    for c in ('esp_visto','n_usados'): x[c]=x[c].fillna(0).astype(float)
    what=np.zeros(len(x))
    for cls,num in ((1,['mov','D_new','v','n_usados','esp_visto','cg']),(0,['mov','D_new','v','cg'])):
        mc=(x.tem==cls).to_numpy()
        for b in x.blk.unique():
            te=(x.blk==b).to_numpy()&mc; lo=x.d[x.blk==b].min()-pd.Timedelta(days=1); hi=x.d[x.blk==b].max()+pd.Timedelta(days=1); trn=mc&((x.d<lo)|(x.d>hi)).to_numpy()
            if te.sum()==0: continue
            Xtr,Xte=design(x[trn],x[te],num); what[te]=Ridge(alpha=1.0).fit(Xtr,x.W[trn]).predict(Xte)
    x['yh0']=x.mvt_s-(x.ref-x.mov-x.t_patio-what); return x
print('\n=== EGLL: GRAFO V2 (nunca pista, sentido, congestionamento, relogios) ===')
print('voos',len(d),'| caminho novo calculado:',round(np.isfinite(d.D_new).mean(),3),'| congestionamento medio no caminho (outros avioes por aresta):',round(d.cg.mean(),2),'| tau empirico:',round(d.tau_emp.notna().mean(),3))
best=None
for beta in (0.0,0.1,0.25,0.5,1.0,2.0):
    x=roda(beta); r=rm(x.y-x.yh0); print(f'  desconto de congestionamento beta={beta}: RMSE do grafo (antes dos relogios) {r:.1f}')
    if best is None or r<best[0]: best=(r,beta,x)
r0,beta,x=best; print('  -> beta escolhido',beta)
# ---- relogios como marcacoes de atraso: o tempo nao pode ser menor que o do relogio (aprende o deslocamento nos outros blocos)
def aplica(x,clocks,q):
    y=x.yh0.to_numpy().copy(); strech=np.zeros(len(x),bool)
    for b in x.blk.unique():
        te=(x.blk==b).to_numpy(); lo=x.d[x.blk==b].min()-pd.Timedelta(days=1); hi=x.d[x.blk==b].max()+pd.Timedelta(days=1); trn=((x.d<lo)|(x.d>hi)).to_numpy()
        for c in clocks:
            m=x['m_'+c]; off=(x.y[trn]-m[trn]).dropna().quantile(q); lim=(m+off)
            upd=te&lim.notna().to_numpy()&(lim.to_numpy()>y); strech|=upd; y=np.where(upd,lim.to_numpy(),y)
    return y,strech
res=[]
for clocks in (['aobt'],['eobt'],['sched'],['aobt','eobt','sched']):
    for q in (0.25,0.5,0.75):
        y,st=aplica(x,clocks,q); res.append((rm(x.y-y),clocks,q,st.mean(),y))
res.sort(key=lambda z:z[0])
for r,c,q,st,_ in res[:5]: print(f'  relogios {c} quantil {q}: RMSE {r:.1f} (esticou {st:.1%} dos voos)')
rbest,cb,qb,stb,yb=res[0]; x['yh']=yb
print('\n=========== EGLL jan+jul 2025, TODAS as partidas (n=%d), GRAFO V2 tudo ligado ==========='%len(x))
print('RMSE CAMPEAO: %.1f s'%rm(x.y-x.champ)); print('RMSE GRAFO:   %.1f s   (relogios %s, quantil %s, beta %s)'%(rm(x.y-x.yh),cb,qb,beta))
for m,z in x.groupby('mes'): print('   %s  campeao %.1f | grafo %.1f  (n=%d)'%(m,rm(z.y-z.champ),rm(z.y-z.yh),len(z)))
print('   com pings na janela (n=%d): campeao %.1f | grafo %.1f ;  sem pings (n=%d): campeao %.1f | grafo %.1f'%((x.tem==1).sum(),rm(x.y[x.tem==1]-x.champ[x.tem==1]),rm(x.y[x.tem==1]-x.yh[x.tem==1]),(x.tem==0).sum(),rm(x.y[x.tem==0]-x.champ[x.tem==0]),rm(x.y[x.tem==0]-x.yh[x.tem==0])))
x[['id','day','mes','y','champ','yh','yh0','tem']].to_parquet(f'{S}/sincronia_grafo/grafo_v2_{APT}.parquet')
