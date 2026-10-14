"""GRAFO V3, tudo ligado, grafo COMPLETO: ultima rota completa observada (mesmo stand e pista) como simulacao + restricoes na pista + congestionamento + patio + posterior de espera + relogios como piso."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import taxigraph as T
from sincro3 import Sync3
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store'); D_PB=70.0; V_PATIO=2.0; KT=0.514444
d=pd.read_parquet(f'{S}/sincronia_grafo/grafo_base_{APT}.parquet')
ctx=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','AIRCRAFT_TYPE_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt']).rename(columns={'MVT_ID_mvt':'id'})
pp=pd.read_parquet(f'{S}/sincronia_grafo/ponto_parada_{APT}.parquet'); d=d.merge(ctx,on='id',how='left').merge(pp,on='id',how='left'); mv=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True)
d['hora']=mv.dt.hour.astype(str); d['dow']=mv.dt.dayofweek.astype(str); d['stand']=d.STAND_mvt.astype(str); d['oper']=d.AIRCRAFT_OPERATOR_flt.astype(str).str[:8]; d['acft']=d.AIRCRAFT_TYPE_mvt.astype(str); d['rwy']=d.rwy.astype(str)
for nm,c in (('sched','SCHED_TIME_UTC_mvt'),('aobt','AOBT_3_flt'),('eobt','EOBT_1_flt')): d['m_'+nm]=(mv-pd.to_datetime(d[c],utc=True,errors='coerce')).dt.total_seconds()
sy=Sync3(APT); g=sy.g; Rt=T.Router(g); idx=T.build_stand_index(g.A); SPp=np.array([g.proj(x['lat'],x['lon']) for x in g.A.stands])
cong={}
for p in glob.glob(f'{S}/sincronia_grafo/congestao/{APT}_*.parquet'):
    a=pd.read_parquet(p); cong[os.path.basename(p)[len(APT)+1:-8]]={k:set(v) for k,v in a.groupby(['bin','k']).icao24}
tkpos={k:i for i,k in enumerate(sy.tk)}
def acesso_dist(F):
    best=np.inf
    for k in sy.tk:
        A_,B_=g.xy[sy.ea[k]],g.xy[sy.eb[k]]; ab=B_-A_; den=ab@ab; f=0. if den==0 else float(np.clip((F-A_)@ab/den,0,1)); best=min(best,float(np.hypot(*(F-(A_+f*ab)))))
    return best
info={}
for s_ in d.stand.dropna().unique():
    la,lo,lv=T.map_stand(s_,idx)
    if lv>=9: continue
    sp=np.array(g.proj(la,lo)); j=int(np.argmin(np.hypot(*(SPp-sp).T))); h=np.radians(90-g.A.stands[j]['hdg']); u=np.array([np.cos(h),np.sin(h)]); F=sp-D_PB*u
    opts,_=Rt.attach(la,lo); info[s_]=dict(t_patio=(acesso_dist(F)+D_PB)/V_PATIO,opts=opts,xy=sp)
anc={}
for (s_,r_) in set(zip(d.stand,d.rwy)):
    try: pair,cands,rl=g.targets(r_); anc[(s_,r_)]=min(cands,key=lambda c:c[1])[0] if cands else None
    except Exception: anc[(s_,r_)]=None
def tracar(M,opts,to_node=None,to_edge=None):
    D,P,Du,Pu=M; best=(np.inf,None,None,True,0.)
    if to_node is not None:
        for nd,acc in opts:
            c=acc+D[nd,to_node]
            if c<best[0]: best=(c,nd,to_node,True,0.)
        if not np.isfinite(best[0]):
            for nd,acc in opts:
                c=acc+Du[nd,to_node]
                if c<best[0]: best=(c,nd,to_node,False,0.)
    else:
        k,f=to_edge; a,b,L=sy.ea[k],sy.eb[k],sy.L[k]
        for dirc,(DD) in ((True,D),(False,Du)):
            for nd,acc in opts:
                for end,off in ((a,f*L),(b,(1-f)*L)):
                    c=acc+DD[nd,end]+off
                    if c<best[0]: best=(c,nd,end,dirc,off)
            if np.isfinite(best[0]): break
    if not np.isfinite(best[0]): return np.nan,[]
    c,nd,via,dirc,off=best; P_=P if dirc else Pu; cur=via; path=[cur]
    while cur!=nd and P_[nd,cur]>=0: cur=int(P_[nd,cur]); path.append(cur)
    path=path[::-1]; ed=[(sy.ek[(a_,b_)],sy.L[sy.ek[(a_,b_)]]) for a_,b_ in zip(path[:-1],path[1:]) if (a_,b_) in sy.ek]
    if to_edge is not None: ed.append((to_edge[0],off))
    return c,ed
Dn=np.full(len(d),np.nan); ED=[None]*len(d); P_xy=d[['px','py']].to_numpy(); kk=np.full(len(d),-1); ff=np.zeros(len(d)); ch=np.where((d.tem==1)&np.isfinite(P_xy[:,0]))[0]
if len(ch): k_,f_=sy.snap_many(P_xy[ch]); kk[ch]=k_; ff[ch]=f_
cache_n={}
for j,(s_,r_,tem) in enumerate(zip(d.stand,d.rwy,d.tem)):
    if s_ not in info: continue
    M=sy.matriz(r_); opts=info[s_]['opts']
    if tem==1 and kk[j]>=0: c,ed=tracar(M,opts,to_edge=(kk[j],ff[j]))
    elif tem==1: c,ed=d.D_star.iloc[j],[]
    else:
        key=(s_,r_)
        if key not in cache_n: cache_n[key]=tracar(M,opts,to_node=anc[key]) if anc.get(key) is not None else (np.nan,[])
        c,ed=cache_n[key]
    Dn[j]=c; ED[j]=ed
d['D_new']=Dn; d['t_patio']=d.stand.map({k:v['t_patio'] for k,v in info.items()})
d['v']=np.where(d.tem==1,d.v_bar.fillna(d.v_h).clip(1.5,12),d.v_h); d['ref']=np.where(d.tem==1,d.t_star,d.mvt_s)
cg=np.zeros(len(d))
for j in range(len(d)):
    ed=ED[j]
    if not ed or not np.isfinite(Dn[j]): continue
    b=int((d.ref.iloc[j]-0.5*Dn[j]/d.v.iloc[j])//600); tab=cong.get(d.day.iloc[j]); own=d.icao.iloc[j] if isinstance(d.icao.iloc[j],str) else None
    if tab is None: continue
    tot=0.;wl=0.
    for k,Lk in ed:
        s2=tab.get((b,k)); n_=(len(s2)-(1 if (own is not None and s2 and own in s2) else 0)) if s2 else 0; tot+=Lk*n_; wl+=Lk
    cg[j]=tot/wl if wl>0 else 0.
d['cg']=cg
# ---- ULTIMA ROTA COMPLETA DISPONIVEL: voo anterior (mesmo stand e pista) cuja trilha observada vai da vizinhanca do stand ate a pista
ref=pd.read_parquet(f'{S}/sincronia_grafo/trilhas_ref_{APT}.parquet'); meta=d.drop_duplicates('id').set_index('id')
ref['mvt']=ref.id.map(meta.mvt_s)
R={}
for i,q in ref.groupby('id'):
    q=q.sort_values('tau'); x=q.x.to_numpy(); y=q.y.to_numpy(); t=q.tau.to_numpy(); dt=np.diff(t); dd=np.hypot(np.diff(x),np.diff(y)); m=(dt>0)&(dt<=30)&(dd/np.maximum(dt,1e-9)>1)
    R[i]=dict(stand=q.stand.iloc[0],rwy=q.rwy.iloc[0],mvt=q.mvt.iloc[0],x=x,y=y,tau=t,v=float(dd[m].sum()/dt[m].sum()) if m.sum()>=3 else np.nan)
R=pd.DataFrame(R).T; R['mvt']=R.mvt.astype(float); stxy={s_:info[s_]['xy'] for s_ in info}
por_sr={k:v.sort_values('mvt') for k,v in R.groupby(['stand','rwy'])}; por_s={k:v.sort_values('mvt') for k,v in R.groupby('stand')}; por_r={k:v.sort_values('mvt') for k,v in R.groupby('rwy')}
ancxy={k:(g.xy[v] if v is not None else None) for k,v in anc.items()}
def tau_na_rota(r,A_,R_,sp):
    x,y=r['x'],r['y']; dd=np.hypot(x-A_[0],y-A_[1]); i=int(dd.argmin())
    if dd[i]>R_: return None
    iS=int(np.argmin(np.hypot(x-sp[0],y-sp[1])))                      # ponto da rota mais perto do MEU stand (o destino e o meu stand, nao o da rota)
    return (r['tau'][i],r['v'],dd[i],r['tau'][iS],x[iS],y[iS])
viz={s_:[k for k,v in stxy.items() if k!=s_ and np.hypot(*(v-stxy[s_]))<=300] for s_ in stxy}; cache_viz={}
tau=np.full(len(d),np.nan); fonte=np.zeros(len(d),int); refid=np.full(len(d),-1,dtype=np.int64); tauS=np.full(len(d),np.nan); sxy=np.full((len(d),2),np.nan); vref=np.full(len(d),np.nan)
for j,(s_,tem,px,py,r_,mt,idj) in enumerate(zip(d.stand,d.tem,d.px,d.py,d.rwy,d.mvt_s,d.id)):
    if s_ not in info: continue
    if tem==1:
        if not np.isfinite(px): continue
        A_=np.array([px,py]); R_=60.0
    else:
        A_=ancxy.get((s_,r_)); R_=150.0
        if A_ is None: continue
    # 1) mesmo stand e pista: a ultima rota completa ANTES do voo (se nao houver antes, a mais proxima depois); 2) mesmo stand, outra pista; 3) stands a <=300 m, mesma pista
    for nivel,tab in ((1,por_sr.get((s_,r_))),(2,por_s.get(s_)),(3,None)):
        if nivel==3:
            if (s_,r_) not in cache_viz:
                lst=[por_sr[(k,r_)] for k in viz.get(s_,[]) if (k,r_) in por_sr]; cache_viz[(s_,r_)]=pd.concat(lst).sort_values('mvt') if lst else None
            tab=cache_viz[(s_,r_)]
        if tab is None or not len(tab): continue
        ant=tab[(tab.mvt<mt)&(tab.index!=idj)].iloc[::-1]; post=tab[(tab.mvt>=mt)&(tab.index!=idj)]
        got=None
        for _,rr in pd.concat([ant,post]).iterrows():            # mais recente antes; depois, a mais proxima depois
            o=tau_na_rota(rr,A_,R_,info[s_]['xy'])
            if o is not None: got=o; rid_=rr.name; break
        if got is not None:
            t_,vr,_,tS,sx_,sy_=got; fonte[j]=nivel; refid[j]=rid_; tau[j]=t_; tauS[j]=tS; sxy[j]=(sx_,sy_); vref[j]=vr; break
d['tau_A']=tau; d['tau_S']=tauS; d['sx']=sxy[:,0]; d['sy_']=sxy[:,1]; d['vref']=vref; d['fonte']=fonte; d['ref_id']=refid
# trecho do MEU stand ate o ponto da rota de referencia mais perto dele (Dijkstra no grafo completo, restricoes de pista), e o trecho guiado pela rota
sk,sf=sy.snap_many(np.nan_to_num(sxy,nan=0.0)); Dleg=np.full(len(d),np.nan)
for j in np.where(fonte>0)[0]:
    s_=d.stand.iloc[j]
    if s_ not in info: continue
    if sk[j]>=0: Dleg[j]=tracar(sy.matriz(d.rwy.iloc[j]),info[s_]['opts'],to_edge=(sk[j],sf[j]))[0]
    else: Dleg[j]=float(np.hypot(*(info[s_]['xy']-sxy[j])))
esc=np.where((np.isfinite(vref))&(vref>0),vref/d.v.to_numpy(),1.0)
guia=np.where(np.isfinite(tauS)&(tau>=tauS),(tau-tauS)*esc+Dleg/d.v.to_numpy(),tau*esc)            # (tempo na rota de referencia, do ponto mais perto do meu stand ate a ancora) + (do meu stand ate la)
d['tau_emp']=np.where(fonte>0,guia,np.nan)
d['ED']=ED
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
DAYS=sorted(d.day.unique()); DI={x:i for i,x in enumerate(DAYS)}; d['didx']=d.day.map(DI); d['dstart']=pd.to_datetime(d.day).astype('int64')//10**9
def rastros(ref_t,mov,ED,didx,dstart,gamma,iters=3):
    """cada partida deixa rastro nas arestas do caminho nos instantes em que passa; o rastro dos OUTROS reduz a velocidade (aumenta o tempo) de quem passa ali. Itera ate estabilizar."""
    n=len(ref_t); nE=len(sy.E); NB=len(DAYS)*1440; mov_cur=mov.copy(); rho_med=np.zeros(n)
    for it in range(iters+1):
        occ=np.zeros((nE,NB),np.float32); vis=[None]*n
        for j in range(n):
            ed=ED[j]
            if not ed or not np.isfinite(mov_cur[j]): continue
            ks=np.array([e[0] for e in ed]); Ls=np.array([e[1] for e in ed],float); tot=Ls.sum()
            if tot<=0: continue
            end=(ref_t[j]-mov_cur[j]-dstart[j])+np.cumsum(Ls)/tot*mov_cur[j]; ini=end-Ls/tot*mov_cur[j]; items=[]; base=didx[j]*1440
            for k,a_,b_ in zip(ks,ini,end):
                dur=max(b_-a_,1e-6)
                for bn in range(int(a_//60),int(b_//60)+1):
                    w=(min(b_,(bn+1)*60)-max(a_,bn*60))/dur
                    if w>0 and 0<=bn<1440: items.append((k,base+bn,w)); occ[k,base+bn]+=w
            vis[j]=items
        if it==iters: break
        for j in range(n):
            items=vis[j]
            if not items: continue
            acc=sum(w*(occ[k,bn]-w) for k,bn,w in items); sw=sum(w for _,_,w in items); rho=acc/max(sw,1e-9); rho_med[j]=rho; mov_cur[j]=mov[j]*(1+gamma*rho)
    return mov_cur,rho_med
def roda(gamma):
    x=d.copy(); mov_base=np.where(np.isfinite(x.tau_emp),x.tau_emp,x.D_new/x.v)
    mov_f,rho=rastros(x.ref.to_numpy(float),mov_base.astype(float),x.ED.tolist(),x.didx.to_numpy(),x.dstart.to_numpy(),gamma)
    x['mov']=mov_f; x['rho']=rho; x['W']=x.ref-x.blk_rotulo-x.mov-x.t_patio
    for c in ('esp_visto','n_usados'): x[c]=x[c].fillna(0).astype(float)
    what=np.zeros(len(x))
    for cls,num in ((1,['mov','D_new','v','n_usados','esp_visto','rho']),(0,['mov','D_new','v','rho'])):
        mc=(x.tem==cls).to_numpy()
        for b in x.blk.unique():
            te=(x.blk==b).to_numpy()&mc; lo=x.d[x.blk==b].min()-pd.Timedelta(days=1); hi=x.d[x.blk==b].max()+pd.Timedelta(days=1); trn=mc&((x.d<lo)|(x.d>hi)).to_numpy()
            if te.sum()==0: continue
            Xtr,Xte=design(x[trn],x[te],num); what[te]=Ridge(alpha=1.0).fit(Xtr,x.W[trn]).predict(Xte)
    x['yh0']=x.mvt_s-(x.ref-x.mov-x.t_patio-what); return x
print('\n=== EGLL: GRAFO V4 (rota de referencia como guia ate o MEU stand + rastros em tempo real) ===')
print('voos',len(d),'| rota observada usada: %.1f%% (mesmo stand+pista %.1f%%, mesmo stand outra pista %.1f%%, stand vizinho %.1f%%) | sem rota observada: Dijkstra no grafo completo %.1f%%'%(100*(d.fonte>0).mean(),100*(d.fonte==1).mean(),100*(d.fonte==2).mean(),100*(d.fonte==3).mean(),100*(d.fonte==0).mean()))
best=None
for gamma in (0.0,0.25,0.5,1.0,2.0):
    x=roda(gamma); r=rm(x.y-x.yh0); print('  rastros em tempo real, gamma=%s: ocupacao media dos OUTROS nas arestas do caminho %.2f | RMSE do grafo (antes dos relogios) %.1f'%(gamma,x.rho.mean(),r),flush=True)
    if best is None or r<best[0]: best=(r,gamma,x)
r0,beta,x=best; print('  -> gamma escolhido',beta)
def aplica(x,clocks,q):
    y=x.yh0.to_numpy().copy(); st=np.zeros(len(x),bool)
    for b in x.blk.unique():
        te=(x.blk==b).to_numpy(); lo=x.d[x.blk==b].min()-pd.Timedelta(days=1); hi=x.d[x.blk==b].max()+pd.Timedelta(days=1); trn=((x.d<lo)|(x.d>hi)).to_numpy()
        for c in clocks:
            m=x['m_'+c]; off=(x.y[trn]-m[trn]).dropna().quantile(q); lim=(m+off); upd=te&lim.notna().to_numpy()&(lim.to_numpy()>y); st|=upd; y=np.where(upd,lim.to_numpy(),y)
    return y,st
res=[]
for clocks in (['aobt'],['eobt'],['sched']):
    for q in (0.1,0.25,0.5):
        y,st=aplica(x,clocks,q); res.append((rm(x.y-y),clocks,q,st.mean(),y))
res.sort(key=lambda z:z[0]); rb,cb,qb,stb,yb=res[0]; x['yh']=yb
print('relogios (marcacoes de atraso): %s quantil %s, esticou %.1f%% dos voos'%(cb,qb,100*stb))
print('\n=========== EGLL jan+jul 2025, TODAS as partidas (n=%d), GRAFO V4 tudo ligado ==========='%len(x))
print('RMSE CAMPEAO: %.1f s'%rm(x.y-x.champ)); print('RMSE GRAFO:   %.1f s'%rm(x.y-x.yh))
for m,z in x.groupby('mes'): print('   %s  campeao %.1f | grafo %.1f  (n=%d)'%(m,rm(z.y-z.champ),rm(z.y-z.yh),len(z)))
print('   com pings na janela (n=%d): campeao %.1f | grafo %.1f ;  sem pings (n=%d): campeao %.1f | grafo %.1f'%((x.tem==1).sum(),rm(x.y[x.tem==1]-x.champ[x.tem==1]),rm(x.y[x.tem==1]-x.yh[x.tem==1]),(x.tem==0).sum(),rm(x.y[x.tem==0]-x.champ[x.tem==0]),rm(x.y[x.tem==0]-x.yh[x.tem==0])))
x[['id','day','mes','y','champ','yh','yh0','tem','fonte','ref_id','stand','rwy','px','py','tau_emp','D_new','ref','mvt_s','blk_rotulo','tau_A','tau_S','sx','sy_','rho']].to_parquet(f'{S}/sincronia_grafo/grafo_v4_{APT}.parquet')
