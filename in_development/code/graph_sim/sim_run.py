import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import taxigraph as T, regride as R
from sim_dia import Dia, KT, ROLL_DEP, ROLL_ARR
def prepara(d):
    """caminhos (stand<->pista), cadeias de ADS-B desde o MVT, flags de observacao e trilhas observadas"""
    sy=d.sy; g=d.g; t0=d.t0; V=d.a.gs.to_numpy()*KT; ok=(d.a.k.to_numpy()>=0)&(V>1)&(V<15); d.v=float(np.median(V[ok])) if ok.any() else 9.0
    D=[]
    for r in d.dep.itertuples():
        si=d.stand_info(r.STAND_mvt); an=d.anchor(str(r.RUNWAY_mvt))
        rec=dict(id=r.id,cs=r.cs,mvt=r.mvt,blk=r.blk,rwy=str(r.RUNWAY_mvt),stand=r.STAND_mvt,ok=False,ed=[],chain=False,ed_h=[],tstar=np.nan,obs=None,own=None)
        if si is None or an is None: D.append(rec); continue
        c,ed=d.caminho(rec['rwy'],si['opts'],to_node=an)
        if not ed: D.append(rec); continue
        rec.update(ok=True,ed=ed,t_patio=si['t_patio'],xy=si['xy'],F=si['F'])
        rec['obs']=d.liga(r.cs,r.mvt-5400,r.mvt+30)
        q,lig=R.pings(d.grp,d.gn,r.cs,r.mvt)
        if q is not None:
            P=np.c_[(q.lon-g.lon0)*g.kx,(q.lat-g.lat0)*g.ky]; t=q.t.to_numpy(); gs=q.gs.to_numpy()*KT; la,lo,lv=T.map_stand(r.STAND_mvt,d.idx)
            i,ds,_=R.regride(d.old,d.Rt,la,lo,P,t,gs); kk,ff=sy.snap_many(P[i:i+1])
            if kk[0]>=0:
                c2,ed2=d.caminho(rec['rwy'],si['opts'],to_edge=(kk[0],ff[0]))
                if ed2: rec.update(chain=True,ed_h=ed2,tstar=float(t[i]),Pstar=P[i],icao=q.icao24.iloc[-1],pings=P[i:],tp=t[i:])
        D.append(rec)
    A=[]
    for r in d.arr.itertuples():
        si=d.stand_info(r.STAND_mvt); an=d.anchor(str(r.RUNWAY_mvt)); rec=dict(id=r.id,cs=r.cs,ok=False,ed=[],obs=None,t_in=r.mvt+ROLL_ARR,t_out=r.blk)
        if si is None or an is None: A.append(rec); continue
        c,ed=d.caminho(str(r.RUNWAY_mvt),si['opts'],to_node=an)
        if ed: rec.update(ok=True,ed=ed[::-1]); rec['obs']=d.liga(r.cs,r.mvt-300,r.blk+300)
        A.append(rec)
    # trilhas observadas (todos os avioes com ping de solo em aresta de taxi): ocupacao em minutos por (aresta, minuto) e lista por icao24
    a=d.a[d.a.k>=0]; tt=a.t.to_numpy(); nx=np.r_[tt[1:],(tt[-1]+30 if len(tt) else 0.0)]; same=np.r_[a.icao24.to_numpy()[1:]==a.icao24.to_numpy()[:-1],False]; dt=np.where(same,np.minimum(nx-tt,30.0),10.0)
    b=np.clip(((tt-t0)//60).astype(int),0,d.NB-1); w=dt/60.0; O=np.zeros((d.nE,d.NB),np.float32); np.add.at(O,(a.k.to_numpy(),b),w)
    own={ic:(g_.k.to_numpy(),np.clip(((g_.t.to_numpy()-t0)//60).astype(int),0,d.NB-1),w[g_.index.map({ix:i for i,ix in enumerate(a.index)}).to_numpy()]) for ic,g_ in a.groupby('icao24')}
    return D,A,O,own
def visitas(ed,t_fim,total,v_dur,t0):
    """arestas com duracao v_dur (s por aresta); termina em t_fim (partidas) -> tempos de entrada/saida"""
    Ls=np.array([e[1] for e in ed],float); ks=np.array([e[0] for e in ed]); end=t_fim-(v_dur.sum()-np.cumsum(v_dur)); ini=end-v_dur; return ks,Ls,ini,end
def adiciona(O,ks,ini,end,t0,w=1.0):
    items=[]; dur=np.maximum(end-ini,1e-6)
    for k,a_,b_,d_ in zip(ks,ini,end,dur):
        for bn in range(int((a_-t0)//60),int((b_-t0)//60)+1):
            if 0<=bn<1440:
                f=(min(b_,t0+(bn+1)*60)-max(a_,t0+bn*60))/d_
                if f>0: items.append((k,bn,f*w)); O[k,bn]+=f*w
    return items
def kvfun(kv,delta):
    out=np.zeros_like(delta,float); m=(delta>=0)&(delta<len(kv)); out[m]=kv[delta[m].astype(int)]; return out
def simula(d,D,A,Oobs,own_obs,modo,Tm,Rm,wads,gamma,iters=3):
    """modo 'S': sem ADS (todos simulados). modo 'A': com ADS (cadeia desde o MVT + rastros observados com peso wads + simulados nos sem ADS)."""
    t0=d.t0; Ak,kv=d.kernel(Tm,Rm); v=d.v
    # fontes simuladas e o trecho simulado de cada partida
    S=[]
    for j,r in enumerate(D):
        if not r['ok']: continue
        if modo=='A' and r['chain']: S.append(dict(src='dep',j=j,ed=r['ed_h'],fim=r['tstar'],tipo='h'))
        elif modo=='A' and r['obs'] is not None and not r['chain']: continue       # observada sem cadeia: so o ADS-B (nao duplica)
        else: S.append(dict(src='dep',j=j,ed=r['ed'],fim=r['mvt']-ROLL_DEP,tipo='f'))
    for j,r in enumerate(A):
        if not r['ok'] or (modo=='A' and r['obs'] is not None): continue
        S.append(dict(src='arr',j=j,ed=r['ed'],t_in=r['t_in'],t_out=r['t_out']))
    for s in S:
        Ls=np.array([e[1] for e in s['ed']],float); s['Ls']=Ls; s['ks']=np.array([e[0] for e in s['ed']]); s['dur']=Ls/v
        if s['src']=='arr': tot=max(s['t_out']-s['t_in'],1.0); s['dur']=Ls/Ls.sum()*tot; s['fixa']=True
    rho_medio=np.zeros(len(S))
    for it in range(iters+1):
        O=np.zeros((d.nE,d.NB),np.float32)
        for s in S:
            if s['src']=='arr': end=s['t_in']+np.cumsum(s['dur']); ini=end-s['dur']
            else: end=s['fim']-(s['dur'].sum()-np.cumsum(s['dur'])); ini=end-s['dur']
            s['ini'],s['end']=ini,end; s['items']=adiciona(O,s['ks'],ini,end,t0)
        Ot=O+(wads*Oobs if modo=='A' else 0)
        rho=d.rho(Ot,Ak,kv)
        if it==iters: break
        for si_,s in enumerate(S):
            if s['src']=='arr': continue
            r=D[s['j']]; mid=((s['ini']+s['end'])/2-t0)//60; mid=np.clip(mid.astype(int),0,d.NB-1); ks=s['ks']; rh=rho[ks,mid].copy()
            # tira o PROPRIO rastro (simulado e, no modo A, observado)
            it_=np.array(s['items']); 
            if len(it_): rh-= (Ak[ks[:,None],it_[:,0].astype(int)[None,:]]*kvfun(kv,mid[:,None]-it_[:,1].astype(int)[None,:])*it_[:,2][None,:]).sum(1)
            if modo=='A' and r.get('icao') in own_obs:
                ok_,ob_,ow_=own_obs[r['icao']]; rh-=wads*(Ak[ks[:,None],ok_[None,:]]*kvfun(kv,mid[:,None]-ob_[None,:])*ow_[None,:]).sum(1)
            rh=np.maximum(rh,0); rho_medio[si_]=float((rh*s['Ls']).sum()/s['Ls'].sum()); s['dur']=(s['Ls']/v)*(1+gamma*rh)
    return S,rho,rho_medio
def resultado(d,D,S,modo):
    sim={s['j']:s for s in S if s['src']=='dep'}; rows=[]
    for j,r in enumerate(D):
        if not r['ok']: continue
        if modo=='A' and r['obs'] is not None and not r['chain']: continue
        s=sim.get(j)
        if s is None: continue
        mov=float(s['dur'].sum())
        part=(r['mvt']-r['tstar'])+mov+r['t_patio'] if (modo=='A' and r['chain']) else ROLL_DEP+mov+r['t_patio']
        rows.append(dict(j=j,y=r['mvt']-r['blk'],part=part,mov=mov,chain=(modo=='A' and r['chain'])))
    x=pd.DataFrame(rows); c=(x.y-x.part).median(); x['res']=x.y-x.part-c; return x
