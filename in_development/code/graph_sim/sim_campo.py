"""Simulacao do DIA INTEIRO por campo de rastro. Todos os avioes (partidas: trecho escondido stand->ancora pela rota recente/Dijkstra, regredindo do MVT; chegadas: pista->stand) andam juntos em passos de 1 unidade
(media entre dois pings do ADS-B); a cada unidade avancada o aviao EMITE rastro em todas as direcoes do grafo ate X km (nucleo linear), que decai no tempo (constante tau);
o rastro dos OUTROS (ADS-B observado com peso wads, simulado com peso 1) reduz a velocidade: f = 1/(1+gamma*campo). Devolve a duracao simulada do trecho escondido de cada partida."""
import sys, os, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from multiprocessing import Pool
from sincro3 import Sync3
R0=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/rotas_'+os.environ.get('APT','EGLL')); APT=os.environ.get('APT','EGLL'); ROLL_DEP=45.0
UNIT=12.0; VREF=9.0
_G={}
def graf():
    global UNIT,VREF
    if 'sy' not in _G:
        import glob as _gl; _ps=sorted(_gl.glob(f'{R0}/20*.pkl'))
        if _ps:
            _m=[pickle.load(open(p_,'rb')) for p_ in _ps[::max(1,len(_ps)//40)]]; _u=[m_['unit'] for m_ in _m if 'unit' in m_ and m_['unit']==m_['unit']]; _v=[m_['vref'] for m_ in _m if 'vref' in m_ and m_['vref']==m_['vref']]; UNIT=float(np.mean(_u)) if _u else 12.0; VREF=float(np.mean(_v)) if _v else 9.0
        sy=Sync3(APT); end=sy.g.A.runways[0]['end1']; Du=sy.matriz(end)[2]; ea,eb=sy.ea,sy.eb
        _G['sy']=sy; _G['dEE']=np.minimum.reduce([Du[np.ix_(ea,ea)],Du[np.ix_(ea,eb)],Du[np.ix_(eb,ea)],Du[np.ix_(eb,eb)]]); _G['guias']=pickle.load(open(f'{R0}/_guias.pkl','rb')) if (os.path.exists(f'{R0}/_guias.pkl') and not os.environ.get('SEM_GUIA')) else {}
    return _G
def agentes(Dia,G,day):
    out=[]
    for i,r in enumerate(Dia['D']):
        if not r['ok']: continue
        gd=G.get((day,i)); ed=gd['ed_guide'] if gd is not None else (r['ed_h'] if (r['chain'] and r['ed_h']) else r['ed'])
        if not ed: continue
        Ls=np.array([e[1] for e in ed],float)
        if Ls.sum()<=0: continue
        fim=r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-ROLL_DEP
        out.append(dict(tipo='dep',i=i,ks=np.array([e[0] for e in ed]),cum=np.cumsum(Ls),tot=float(Ls.sum()),fim=float(fim),emit=(r['obs'] is None or r['chain']),guia=(gd is not None)))
    for j,r in enumerate(Dia['A']):
        if not r['ok'] or r['obs'] is not None or not r['ed']: continue
        Ls=np.array([e[1] for e in r['ed']],float)
        if Ls.sum()<=0 or r['t_out']<=r['t_in']: continue
        out.append(dict(tipo='arr',i=j,ks=np.array([e[0] for e in r['ed']]),cum=np.cumsum(Ls),tot=float(Ls.sum()),ini=float(r['t_in']),fim=float(r['t_out']),emit=True))
    return out
def passe(ag,ads,t0,pr,K,starts,gmul=None,emul=None):
    X,gam0,tau,wads=pr['X'],pr['gamma'],pr['tau'],pr['wads']; dt=UNIT/VREF; dec=np.exp(-dt/tau); ge=1-dec; nE=K.shape[0]
    t_ini=min(starts.values())-5; t_fim=max(a['fim'] for a in ag)+1800; ns=int((t_fim-t_ini)/dt)+1
    I=np.zeros(nE); order=sorted(range(len(ag)),key=lambda q:starts[q]); ptr=0; act=[]; S0={};S1={};pos={};prog={};dur={};ia=0
    ta,ka,wa=ads['t'],ads['k'],ads['w']; na=len(ta); wad=wads*wa/dt
    for s in range(ns):
        t=t_ini+s*dt; I*=dec
        j=ia
        while j<na and ta[j]<t+dt: j+=1
        if j>ia: I+=ge*(wad[ia:j,None]*K[ka[ia:j]]).sum(0); ia=j
        while ptr<len(order) and starts[order[ptr]]<=t: q=order[ptr]; act.append(q); pos[q]=0.0; prog[q]=0.0; S0[q]=0.0; S1[q]=0.0; ptr+=1
        if not act: continue
        em=[]; emq=[]; fin=[]
        for q in act:
            a=ag[q]
            if a['tipo']=='arr':
                tgt=a['tot']*(t-a['ini'])/(a['fim']-a['ini']); 
                while pos[q]+UNIT<=tgt and pos[q]<a['tot']: pos[q]+=UNIT; S0[q]=S0[q]*dec+1; S1[q]=S1[q]*dec+pos[q]; em.append(a['ks'][min(int(np.searchsorted(a['cum'],pos[q]+1e-9)),len(a['ks'])-1)]); emq.append(q)
                if t>=a['fim']: fin.append(q)
                continue
            e=a['ks'][min(int(np.searchsorted(a['cum'],pos[q]+1e-9)),len(a['ks'])-1)]
            gam=gam0*(gmul[q] if gmul is not None else 1.0)
            if gam>0 and e>=0:
                own=max(0.0,ge*(S0[q]*(1-pos[q]/X)+S1[q]/X)); f=1.0/(1.0+gam*max(0.0,I[e]-own))
            else: f=1.0
            prog[q]+=f
            if prog[q]>=1.0:
                prog[q]-=1.0; pos[q]+=UNIT; S0[q]=S0[q]*dec+1; S1[q]=S1[q]*dec+pos[q]
                if a['emit'] and e>=0: em.append(e); emq.append(q)
            if pos[q]>=a['tot']-1e-6: fin.append(q); dur[q]=t+dt-starts[q]
        if em: I+=ge*(np.array([emul[q_] if emul is not None else 1.0 for q_ in emq])[:,None]*K[np.array(em)]).sum(0)
        for q in fin: act.remove(q)
    return dur

def passe_rev(ag,ads,t0,pr,K,gmul=None,emul=None,rec=None,stats=None):
    """TEMPO REVERSO: cada partida comeca no seu MVT (ancora) e REGRIDE ate o stand; quem tem MVT maior regride primeiro. Rastros de quem ja esta regredindo e do ADS-B (ancoras imutaveis, peso maior)
    IMPEDEM quem sai depois: f = 1/(1+(campo/theta)^p). Rastro emitido a cada unidade avancada, em todas as direcoes do grafo ate X, nucleo (1-d/X)^q."""
    X,tau,wads,th,pw=pr['X'],pr['tau'],pr['wads'],pr['theta'],pr['p']; dt=UNIT/VREF; dec=np.exp(-dt/tau); ge=1-dec; nE=K.shape[0]
    T_hi=max(a['fim'] for a in ag)+5; T_lo=min((a['ini']-60) if a['tipo']=='arr' else (a['fim']-6*a['tot']/VREF-600) for a in ag); ns=int((T_hi-T_lo)/dt)+1
    I=np.zeros(nE); order=sorted(range(len(ag)),key=lambda q:-ag[q]['fim']); ptr=0; act=[]; S0={};S1={};pos={};prog={};dur={};ta,ka,wa=ads['t'],ads['k'],ads['w']; ia=len(ta); wad=wads*wa/dt; last={}
    for s in range(ns):
        t=T_hi-s*dt; I*=dec
        j=ia
        while j>0 and ta[j-1]>=t: j-=1
        if j<ia: I+=ge*(wad[j:ia,None]*K[ka[j:ia]]).sum(0); ia=j
        while ptr<len(order) and ag[order[ptr]]['fim']>=t: q=order[ptr]; act.append(q); pos[q]=0.0; prog[q]=0.0; S0[q]=0.0; S1[q]=0.0; last[q]=-1; ptr+=1
        if rec is not None and rec.get('t_snap') is not None and abs(t-rec['t_snap'])<dt/2 and 'I' not in rec:
            rec['I']=I.copy(); ags=[]
            for q in act:
                a_=ag[q]
                if a_['tipo']=='dep': ags.append((q,a_['ks'][::-1][min(int(np.searchsorted(a_['cum_r'],pos[q]+1e-9)),len(a_['ks'])-1)]))
                else:
                    pp=a_['tot']*(t-a_['ini'])/(a_['fim']-a_['ini']); ags.append((q,a_['ks'][min(int(np.searchsorted(a_['cum'],pp+1e-9)),len(a_['ks'])-1)]))
            rec['agentes']=ags
        if not act: continue
        em=[]; emq=[]; fin=[]
        for q in act:
            a=ag[q]
            if a['tipo']=='arr':
                if t<a['ini']: fin.append(q); continue
                pp=a['tot']*(t-a['ini'])/(a['fim']-a['ini']); un=int(pp/UNIT); ea_=a['ks'][min(int(np.searchsorted(a['cum'],pp+1e-9)),len(a['ks'])-1)]
                if rec is not None and rec.get('tl') is not None: rec['tl'].append((q,t,int(ea_)))
                if un!=last[q]: last[q]=un; em.append(ea_); emq.append(q)
                continue
            ks_r=a['ks'][::-1]; e=ks_r[min(int(np.searchsorted(a['cum_r'],pos[q]+1e-9)),len(ks_r)-1)]
            if rec is not None and rec.get('tl') is not None: rec['tl'].append((q,t,int(e)))
            if e>=0:
                own=max(0.0,ge*(S0[q]*(1-pos[q]/X)+S1[q]/X)); fld=max(0.0,I[e]-own)*(gmul[q] if gmul is not None else 1.0); f=1.0/(1.0+(fld/th)**pw)
            else: fld=0.0; f=1.0
            if rec is not None and rec.get('alvo')==q: rec.setdefault('traj',[]).append((t,pos[q],int(e),f,fld))
            if stats is not None:
                z=stats.get(q)
                if z is None: z=stats[q]=dict(n=0,n_lo=0,n_vlo=0,sf=0.0,sf2=0.0,mn=1.0,sfld=0.0,mxfld=0.0,run=0,maxrun=0,cross=0,prevlo=False,sf3=[0.0,0.0,0.0],n3=[0,0,0],nstop=0)
                z['n']+=1; z['sf']+=f; z['sf2']+=f*f; z['mn']=min(z['mn'],f); z['sfld']+=fld; z['mxfld']=max(z['mxfld'],fld)
                lo=f<0.5; z['n_lo']+=lo; z['n_vlo']+=(f<0.2)
                if lo: z['run']+=1; z['maxrun']=max(z['maxrun'],z['run'])
                else: z['run']=0
                if lo!=z['prevlo']: z['cross']+=1
                z['prevlo']=lo; i3=min(2,int(3*pos[q]/a['tot'])); z['sf3'][i3]+=f; z['n3'][i3]+=1
            prog[q]+=f
            if prog[q]>=1.0:
                prog[q]-=1.0; pos[q]+=UNIT; S0[q]=S0[q]*dec+1; S1[q]=S1[q]*dec+pos[q]
                if a['emit'] and e>=0: em.append(e); emq.append(q)
            if pos[q]>=a['tot']-1e-6: fin.append(q); dur[q]=a['fim']-(t-dt)
            elif a['fim']-t>6*a['tot']/VREF+600: fin.append(q); dur[q]=a['fim']-t
        if em: I+=ge*(np.array([emul[q_] if emul is not None else 1.0 for q_ in emq])[:,None]*K[np.array(em)]).sum(0)
        for q in fin: act.remove(q)
    return dur

def simula_dia(args):
    day,pr=args[0],args[1]; gm=pr.get('gm',{}); em_=pr.get('em',{}); G=graf(); Dia=pickle.load(open(f'{R0}/{day}.pkl','rb')); ag=agentes(Dia,G['guias'],day)
    if not any(a['tipo']=='dep' for a in ag): return pd.DataFrame()
    dEE=G['dEE']; K=np.clip(1-dEE/pr['X'],0,None); K[~np.isfinite(K)]=0
    if pr.get('rev'):
        for a in ag: a['cum_r']=np.cumsum(np.r_[a['cum'][0],np.diff(a['cum'])][::-1]) if False else np.cumsum(np.diff(np.r_[0.0,a['cum']])[::-1])
        gmul=[gm.get(a['i'],1.0) if a['tipo']=='dep' else 1.0 for a in ag]; emul=[em_.get(a['i'],1.0) if a['tipo']=='dep' else 1.0 for a in ag]; rec=pr.get('rec')
        if rec is not None and rec.get('alvo_i') is not None: rec['alvo']=[q for q,a in enumerate(ag) if a['tipo']=='dep' and a['i']==rec['alvo_i']][0]; rec['ag']=ag
        stt={} if pr.get('stats') else None; d1=passe_rev(ag,Dia['ads'],Dia['t0'],pr,K,gmul,emul,rec,stt); rows=[]
        for q,a in enumerate(ag):
            if a['tipo']=='dep':
                row=dict(day=day,i=a['i'],id=Dia['D'][a['i']]['id'],L=a['tot'],dur=d1.get(q,np.nan),dur_livre=a['tot']/VREF,guia=a['guia'],fim=a['fim'])
                if stt is not None:
                    z=stt.get(q); dtp=UNIT/VREF
                    if z and z['n']>0:
                        n=z['n']; mu=z['sf']/n; row.update(f_media=mu,f_dp=float(np.sqrt(max(0.0,z['sf2']/n-mu*mu))),f_min=z['mn'],t_cong=z['n_lo']*dtp,frac_cong=z['n_lo']/n,t_parado=z['n_vlo']*dtp,maxbloco_cong=z['maxrun']*dtp,n_paradas=z['cross']//2,fld_medio=z['sfld']/n,fld_max=z['mxfld'],**{'f_t%d'%(k+1):(z['sf3'][k]/z['n3'][k] if z['n3'][k] else np.nan) for k in range(3)})
                rows.append(row)
        return pd.DataFrame(rows)
    # passe 0: inicio nominal (fim - L/vref); passe 1: inicio = fim - duracao simulada do passe 0 (regredindo do MVT)
    st={q:(a['ini'] if a['tipo']=='arr' else a['fim']-a['tot']/VREF) for q,a in enumerate(ag)}
    gmul=[gm.get(a['i'],1.0) if a['tipo']=='dep' else 1.0 for a in ag]; emul=[em_.get(a['i'],1.0) if a['tipo']=='dep' else 1.0 for a in ag]
    d0=passe(ag,Dia['ads'],Dia['t0'],pr,K,st,gmul,emul); st={q:(a['ini'] if a['tipo']=='arr' else a['fim']-d0.get(q,a['tot']/VREF)) for q,a in enumerate(ag)}
    d1=passe(ag,Dia['ads'],Dia['t0'],pr,K,st,gmul,emul); rows=[]
    for q,a in enumerate(ag):
        if a['tipo']=='dep': rows.append(dict(day=day,i=a['i'],id=Dia['D'][a['i']]['id'],L=a['tot'],dur=d1.get(q,np.nan),dur_livre=a['tot']/VREF,guia=a['guia'],fim=a['fim']))
    return pd.DataFrame(rows)
if __name__=='__main__':
    day=sys.argv[1]; pr=dict(X=float(sys.argv[2])*1000,gamma=float(sys.argv[3]),tau=float(sys.argv[4]),wads=float(sys.argv[5])); import time; t=time.time(); r=simula_dia((day,pr)); print('dia',day,'| partidas simuladas',len(r),'| duracao livre media %.0f s | com rastros media %.0f s | razao media %.3f | %.0f s'%(r.dur_livre.mean(),r.dur.mean(),(r.dur/r.dur_livre).mean(),time.time()-t))
