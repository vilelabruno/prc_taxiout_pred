"""Teste falsificavel da probRisco: o risco do grafo vivo (rastros projetados ao longo da rota do voo, sem o proprio voo) e MAIOR nas janelas em que o aviao ESPEROU no stand do que na janela em que ele SAIU?
Trafego do dia reconstruido por oraculo (horarios reais dos outros voos, progresso uniforme ao longo da rota). Janelas de 30 s."""
import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_campo as SC
OUT=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); DAY=sys.argv[1]; X=float(sys.argv[2])*1000 if len(sys.argv)>2 else 1000.; Hs=float(sys.argv[3]) if len(sys.argv)>3 else 660.; BIN=30.
G=SC.graf(); sy=G['sy']; dEE=G['dEE']; K=np.clip(1-dEE/X,0,None); K[~np.isfinite(K)]=0; nE=K.shape[0]
Dia=pickle.load(open(f'{SC.R0}/{DAY}.pkl','rb')); S=pickle.load(open(f'{OUT}/solver_rev_{DAY}.pkl','rb')); T=S['T'].copy(); T['dur']=S['dur']; t0=Dia['t0']; vf=Dia['vref']; nT=int(86400/BIN)+200
def entradas(ed,t_ini,t_fim):
    """presenca uniforme ao longo das arestas entre t_ini e t_fim: lista (bin, aresta, peso)"""
    Ls=np.array([e[1] for e in ed],float); ks=np.array([e[0] for e in ed]); tot=Ls.sum(); out=[]
    if tot<=0 or t_fim<=t_ini: return out
    cum=np.r_[0,np.cumsum(Ls)]/tot; ts=t_ini+cum*(t_fim-t_ini)
    for k,a,b in zip(ks,ts[:-1],ts[1:]):
        if k<0: continue
        for bn in range(int((a-t0)//BIN),int((b-t0)//BIN)+1):
            w=(min(b,t0+(bn+1)*BIN)-max(a,t0+bn*BIN))/BIN
            if w>0 and 0<=bn<nT: out.append((bn,k,w))
    return out
O=np.zeros((nE,nT),np.float32); own={}
for i,r in enumerate(Dia['D']):
    if not r['ok'] or not r['ed']: continue
    L=sum(e[1] for e in r['ed']); fim=r['mvt']-SC.ROLL_DEP; en=entradas(r['ed'],fim-L/vf,fim); own[('d',i)]=en
    for bn,k,w in en: O[k,bn]+=w
for j,a in enumerate(Dia['A']):
    if not a['ok'] or not a['ed']: continue
    en=entradas(a['ed'],a['t_in'],a['t_out']); own[('a',j)]=en
    for bn,k,w in en: O[k,bn]+=w
OS=K@O                                              # rastro espalhado pelo grafo ate X, para todas as arestas e janelas
print('DIA',DAY,'| X=%.1f km | horizonte H=%.0f s | trafego oraculo: %d partidas + %d chegadas'%(X/1000,Hs,sum(k[0]=='d' for k in own),sum(k[0]=='a' for k in own)))
nH=int(Hs/BIN); wH=1-np.arange(nH)/nH
def risco_serie(r,i,bins):
    """risco de liberar o voo i na janela b: media, pelo comprimento, do rastro espalhado nas arestas da rota, no instante em que ele estaria la (b + tempo ate a aresta / BIN), com peso 1-d/H, sem o proprio voo"""
    ed=[e for e in r['ed'] if e[0]>=0]; Ls=np.array([e[1] for e in ed],float); ks=np.array([e[0] for e in ed]); tau=(np.cumsum(Ls)-Ls/2)/vf/BIN
    mine=own.get(('d',i),[]); by={}
    for bn,k,w in mine: by.setdefault(bn,[]).append((k,w))
    out=[]
    for b in bins:
        idx=(b+tau).astype(int); idx=np.clip(idx,0,nT-1); tot=(Ls*OS[ks,idx]).sum()/Ls.sum()
        prop=0.0
        for e_,bn in zip(range(len(ks)),idx):
            for (k2,w2) in by.get(bn,[]): prop+=Ls[e_]*w2*K[ks[e_],k2]
        out.append(tot-prop/Ls.sum())
    return np.array(out)
# ---- voos do solver: janelas entre o instante em que ficou pronto (BLOCK + 120 s de pushback) e a liberacao (t_frente = fim - duracao simulada)
rows=[]; curvas={}
for _,t in T.iterrows():
    i=int(t.i); r=Dia['D'][i]
    if not r['ok'] or not r['ed']: continue
    fim=r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-SC.ROLL_DEP; tf=fim-t.dur; ready=r['blk']+120
    b_rel=int((tf-t0)//BIN); b_rdy=int((ready-t0)//BIN)
    if b_rel-b_rdy<2: continue                                   # sem espera mensuravel: nao ha janela segurada
    bins=list(range(b_rdy,b_rel+1)); rs=risco_serie(r,i,bins); rows.append(dict(cs=r['cs'],espera=(tf-ready)/60,risco_liberacao=rs[-1],risco_espera_medio=rs[:-1].mean(),risco_espera_max=rs[:-1].max(),n_jan=len(bins)))
    if r['cs']=='BAW105': curvas['BAW105']=(np.array(bins)*BIN+t0-r['mvt'])/60,rs
R=pd.DataFrame(rows); print('voos com espera mensuravel (>= 1 min entre pronto e liberado): %d de %d'%(len(R),len(T)))
print('risco medio NA JANELA DE LIBERACAO: %.3f | risco medio DURANTE a espera: %.3f | maximo durante a espera: %.3f'%(R.risco_liberacao.mean(),R.risco_espera_medio.mean(),R.risco_espera_max.mean()))
print('fracao dos voos em que o risco medio durante a espera e MAIOR que o da liberacao: %.0f%%'%(100*(R.risco_espera_medio>R.risco_liberacao).mean()))
# AUC: janelas de espera (0) contra janela de liberacao (1), usando -risco
from itertools import islice
pos=R.risco_liberacao.values; neg=R.risco_espera_medio.values; auc=np.mean([(p<n)+0.5*(p==n) for p in pos for n in neg[:400]]); print('AUC de "risco baixo => liberou" (voo a voo, liberacao x espera media): %.2f  (0.5 = sem informacao)'%auc)
print('correlacao entre a duracao da espera e o risco medio durante a espera: %.2f | e o risco na liberacao: %.2f'%(np.corrcoef(R.espera,R.risco_espera_medio)[0,1],np.corrcoef(R.espera,R.risco_liberacao)[0,1]))
if 'BAW105' in curvas:
    m,rs=curvas['BAW105']; print('\nBAW105: risco por janela (minutos relativos ao MVT; a liberacao e a ultima):'); print('  '+' | '.join('%.1f: %.2f'%(a,b) for a,b in zip(m[::4],rs[::4])),'| liberacao %.1f: %.2f'%(m[-1],rs[-1]))
