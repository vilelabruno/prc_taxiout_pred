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

tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','ADEP_mvt','ADES_mvt','BLOCK_TIME_UTC_mvt','EOBT_1_flt'])
tr=tr[tr.ADEP_mvt=='EGLL'].copy(); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).astype('int64')//10**9 if False else pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['eobt']=pd.to_datetime(tr.EOBT_1_flt,utc=True,errors='coerce').dt.as_unit('s').astype('int64')
eo={(a,b):(c,d) for a,b,c,d in zip(tr.FLIGHT_mvt,tr.blk,tr.eobt,tr.ADES_mvt)}
fims=np.sort([ (r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-SC.ROLL_DEP) for r in Dia['D'] if r['ok'] and r['ed']])
rows=[]
for _,t in T.iterrows():
    i=int(t.i); r=Dia['D'][i]
    if not r['ok'] or not r['ed']: continue
    fim=r['tstar'] if (r['chain'] and r['tstar']==r['tstar']) else r['mvt']-SC.ROLL_DEP; tf=fim-t.dur; ready=r['blk']+120
    b_rel=int((tf-t0)//BIN); b_rdy=int((ready-t0)//BIN)
    if b_rel-b_rdy<2: continue
    bins=list(range(b_rdy,b_rel+1)); rs=risco_serie(r,i,bins)
    e=eo.get((r['cs'],int(r['blk'])),(np.nan,None))
    for n,(b,rk) in enumerate(zip(bins,rs)):
        tt=t0+b*BIN
        nd=np.searchsorted(fims,tt+Hs)-np.searchsorted(fims,tt)      # partidas previstas a chegar na pista nos proximos 11 min
        rows.append(dict(f=i,y=int(n==len(bins)-1),risk=rk,eobt_slack=(tt-e[0])/60 if e[0]==e[0] else np.nan,ndep=nd,t_rel=(tt-tf)/60,hr=((tt)%86400)/3600))
W=pd.DataFrame(rows); print('janelas',len(W),'voos',W.f.nunique(),'positivas',W.y.sum())
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
W=W.fillna(W.median()); 
def cv(cols):
    p=np.zeros(len(W))
    for a,b in GroupKFold(5).split(W,W.y,W.f):
        m=GradientBoostingClassifier(n_estimators=150,max_depth=3).fit(W.iloc[a][cols],W.y.iloc[a]); p[b]=m.predict_proba(W.iloc[b][cols])[:,1]
    W['p']=p; rk=W.groupby('f').apply(lambda g:(g.p<=g.p[g.y==1].iloc[0]).mean()*0+ (g.p>g.p[g.y==1].iloc[0]).mean()).mean()
    return roc_auc_score(W.y,p),rk
for nome,c in [('so risco local',['risk']),('risco+ndep',['risk','ndep']),('+EOBT slack',['risk','ndep','eobt_slack']),('+hora',['risk','ndep','eobt_slack','hr']),('so EOBT',['eobt_slack'])]:
    a,rk=cv(c); print('%-16s AUC janela %.3f | fracao de janelas do mesmo voo com p maior que a da liberacao %.2f (0 = liberacao e a mais provavel)'%(nome,a,rk))
