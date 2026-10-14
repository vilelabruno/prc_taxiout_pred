import sys,time,numpy as np,pandas as pd; sys.path.insert(0,'src/sincronia_grafo')
import sim_dia as SD, sim_run as SR, foco as F
DIAS=['2025-07-15','2025-07-18','2025-07-31']; DAT={}
for dia in DIAS:
    d=SD.Dia('EHAM',dia); D,A,Oobs,own=SR.prepara(d); cand=[j for j in F.candidatos(d,D,A) if F.referencia(d,D,j) is not None]; DAT[dia]=(d,D,A,cand)
rng=np.random.default_rng(5); amostra=[]
for dia in DIAS: amostra+=[(dia,int(j)) for j in rng.choice(DAT[dia][3],size=min(4,len(DAT[dia][3])),replace=False)]
print('voos na varredura:',len(amostra),flush=True)
def prepara_voo(dia,j):
    d,D,A,_=DAT[dia]; r=D[j]; ref=D[F.referencia(d,D,j)]; PD=r['pings']; segsD=F.segs_de_pings(d,r['rwy'],PD); gsD=np.hypot(*np.diff(PD,axis=0).T)/np.maximum(np.diff(r['tp']),1e-6); vD=float(np.quantile(gsD[(gsD>1)&(gsD<15)],.8))
    PR=ref['pings']; iS=int(np.argmin(np.hypot(*(PR-r['xy']).T))); segsR=F.segs_de_pings(d,r['rwy'],PR[iS:]); gsR=np.hypot(*np.diff(PR,axis=0).T)/np.maximum(np.diff(ref['tp']),1e-6); vR=float(np.quantile(gsR[(gsR>1)&(gsR<15)],.8)); kS,fS=d.sy.snap_many(PR[iS:iS+1])
    if kS[0]>=0: c,ed=d.caminho(r['rwy'],d.stand_info(r['stand'])['opts'],to_edge=(kS[0],fS[0])); segsR=ed+segsR
    return d,D,A,r,segsD,vD,segsR,vR
PV=[prepara_voo(*a) for a in amostra]
CFG={'base (T=15 min, alcance 100->600 m, peso ADS 3)':dict(),'duracao maior (T=30 min)':dict(T=1800.0),'duracao menor (T=5 min)':dict(T=300.0),'alcance maior (300->1000 m)':dict(R0=300.0,Rmax=1000.0,vexp=120.0),'alcance menor (50->200 m)':dict(R0=50.0,Rmax=200.0,vexp=20.0),'peso ADS 10':dict(wads=10.0),'peso ADS 1 (igual ao simulado)':dict(wads=1.0)}
GAM=(0.0,0.002,0.005,0.01,0.03); out=[]
for nome,over in CFG.items():
    pr=dict(F.P_DEF); pr.update(over); E={('D',g):[] for g in GAM}; E.update({('R',g):[] for g in GAM})
    for (d,D,A,r,segsD,vD,segsR,vR) in PV:
        t_real=r['tstar']; t_alvo=r['mvt']-SD.ROLL_DEP; dep=F.deposito(d,D,A,t_real-1500,t_alvo+60,r['icao'],r['id'],pr['wads'])
        for g_ in GAM:
            for nm,segs,v in (('D',segsD,vD),('R',segsR,vR)): E[(nm,g_)].append(F.prever_inicio(segs,v,g_,dep,d.dEE,t_alvo,t_real,pr)-t_real)
    for nm in ('D','R'):
        m=[np.mean(np.abs(E[(nm,g_)])) for g_ in GAM]; b=int(np.argmin(m)); out.append(dict(cfg=nome,sim=nm,erro_g0=m[0],melhor_g=GAM[b],erro_melhor=m[b],ganho=m[b]-m[0]))
    print(nome,'| D: g0 %.1f -> melhor %.1f (gamma %s) | R: g0 %.1f -> melhor %.1f (gamma %s)'%(out[-2]['erro_g0'],out[-2]['erro_melhor'],out[-2]['melhor_g'],out[-1]['erro_g0'],out[-1]['erro_melhor'],out[-1]['melhor_g']),flush=True)
pd.DataFrame(out).to_csv('/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo/foco_varredura.csv',index=False); print('FIM')
