import sys,time,numpy as np,pandas as pd,pickle; sys.path.insert(0,'src/sincronia_grafo')
import sim_dia as SD, sim_run as SR, foco as F
DIAS=sys.argv[1].split(','); DAT={}
for dia in DIAS:
    d=SD.Dia('EHAM',dia); D,A,Oobs,own=SR.prepara(d); cand=[j for j in F.candidatos(d,D,A) if F.referencia(d,D,j) is not None]; DAT[dia]=(d,D,A,cand); print(dia,'partidas com ADS completo do stand ate o MVT e com rota recente disponivel:',len(cand),flush=True)
pr=dict(F.P_DEF); res=[]; GAM=(0.0,0.01,0.02,0.05,0.1)
def mede(dia,j,gammas=GAM):
    d,D,A,_=DAT[dia]; r=D[j]; ref=D[F.referencia(d,D,j)]; t_real=r['tstar']; t_alvo=r['mvt']-SD.ROLL_DEP
    # (D) Dijkstra + ADS ponto a ponto (rota real do voo)
    PD=r['pings']; segsD=F.segs_de_pings(d,r['rwy'],PD); gsD=np.hypot(*np.diff(PD,axis=0).T)/np.maximum(np.diff(r['tp']),1e-6); vD=float(np.quantile(gsD[(gsD>1)&(gsD<15)],0.8))
    # (R) rota recente: guia da referencia do ponto mais perto do MEU stand ate o fim, ligada ao meu stand
    PR=ref['pings']; iS=int(np.argmin(np.hypot(*(PR-r['xy']).T))); segsR=F.segs_de_pings(d,r['rwy'],PR[iS:]); gsR=np.hypot(*np.diff(PR,axis=0).T)/np.maximum(np.diff(ref['tp']),1e-6); vR=float(np.quantile(gsR[(gsR>1)&(gsR<15)],0.8))
    kS,fS=d.sy.snap_many(PR[iS:iS+1])
    leg=[]
    if kS[0]>=0:
        c,ed=d.caminho(r['rwy'],d.stand_info(r['stand'])['opts'],to_edge=(kS[0],fS[0])); leg=ed
    segsR=leg+segsR
    dep=F.deposito(d,D,A,t_real-1500,t_alvo+60,r['icao'],r['id'],pr['wads'])
    out=dict(dia=dia,j=j,t_real=t_real,t_alvo=t_alvo,dur_real=t_alvo-t_real,L_D=sum(s[1] for s in segsD),L_R=sum(s[1] for s in segsR),vD=vD,vR=vR,n_obs=dep['n_obs'],n_sim=dep['n_sim'],ref=F.referencia(d,D,j))
    for g_ in gammas:
        for nome,segs,v in (('D',segsD,vD),('R',segsR,vR)): out[f'{nome}_g{g_}']=F.prever_inicio(segs,v,g_,dep,d.dEE,t_alvo,t_real,pr)-t_real     # erro do inicio previsto (s): >0 = previu o inicio depois do real
    return out,(segsD,segsR,vD,vR,dep)
rng=np.random.default_rng(5); amostra=[]
for dia in DIAS:
    cand=DAT[dia][3]; amostra+= [(dia,int(j)) for j in rng.choice(cand,size=min(14,len(cand)),replace=False)]
for n,(dia,j) in enumerate(amostra):
    o,_=mede(dia,j); res.append(o)
    if n%8==7: print('  calibracao: %d/%d voos'%(n+1,len(amostra)),flush=True)
R_=pd.DataFrame(res); R_.to_csv('/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo/foco_calibra.csv',index=False)
print('\nERRO DO INICIO PREVISTO DO MOVIMENTO (s) em %d voos com ADS completo (previsto - real; positivo = previu depois do real)'%len(R_))
for nome in ('D','R'):
    for g_ in GAM:
        e=R_[f'{nome}_g{g_}']; print(f'  sim {nome} gamma={g_}: erro medio {e.mean():7.1f} s | erro absoluto medio {e.abs().mean():6.1f} s | mediana |erro| {e.abs().median():6.1f} s | RMSE {np.sqrt((e**2).mean()):6.1f} s')
pickle.dump(dict(amostra=amostra),open('/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo/foco_estado.pkl','wb'))
