import sys,time,itertools,pickle,numpy as np,pandas as pd; sys.path.insert(0,'src/sincronia_grafo')
import sim_dia as SD, sim_run as SR
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
dias=['2025-07-15','2025-07-18','2025-07-31']; DS={}
for dia in dias:
    d=SD.Dia('EHAM',dia); D,A,Oobs,own=SR.prepara(d); DS[dia]=(d,D,A,Oobs,own); print('preparado',dia,flush=True)
def roda(modo,Tm,Rm,w,g_):
    out=[];base=[]
    for dia,(d,D,A,Oobs,own) in DS.items():
        S,rho,rmd=SR.simula(d,D,A,Oobs,own,modo,Tm,Rm,w,g_); x=SR.resultado(d,D,S,modo); out.append(x.res.to_numpy()); base.append((x.y-x.y.median()).to_numpy())
    return rm(np.concatenate(out)),rm(np.concatenate(base)),len(np.concatenate(out))
res=[]
for modo in ('S','A'):
    r0,b0,n=roda(modo,0,0,1.0,0.0); print(f'\n=== modo {modo} ({"sem ADS: tudo simulado" if modo=="S" else "com ADS: cadeia desde o MVT + rastros observados"}) | n={n} partidas em 3 dias | RMSE sem rastros (gamma=0): {r0:.1f} | RMSE da constante (y - mediana): {b0:.1f}',flush=True)
    for Tm,Rm,w,g_ in itertools.product((0,5,15,30),(0,150,400),((1.0,) if modo=='S' else (1.0,3.0,10.0)),(1.0,2.0,4.0)):
        r,_,_=roda(modo,Tm,Rm,w,g_); res.append(dict(modo=modo,T=Tm,R=Rm,w=w,g=g_,rmse=r,base=r0))
R=pd.DataFrame(res); R['ganho']=R.rmse-R.base; R.to_csv('/Users/bruno.quirino@latam.com/Projetos/oof_store/sincronia_grafo/sim_grade_EHAM.csv',index=False)
for modo in ('S','A'):
    q=R[R.modo==modo].sort_values('rmse'); print(f'\nmodo {modo}: 8 melhores combinacoes (T min, R m, peso ADS, gamma) e ganho sobre gamma=0'); print(q.head(8)[['T','R','w','g','rmse','ganho']].round(1).to_string(index=False)); print('pior:',q.tail(1)[['T','R','w','g','rmse','ganho']].round(1).to_dict('records')[0])
    print('efeito medio por duracao T (todas as outras combinacoes):',q.groupby('T').ganho.mean().round(2).to_dict(),'| por alcance R:',q.groupby('R').ganho.mean().round(2).to_dict(),'| por gamma:',q.groupby('g').ganho.mean().round(2).to_dict(),('| por peso ADS: '+str(q.groupby('w').ganho.mean().round(2).to_dict())) if modo=='A' else '')
pickle.dump(None,open('/tmp/_x','wb'))
