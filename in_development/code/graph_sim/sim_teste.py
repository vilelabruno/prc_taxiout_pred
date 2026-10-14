import sys,time,numpy as np,pandas as pd; sys.path.insert(0,'src/sincronia_grafo')
import sim_dia as SD, sim_run as SR
t=time.time(); d=SD.Dia('EHAM',sys.argv[1]); D,A,Oobs,own=SR.prepara(d)
print('preparo: %.0f s | partidas simulaveis %d | com cadeia desde o MVT %d | observadas (ADS) %d | chegadas simulaveis %d (observadas %d)'%(time.time()-t,sum(r['ok'] for r in D),sum(r['chain'] for r in D),sum(r['obs'] is not None for r in D),sum(r['ok'] for r in A),sum(r['obs'] is not None for r in A)))
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
for modo in ('S','A'):
    for g_ in (0.0,1.0):
        t=time.time(); S,rho,rm_=SR.simula(d,D,A,Oobs,own,modo,0,0,1.0,g_); x=SR.resultado(d,D,S,modo)
        print(f'modo {modo} gamma={g_} T=0 R=0: n={len(x)} | RMSE de y (constante ajustada no dia) {rm(x.res):.1f} | ocupacao media dos outros {np.mean(rm_[rm_>0]) if (rm_>0).any() else 0:.2f} | {time.time()-t:.0f} s')
