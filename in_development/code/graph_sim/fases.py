"""Fases da saida do stand a partir do perfil de velocidade (gs em kt): pushback (lento) -> taxi com motores proprios (gs>=VT)."""
import os,sys,numpy as np,pandas as pd
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/regride'); VT=float(sys.argv[2]) if len(sys.argv)>2 else 8.0
q=lambda x:' / '.join(f'{v:.0f}' for v in np.nanquantile(x,[.1,.5,.9]))
for apt in sys.argv[1].split(','):
    d=pd.read_parquet(f'{O}/{apt}.parquet'); c=d[d.cat.isin(['cruzou','cruzou0'])&(d.gap<=30)&d.arr.notna()]; R=[]
    for r in c.itertuples():
        a=np.array(r.arr).reshape(-1,3); t,s,gs=a.T; i0=int(np.searchsorted(t,r.t_exit)); 
        if i0>=len(t): continue
        k=np.where(gs[i0:]>=VT)[0]
        if not len(k): continue
        j=i0+int(k[0]); seg=slice(i0,j+1); dt=np.diff(t[seg]); g=gs[seg][:-1]
        R.append(dict(blk_exit=r.t_exit-r.blk,dur=t[j]-r.t_exit,ds=s[j]-30,v_med=np.median(gs[seg])*0.5144,par=dt[g<1].sum(),mov=dt[g>=1].sum(),vmax_push=np.percentile(gs[seg],90)))
    R=pd.DataFrame(R); R=R[(R.dur<900)&(R.dur>=0)]
    print(f'\n{apt} n={len(R)} (limiar de taxi proprio {VT} kt)'); 
    print(' duracao do trecho lento (t_exit -> gs>=limiar), q10/50/90 s:',q(R.dur),'| sd',round(R.dur.std()))
    print(' distancia percorrida nele, q10/50/90 m:',q(R.ds),'| sd',round(R.ds.std()))
    print(' velocidade mediana no trecho, q10/50/90 m/s:',' / '.join(f'{v:.1f}' for v in np.nanquantile(R.v_med,[.1,.5,.9])))
    print(' parado (<1 kt) dentro dele, q10/50/90 s:',q(R.par),'| andando (>=1 kt):',q(R.mov))
    print(' BLOCK -> t_exit sd',round(R.blk_exit.std()),'| duracao+BLOCK->exit sd',round((R.dur+R.blk_exit).std()))
