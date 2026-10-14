"""Separa MVT - 1o movimento em tempo andando (gs>3 kt) e parado, por voo com trilha desde o stand."""
import sys,os,numpy as np,pandas as pd
D=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo')
apt=sys.argv[1]; days=sys.argv[2].split(',')
o=pd.read_parquet(f'{D}/sim/{apt}.parquet'); o=o[o.adsb & o.stand_start.astype('boolean').fillna(False).astype(bool) & o.day.isin(days)]
R=[]
for day,od in o.groupby('day'):
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    for r in od.itertuples():
        v=grp[r.cs]; v=v[(v.t>=r.t_move)&(v.t<=r.mvt)&(v.alt==0)]
        if len(v)<5: continue
        t=np.r_[v.t.to_numpy(),r.mvt]; gs=v.gs.fillna(0).to_numpy(); dt=np.diff(t); dt=np.minimum(dt,60)
        R.append(dict(cs=r.cs,rwy=r.rwy,L=r.L,tm=r.mvt-r.t_move,mov=dt[gs>3].sum(),par=dt[gs<=3].sum(),cob=dt.sum()/(r.mvt-r.t_move)))
R=pd.DataFrame(R); R=R[R.cob>0.8]
c=lambda x,y:np.corrcoef(x,y)[0,1]
print(apt,len(R),'tm med %.0f | andando med %.0f sd %.0f | parado med %.0f sd %.0f | corr(L,andando) %.2f corr(L,parado) %.2f | var(tm) explicada por parado %.0f%%'%(
 R.tm.median(),R.mov.median(),R.mov.std(),R.par.median(),R.par.std(),c(R.L,R.mov),c(R.L,R.par),100*np.cov(R.tm,R.par)[0,1]/R.tm.var()))
print('   velocidade andando (L/andando) q10/50/90',(R.L/R.mov).quantile([.1,.5,.9]).round(1).tolist())
