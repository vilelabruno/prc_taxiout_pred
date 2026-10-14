"""Regressao pelo ADS-B desde o MVT ate o stand.  1) anda para tras na trilha ate o ponto de maior aproximacao ao stand
no CAMINHO (para quando a distancia volta a subir -> nao pula o stand).  2) t_saida = ultimo ping a <=30 m do stand.
Teste: escondo ate X m (Dijkstra desde o stand) e estimo t_saida pelo 1o ping visivel: Dijkstra stand->ping + aceleracao desde 0."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
import taxigraph as T
KT=0.514444; R_STAND=60.; R_SAIDA=30.
def tau(D,v,a):
    da=v*v/(2*a)
    return v/a+(D-da)/v if D>da else 2*D/v
def regride(P,sp):
    """indices da trilha percorrida de tras para frente; devolve (j_stand, alcancou)"""
    d=np.hypot(*(P-sp).T); best=None; j=len(P)-1
    while j>=0:
        if best is None or d[j]<d[best]: best=j
        if d[best]<=R_STAND and d[j]>d[best]+25: break      # ja passou do stand e esta se afastando
        j-=1
    return (best if d[best]<=R_STAND else None), d
def dist_stand(sy,opts,k,f):
    """Dijkstra: stand (opcoes de acesso) -> ponto (aresta k, fracao f)"""
    L=sy.L[k]; best=np.inf
    for n,acc in opts:
        best=min(best,acc+f*L+sy.D[n,sy.ea[k]],acc+(1-f)*L+sy.D[n,sy.eb[k]])
    return best
if __name__=='__main__':
    apt=sys.argv[1]; days=sys.argv[2].split(','); sy=Sync(apt); g=sy.g; idx=T.build_stand_index(g.A); Rt=T.Router(g)
    tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt'])
    tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
    tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
    F=[];ST=[]
    for day in days:
        t0=pd.Timestamp(day,tz='UTC').value//10**9
        a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
        grp={k:v for k,v in a.groupby('cs')}
        for r in tr[(tr.mvt>=t0+3600)&(tr.mvt<t0+23*3600)].itertuples():
            v=grp.get(r.cs); la,lo,lv=T.map_stand(r.STAND_mvt,idx)
            if v is None or lv>2: continue
            v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt)&(v.alt==0)]
            if len(v)<10: continue
            v=v[v.icao24==v.icao24.iloc[-1]]; v=v.reset_index(drop=True)
            P=np.c_[(v.lon-g.lon0)*g.kx,(v.lat-g.lat0)*g.ky]; sp=np.array(g.proj(la,lo)); t=v.t.to_numpy(); gs=v.gs.to_numpy()*KT
            j,d=regride(P,sp)
            ST.append(dict(alcanca=j is not None,n=len(v)))
            if j is None: continue
            # t_saida: ultimo ping a <=30 m do stand, a partir de j
            k=j
            while k+1<len(P) and d[k+1]<=R_SAIDA: k+=1
            t_saida=t[k]
            ST[-1].update(t_saida=t_saida,blk_to_saida=t_saida-r.blk,pings_no_stand=int((d[j:k+1]<=R_SAIDA).sum()),desl=float(np.hypot(*(P[k]-P[j]))) if k>j else 0.,
                          mvt_menos_saida=r.mvt-t_saida)
            opts,_=Rt.attach(la,lo)
            sn=sy.snap(P[k+1:]); 
            for X in (100,200,400,700,1000):
                for q,(ek,ef,_) in enumerate(sn):
                    i=k+1+q
                    if ek<0 or gs[i]!=gs[i] or gs[i]<0.8: continue
                    D=dist_stand(sy,opts,ek,ef)
                    if D>=X: break
                else: continue
                row=dict(X=X,D=D,v1=gs[i],true=t_saida-t[i],hid=t[i]-t_saida)
                for nm,aa in (('a05',.5),('a10',1.0),('a20',2.0)): row[nm]=-tau(D,gs[i],aa)
                row['const']=-D/gs[i]
                F.append(row)
    s=pd.DataFrame(ST); d=pd.DataFrame(F)
    print(apt,'voos',len(s),'| regressao alcanca o stand:',round(s.alcanca.mean(),3))
    z=s[s.alcanca]
    print('pings a <=30 m do stand: mediana',z.pings_no_stand.median(),'| com deslocamento >5 m (pushback?):',round((z.desl>5).mean(),3),
          '| BLOCK->saida (s) q10/50/90',z.blk_to_saida.quantile([.1,.5,.9]).round(0).tolist(),'| saida->MVT',z.mvt_menos_saida.quantile([.1,.5,.9]).round(0).tolist())
    rows=[]
    for X,q in d.groupby('X'):
        r={'X':X,'n':len(q),'D_med':q.D.median(),'v1_med':q.v1.median(),'t_escond_med':q.hid.median(),'t_escond_sd':q.hid.std()}
        r['base_const_rmse']=np.sqrt(((q.hid-q.hid.median())**2).mean())
        for m in ('const','a05','a10','a20'):
            e=q.true-q[m]; r[m+'_med']=e.median(); r[m+'_mad']=(e-e.median()).abs().median(); r[m+'_rmse']=np.sqrt((e**2).mean())
        rows.append(r)
    pd.set_option('display.width',260); print(pd.DataFrame(rows).round(1).to_string(index=False))
    d.to_csv(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/regressao_{apt}.csv'),index=False)
