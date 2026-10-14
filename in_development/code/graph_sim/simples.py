"""Simulador simples: y = c + L/v.  L = Dijkstra stand->cabeceira (Router); v = velocidade media empregada
(sum ds / sum dt do casamento ADS-B<->grafo) na mesma pista: janela de +-30 min do dia-alvo, encolhida para o mes (dias a +-1 do alvo purgados).
c = media de (y - L/v_mes) nos outros dias (mesma purga).  O alvo nunca contribui com a propria trilha."""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import taxigraph as T
OBS=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/obs'); K=600.0; HW=1800.0
def comprimentos(apt,fl):
    g=T.TaxiGraph(apt); Rt=T.Router(g); idx=T.build_stand_index(g.A); cache={}; L=[]
    for s,r in zip(fl.stand,fl.rwy):
        k=(s,r)
        if k not in cache:
            la,lo,lv=T.map_stand(s,idx); v=np.nan
            if lv<9:
                try:
                    o=Rt.route(la,lo,r); v=o['len_full'] if o else np.nan
                except Exception: v=np.nan
            cache[k]=v
        L.append(cache[k])
    return np.array(L)
def rodar(apt):
    fl=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{OBS}/{apt}_*.parquet'))],ignore_index=True)
    fl['y']=fl.mvt-fl.blk; fl=fl[(fl.y>0)&(fl.y<7200)].copy()
    fl['L']=comprimentos(apt,fl); fl['d']=pd.to_datetime(fl.day); fl['mes']=fl.day.str[:7]
    ref=fl[fl.adsb&(fl.dt_taxi>30)&(fl.ds_taxi>200)&(fl.ds_taxi/fl.dt_taxi<20)].copy()
    out=[]
    for mes,F in fl.groupby('mes'):
        Rm=ref[ref.mes==mes]; days=sorted(F.d.unique())
        sd=Rm.groupby(['rwy','d'])[['ds_taxi','dt_taxi']].sum()
        # velocidade global (todas as pistas) como ultimo recurso
        for dd in days:
            lo,hi=dd-pd.Timedelta(days=1),dd+pd.Timedelta(days=1)
            tgt=F[F.d==dd]; trn=F[(F.d<lo)|(F.d>hi)]
            rm=Rm[(Rm.d<lo)|(Rm.d>hi)]
            vm_all=rm.ds_taxi.sum()/max(rm.dt_taxi.sum(),1)
            vmr=(rm.groupby('rwy').ds_taxi.sum()/rm.groupby('rwy').dt_taxi.sum())
            # c: residuo nos dias de treino usando v_mes por pista (mesmo conjunto rm)
            vt=trn.rwy.map(vmr).fillna(vm_all); ok=np.isfinite(trn.L)
            c=float((trn.y[ok]-trn.L[ok]/vt[ok]).mean()); y_const=float(trn.y.mean())
            Rd=Rm[Rm.d==dd]
            for r in tgt.itertuples():
                vm=vmr.get(r.rwy,vm_all)
                h=Rd[(Rd.rwy==r.rwy)&(Rd.t_taxi_end>r.mvt-HW)&(Rd.t_taxi_end<r.mvt+HW)&(Rd.mvt!=r.mvt)]
                S,Tt=h.ds_taxi.sum(),h.dt_taxi.sum()
                vh=(S+K*vm)/(Tt+K) if Tt>0 else vm
                Lr=r.L if np.isfinite(r.L) else np.nan
                out.append(dict(apt=apt,mes=mes,day=r.day,y=r.y,L=Lr,vm=vm,vh=vh,n_h=len(h),c=c,y_const=y_const,
                    y_mes=c+Lr/vm,y_hora=c+Lr/vh,adsb=r.adsb,stand_start=getattr(r,'stand_start',False),ds=r.ds_taxi,dt=r.dt_taxi))
    return pd.DataFrame(out)
def rmse(e): e=e[np.isfinite(e)]; return float(np.sqrt((e**2).mean()))
if __name__=='__main__':
    R=[]
    for apt in sys.argv[1].split(','):
        d=rodar(apt); d.to_parquet(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/simples_{apt}.parquet')); R.append(d)
        ok=np.isfinite(d.L)
        print(apt,len(d),'L ok',round(ok.mean(),3),'| RMSE const',round(rmse(d.y-d.y_const),1),'mes',round(rmse((d.y-d.y_mes)[ok]),1),'hora',round(rmse((d.y-d.y_hora)[ok]),1),
              '| const(ok)',round(rmse((d.y-d.y_const)[ok]),1),flush=True)
