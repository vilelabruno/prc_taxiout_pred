"""Teste do trecho escondido: trilha comeca no stand; escondo os X m iniciais da rota; estimo t_move pelo 1o ponto visivel
(posicao na rota via Sync, velocidade gs do proprio ADS-B, aceleracao desde 0)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
import taxigraph as T
KT=0.514444
def tau(D,v,a,modo):
    if modo=='const': return D/v
    if modo=='tri': return 2*D/v
    da=v*v/(2*a)
    return v/a+(D-da)/v if D>da else 2*D/v
apt=sys.argv[1]; days=sys.argv[2].split(','); sy=Sync(apt); g=sy.g; idx=T.build_stand_index(g.A)
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip()
F=[];PRE=[];GS=[]
for day in days:
    t0=pd.Timestamp(day,tz='UTC').value//10**9
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    grp={k:v for k,v in a.groupby('cs')}
    for r in tr[(tr.mvt>=t0+3600)&(tr.mvt<t0+23*3600)].itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt)&(v.alt==0)]
        if len(v)<10: continue
        v=v[v.icao24==v.icao24.iloc[-1]]; seg=(v.t.diff().fillna(0)>120).cumsum(); v=v[seg==seg.iloc[0]].reset_index(drop=True)
        la,lo,lv=T.map_stand(r.STAND_mvt,idx)
        if lv>2: continue
        P=np.c_[(v.lon-g.lon0)*g.kx,(v.lat-g.lat0)*g.ky]; d0=np.hypot(*(P[0]-np.array(g.proj(la,lo))))
        if d0>60: continue                       # trilha realmente comeca no stand
        t=v.t.to_numpy(); gs=v.gs.to_numpy()*KT
        if np.isnan(gs).mean()>0.2: continue
        sn=sy.snap(P); ed=np.array([x[0] for x in sn])
        if (ed<0).mean()>0.3: continue
        # ancora do alvo: ultimo ponto parado (gs<0.5 m/s) antes do 1o movimento sustentado
        mv=np.where(gs>1.0)[0]
        if not len(mv): continue
        i0=mv[0]; tm=t[i0]
        # pushback: pings antes do movimento proprio, com deslocamento
        pre=P[:i0+1]; PRE.append(dict(n_pre=i0,desl_pre=float(np.hypot(*(pre[-1]-pre[0]))) if i0>0 else 0.,dt_pre=t[i0]-t[0],d0=d0,blk_to_first=t[0]-r.blk,blk_to_move=tm-r.blk))
        ok=ed>=0; nodes=sy.route(list(ed[ok])); S,_,_=sy.along(nodes,P); S=np.maximum.accumulate(S)
        # gs vs deslocamento
        dd=np.hypot(*np.diff(P,axis=0).T)/np.maximum(np.diff(t),1e-6); m=(np.diff(t)<=6)&(gs[1:]>1)
        if m.any(): GS.append(np.median((gs[1:][m]/np.maximum(dd[m],1e-6))))
        for X in (100,200,400,700,1000):
            k=np.where(S-S[i0]>=X)[0]
            if not len(k) or k[0]<=i0+1: continue
            i1=k[0]; D=S[i1]-S[i0]; v1=gs[i1]
            if v1<0.8: continue
            row=dict(X=X,D=D,v1=v1,true=tm-t[i1])
            for modo,aa in (('const',0),('tri',0),('acc05',.5),('acc10',1.0)):
                row[modo]=-tau(D,v1,aa if aa else 1,'tri' if modo=='tri' else 'const' if modo=='const' else 'acc')
            F.append(row)
d=pd.DataFrame(F); p=pd.DataFrame(PRE)
print(apt,'voos com trilha no stand:',len(p),'| gs/deslocamento (mediana):',round(float(np.median(GS)),2))
print('antes do movimento: pings mediana',p.n_pre.median(),'| com >=2 pings antes e desl>5m:',round(((p.n_pre>=2)&(p.desl_pre>5)).mean(),3),'| desl mediano (m)',round(p.desl_pre.median(),1))
print('BLOCK->1o ping (s) quantis 10/50/90',p.blk_to_first.quantile([.1,.5,.9]).round(0).tolist(),' BLOCK->mov',p.blk_to_move.quantile([.1,.5,.9]).round(0).tolist())
rows=[]
for X,q in d.groupby('X'):
    r={'X':X,'n':len(q),'D_med':q.D.median(),'v1_med':q.v1.median(),'t_escondido_med':-q.true.median()*-1}
    for modo in ('const','tri','acc05','acc10'):
        e=q.true-q[modo]*-1*-1   # erro = real - estimado  (estimado = -tau)
        e=q.true-q[modo]; r[modo+'_med']=e.median(); r[modo+'_mad']=(e-e.median()).abs().median(); r[modo+'_rmse']=np.sqrt((e**2).mean())
    rows.append(r)
pd.set_option('display.width',250); print(pd.DataFrame(rows).round(1).to_string(index=False))
d.to_csv(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/trecho_{apt}.csv'),index=False)
