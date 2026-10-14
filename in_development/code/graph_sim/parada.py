"""Duracao da parada entre o pushback (1a corrida) e o taxi proprio (2a corrida) + estado da frota no BLOCK e no inicio da parada."""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from regride import regua, proj, regride, Sync, T, KT
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/parada'); _C={}
def corridas(t,gs):
    mv=gs>=1; runs=[]; i=0; n=len(t)
    while i<n:
        if not mv[i]: i+=1; continue
        j=i
        while j+1<n and (mv[j+1] or (t[j+1]-t[j]<10 and np.any(mv[j+1:j+4]))): j+=1
        runs.append((i,j)); i=j+1
    return [x for x in runs if t[x[1]]-t[x[0]]>=5]
def setup(apt):
    if apt in _C: return _C[apt]
    sy=Sync(apt); idx=T.build_stand_index(sy.g.A)
    tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','RUNWAY_mvt','STAND_mvt'])
    tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
    tr['mvt']=pd.to_datetime(tr.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64'); tr['blk']=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
    tr['cs']=tr.CALLSIGN_flt.fillna(tr.FLIGHT_mvt).str.strip(); tr=tr.rename(columns={'RUNWAY_mvt':'rwy','STAND_mvt':'stand'}); _C[apt]=(sy,idx,tr); return _C[apt]
def estado(R,fl,me,s_self,sx):
    """features da frota (sem o proprio aviao) no minuto"""
    if fl is None: return dict(n_mov=0,n_ground=0,n_near=0,n_near_mov=0,n_ahead=0,n_ahead_stop=0,n_ahead_1k=0)
    X,gs,ic=fl; k=ic!=me; X,gs=X[k],gs[k]
    dst=np.hypot(*(X-sx).T) if len(X) else np.array([])
    out=dict(n_mov=int((gs>=1).sum()),n_ground=len(X),n_near=int((dst<300).sum()),n_near_mov=int(((dst<300)&(gs>=1)).sum()))
    if len(X):
        s,d=proj(R,X); ah=(d<=40)&(s>s_self+20)
        out.update(n_ahead=int(ah.sum()),n_ahead_stop=int((ah&(gs<1)).sum()),n_ahead_1k=int((ah&(s<s_self+1000)).sum()))
    else: out.update(n_ahead=0,n_ahead_stop=0,n_ahead_1k=0)
    return out
def dia(args):
    apt,day=args; out=f'{O}/{apt}_{day}.parquet'
    if os.path.exists(out): return out
    sy,idx,tr=setup(apt); g=sy.g; t0=pd.Timestamp(day,tz='UTC').value//10**9; tt=tr[(tr.mvt>=t0)&(tr.mvt<t0+86400)]
    a=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a['cs']=a.callsign.str.strip()
    gr=a[a.alt==0].copy(); gr['m']=(gr.t//60).astype(int); gr=gr.groupby(['m','icao24']).last().reset_index()
    gr['x']=(gr.lon-g.lon0)*g.kx; gr['y']=(gr.lat-g.lat0)*g.ky; gr['gs']=gr.gs.fillna(0)
    fleet={m:(np.c_[h.x.to_numpy(),h.y.to_numpy()],h.gs.to_numpy(),h.icao24.to_numpy()) for m,h in gr.groupby('m')}
    hod=gr.groupby('m').agg(n_mov=('gs',lambda s:(s>=1).sum()),n_ground=('gs','size')).reset_index(); hod['hour']=(hod.m*60%86400)//3600
    grp={k:v for k,v in a.groupby('cs')}; R=[]
    for r in tt.itertuples():
        v=grp.get(r.cs)
        if v is None: continue
        v=v[(v.t>r.mvt-5400)&(v.t<=r.mvt+60)]; gd=v[v.alt==0]
        if len(gd)<8 or r.mvt-gd.t.max()>30: continue
        me=gd.icao24.iloc[-1]; v=v[v.icao24==me]; gd=v[(v.alt==0)&(v.t<=r.mvt)].reset_index(drop=True)
        if len(gd)<8: continue
        Pp=np.c_[(gd.lon-g.lon0)*g.kx,(gd.lat-g.lat0)*g.ky]; t=gd.t.to_numpy().astype(float); gs=gd.gs.fillna(0).to_numpy()
        Rg=regua(sy,g,idx,r.stand,Pp)
        if Rg is None: continue
        res=regride(t,Pp,gs,Rg,r.mvt)
        if isinstance(res,dict) or res[0].get('cat') not in ('cruzou','cruzou0'): continue
        rec,s,ok=res
        if rec.get('gap',99)>30: continue
        rs=corridas(t,gs)
        if len(rs)<2: continue
        e1,s2=rs[0][1],rs[1][0]; ini1=t[rs[0][0]]-r.blk
        if not(-120<=ini1<=180): continue
        par=t[s2]-t[e1]
        if par>900: continue
        row=dict(day=day,hour=int((r.blk%86400)//3600),stand=r.stand,rwy=str(r.rwy),blk=r.blk,mvt=r.mvt,ini1=ini1,dur1=t[e1]-t[rs[0][0]],par=par,ini2=t[s2]-r.blk,y=r.mvt-r.blk,s_e1=float(s[e1]))
        for suf,T0,ss in (('b',r.blk,0.0),('s',t[e1],float(s[e1]))):
            e=estado(Rg,fleet.get(int(T0//60)),me,ss,Rg[4]); row.update({f'{k}_{suf}':val for k,val in e.items()})
        R.append(row)
    os.makedirs(O,exist_ok=True); pd.DataFrame(R).to_parquet(out); hod.assign(day=day).to_parquet(f'{O}/hod_{apt}_{day}.parquet'); return out
if __name__=='__main__':
    apt=sys.argv[1]; days=[f'2025-01-{d:02d}' for d in range(8,23)]+[f'2025-07-{d:02d}' for d in range(8,23)]
    with Pool(8) as p:
        for o in p.imap_unordered(dia,[(apt,d) for d in days]): pass
    print(apt,'ok')
