"""Teste: a fila REAL na cabeceira (avioes parados, do ADS-B dos OUTROS voos, perto do no-ancora da pista do voo) nos 15 min antes do MVT explica o excesso y - exot_graf?
Features por partida: n_par = no. de outros avioes com ping parado (gs<2 kt) a <=R m da ancora da pista em [MVT-900, MVT]; t_par = soma dos segundos parados; n_par_5 = idem em [MVT-300, MVT]."""
import sys,os,pickle,glob,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
os.environ['APT']='EGLL'; sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
import sim_dia as SD
R=float(sys.argv[1]) if len(sys.argv)>1 else 500.
F=pd.read_parquet('work/ablation/b/feat_b_v1_EGLL.parquet'); T=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','RUNWAY_mvt']); F=F.merge(T,on='MVT_ID_mvt'); F['y']=F.TAXITIME_SEC_mvt
days=sorted(d for d in F.day.unique() if d.startswith('2025-01'))[:10]; rows=[]
for day in days:
    d=SD.Dia('EGLL',day); g=d.g; raw=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==','EGLL')]); raw=raw[raw.alt==0].sort_values('t'); raw['x']=(raw.lon-g.lon0)*g.kx; raw['y']=(raw.lat-g.lat0)*g.ky
    nx=np.r_[raw.t.to_numpy()[1:],raw.t.to_numpy()[-1]+30]; same=np.r_[raw.icao24.to_numpy()[1:]==raw.icao24.to_numpy()[:-1],False]; raw['w']=np.where(same,np.minimum(nx-raw.t.to_numpy(),30.0),10.0)
    Dia=pickle.load(open(os.path.expanduser(f'~/Projetos/oof_store/sincronia_grafo/rotas_EGLL/{day}.pkl'),'rb')); cache={}
    for r in Dia['D']:
        if not r['ok']: continue
        rw=r['rwy']; an=d.anchor(rw)
        if an is None: continue
        if rw not in cache:
            xy=g.xy[an]; s=raw[(raw.gs<2)&(np.hypot(raw.x-xy[0],raw.y-xy[1])<=R)]; cache[rw]=(s.t.to_numpy(),s.icao24.to_numpy(),s.w.to_numpy())
        t,ic,w=cache[rw]; a=np.searchsorted(t,r['mvt']-900); b=np.searchsorted(t,r['mvt']); a5=np.searchsorted(t,r['mvt']-300); m=ic[a:b]!=r.get('icao'); own=r.get('icao')
        rows.append(dict(MVT_ID_mvt=r['id'],n_par=len(set(ic[a:b][m])),t_par=float(w[a:b][m].sum()),n_par_5=len(set(ic[a5:b][ic[a5:b]!=own]))))
Q=pd.DataFrame(rows).merge(F[['MVT_ID_mvt','y','exot_graf','t_cong','f_media','cadeia_ads','tem_ads']],on='MVT_ID_mvt'); Q=Q[(Q.y>=833)&(Q.y<=2108)].copy(); Q['exc']=Q.y-Q.exot_graf
print('R=%.0f m | voos %d | dias %d'%(R,len(Q),len(days)))
print('Spearman com y:   ',Q[['n_par','t_par','n_par_5','t_cong','f_media','exot_graf']].corrwith(Q.y,method='spearman').round(3).to_dict())
print('Spearman com excesso (y - exot_graf):',Q[['n_par','t_par','n_par_5','t_cong','f_media']].corrwith(Q.exc,method='spearman').round(3).to_dict())
print('mediana do excesso por faixa de t_par (quartis):'); print(Q.groupby(pd.qcut(Q.t_par.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])).agg(n=('y','size'),t_par=('t_par','median'),y=('y','median'),exc=('exc','median')).round(0).to_string())
