"""Auditoria de alinhamento lat/long e temporal nos voos de maior erro (EGLL, dias com ADS-B). Erro = y - (exot_graf + 45 s); controle = voos de erro pequeno.
Mede por voo: (T1) t(80 kt) - MVT; (T2) 1o movimento a partir do stand - BLOCK; (E1) menor distancia ping-stand perto do BLOCK; (E2) mediana da distancia dos pings ao grafo entre BLOCK e MVT;
(P1) maior parada (deslocamento <30 m) entre BLOCK e MVT e onde ocorre (no stand / no grafo)."""
import sys,os,glob,pickle,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src/sincronia_grafo'); sys.path.insert(0,'src/physical')
os.environ['APT']='EGLL'
import sim_dia as SD
F=pd.read_parquet('work/ablation/b/feat_b_v0_EGLL.parquet'); T=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt'])
F=F.merge(T,on='MVT_ID_mvt'); F=F[(F.TAXITIME_SEC_mvt>833)&(F.TAXITIME_SEC_mvt<2108)&(F.cadeia_ads==1)].copy(); F['err']=F.TAXITIME_SEC_mvt-(F.exot_graf+45)
days=[d for d in sorted(F.day.unique()) if d.startswith('2025-01')][:8]; F=F[F.day.isin(days)]
rows=[]
for day in days:
    d=SD.Dia('EGLL',day); g=d.g; Dia=pickle.load(open(f'{os.path.expanduser("~/Projetos/oof_store/sincronia_grafo/rotas_EGLL")}/{day}.pkl','rb'))
    raw=pd.read_parquet(f'data/processed/adsb_surface/{day}.parquet',filters=[('apt','==','EGLL')]); raw['x']=(raw.lon-g.lon0)*g.kx; raw['y']=(raw.lat-g.lat0)*g.ky
    for r in F[F.day==day].itertuples():
        D=[x for x in Dia['D'] if x['id']==r.MVT_ID_mvt][0]; ic=D.get('icao')
        if ic is None or not D['ok']: continue
        q=raw[(raw.icao24==ic)&(raw.t>=D['blk']-900)&(raw.t<=D['mvt']+240)].sort_values('t')
        if len(q)<5: continue
        sxy=np.asarray(D['xy']); gs=q.gs.to_numpy(); t=q.t.to_numpy(); P=q[['x','y']].to_numpy(); ds=np.hypot(*(P-sxy).T)
        a80=q[(q.gs>=80)&(q.t>=D['mvt']-300)]; T1=(a80.t.iloc[0]-D['mvt']) if len(a80) else np.nan
        near=(t>=D['blk']-300)&(t<=D['blk']+900); E1=ds[near].min() if near.any() else np.nan
        mv=np.where((ds>25)&(t>=D['blk']-300))[0]; T2=(t[mv[0]]-D['blk']) if len(mv) else np.nan
        w=(t>=D['blk'])&(t<=D['mvt']); Pw=P[w]; tw=t[w]; best=(0.0,np.nan)
        if len(Pw)>2:
            i=0
            for j in range(len(Pw)):
                while np.hypot(*(Pw[j]-Pw[i]))>30: i+=1
                if tw[j]-tw[i]>best[0]: best=(tw[j]-tw[i],float(np.hypot(*(Pw[i]-sxy))))
        dk,_=d.sy.tree.query(Pw) if len(Pw) else (np.array([np.nan]),0)
        rows.append(dict(day=day,id=r.MVT_ID_mvt,cs=D['cs'],stand=D['stand'],y=r.TAXITIME_SEC_mvt,err=r.err,T1_t80_menos_mvt=T1,T2_mov_menos_block=T2,E1_dist_stand=E1,E2_dist_grafo_med=float(np.nanmedian(dk)),P1_parada_s=best[0],T3_1o_ping_menos_block=float(t[t>=D['blk']-60][0]-D['blk']) if (t>=D['blk']-60).any() else np.nan,T4_tstar_menos_block=D['tstar']-D['blk'],P1_dist_ao_stand=best[1],n=len(q)))
R=pd.DataFrame(rows); R['grupo']=np.where(R.err.abs()>=R.err.abs().quantile(0.95),'maior erro',np.where(R.err.abs()<=R.err.abs().quantile(0.5),'controle','meio'))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print('voos auditados',len(R),'| dias',len(days))
print(R.groupby('grupo')[['err','T1_t80_menos_mvt','T2_mov_menos_block','E1_dist_stand','E2_dist_grafo_med','P1_parada_s','P1_dist_ao_stand']].median().round(1))
print('\nfracao com sinal de desalinhamento (por grupo):')
R['flag_T1']=R.T1_t80_menos_mvt.abs()>60; R['flag_T2']=R.T2_mov_menos_block.abs()>120; R['flag_E1']=R.E1_dist_stand>40; R['flag_E2']=R.E2_dist_grafo_med>25
print(R.groupby('grupo')[['flag_T1','flag_T2','flag_E1','flag_E2']].mean().round(2))
R['parada_no_stand']=(R.P1_parada_s>300)&(R.P1_dist_ao_stand<40); R['parada_fora']=(R.P1_parada_s>300)&(R.P1_dist_ao_stand>=40)
print('\nparada >5 min:'); print(R.groupby('grupo')[['parada_no_stand','parada_fora']].mean().round(2))
R.to_csv('work/ablation/b/audit_voos_EGLL.csv',index=False)
print('\nprimeiro ping do voo depois do BLOCK (s):'); print(R.groupby('grupo')[['T3_1o_ping_menos_block','T4_tstar_menos_block']].median().round(0)); R['ping_tardio']=R.T3_1o_ping_menos_block>300; print('fracao com 1o ping >5 min depois do BLOCK:'); print(R.groupby('grupo').ping_tardio.mean().round(2)); print('correlacao err x (1o ping - BLOCK): %.2f'%R[['err','T3_1o_ping_menos_block']].corr(method='spearman').iloc[0,1])
print('\n12 maiores erros:'); print(R[R.grupo=='maior erro'].sort_values('err',key=abs,ascending=False).head(12)[['day','cs','stand','y','err','T1_t80_menos_mvt','T2_mov_menos_block','E1_dist_stand','E2_dist_grafo_med','P1_parada_s','P1_dist_ao_stand']].round(0).to_string(index=False))
