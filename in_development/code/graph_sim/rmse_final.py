"""Ancora na pista: sem cadeia de pings ate o MVT, o ponto inicial e a pista (lat/long do MVT). Rota do stand a pista por Dijkstra (Router) / velocidade media
(ultima hora na mesma pista + ultimo mes, dias a +-1 do alvo purgados) + constante de patio/pushback/espera aprendida nos outros dias."""
import sys, glob, os, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src/physical'); import taxigraph as T
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store')
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{S}/sincronia_grafo/regride/{APT}_*.parquet'))])
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','RUNWAY_mvt']).rename(columns={'MVT_ID_mvt':'id'})
d=pd.read_parquet(f'{S}/grafo_por_aeroporto/{APT}/cenario_b_oof.parquet').merge(f.drop(columns=['mvt','day']),on='id',how='left').merge(tr,on='id',how='left')
d['d']=pd.to_datetime(d.day); d['tem']=d.tem.fillna(0); d['rwy']=d.RUNWAY_mvt.astype(str)
g=T.TaxiGraph(APT); Rt=T.Router(g); idx=T.build_stand_index(g.A); cache={}
def L(s,r):
    k=(s,r)
    if k not in cache:
        la,lo,lv=T.map_stand(s,idx); v=np.nan
        if lv<9:
            try: o=Rt.route(la,lo,r); v=o['len_full'] if o else np.nan
            except Exception: v=np.nan
        cache[k]=v
    return cache[k]
d['L']=[L(s,r) for s,r in zip(d.stand.fillna('?'),d.rwy)]
print(APT,len(d),'| rota stand->pista calculada:',round(d.L.notna().mean(),3),'| mediana',round(d.L.median()),'m')
ref=d[(d.tem==1)&d.v_bar.notna()&(d.t_mov>=60)].copy(); vg=ref.v_bar.median()
d['mvt_t']=pd.to_datetime(d.day).astype('int64')//10**9*0+d.mvt.fillna(0) if False else 0
d['id_i']=np.arange(len(d))
mv=pd.read_parquet(f'{S}/grafo_por_aeroporto/{APT}/cenario_b_oof.parquet')[['id']]
# mvt de todos os voos
tt=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt']).rename(columns={'MVT_ID_mvt':'id'}); tt['mvt_s']=pd.to_datetime(tt.MVT_TIME_UTC_mvt,utc=True).dt.as_unit('s').astype('int64')
d=d.merge(tt[['id','mvt_s']],on='id',how='left')
ref=ref.merge(tt[['id','mvt_s']],on='id',how='left')
K=600.0; HW=1800.0; vh=np.zeros(len(d)); nh=np.zeros(len(d))
rbyd={k:v for k,v in ref.groupby('d')}
for dd,idxs in d.groupby('d').groups.items():
    lo,hi=dd-pd.Timedelta(days=1),dd+pd.Timedelta(days=1); rm_=ref[(ref.d<lo)|(ref.d>hi)]
    vmr=(rm_.groupby('rwy').apply(lambda q:(q.v_bar*q.t_mov).sum()/q.t_mov.sum())) if 'rwy' in rm_ else None
    rd=rbyd.get(dd)
    for i in idxs:
        r=d.loc[i]; vm=vmr.get(r.rwy,vg) if vmr is not None else vg
        if rd is None: vh[d.index.get_loc(i)]=vm; continue
        h=rd[(rd.rwy==r.rwy)&((rd.mvt_s-r.mvt_s).abs()<=HW)&(rd.id!=r.id)]
        S_=(h.v_bar*h.t_mov).sum(); T_=h.t_mov.sum(); vh[d.index.get_loc(i)]=(S_+K*vm)/(T_+K) if T_>0 else vm; nh[d.index.get_loc(i)]=len(h)
d['v_h']=np.clip(vh,1.5,12); d['n_h']=nh; d['rota_s']=d.L/d.v_h
rm=lambda e: float(np.sqrt(np.mean(np.square(e)))); days=np.array(sorted(d.d.unique()))
# constante de patio/pushback/espera aprendida nos outros dias (mediana de y - L/v)
c=pd.Series(np.nan,index=d.index)
for dd in days:
    te=d.d==dd; trn=d[(d.d<dd-pd.Timedelta(days=1))|(d.d>dd+pd.Timedelta(days=1))]; c[te]=(trn.y-trn.rota_s).median()
d['c']=c; d['y_fis0']=d.rota_s+d.c
ok=d.L.notna(); u=ok&(d.tem==0)
print('constante c (patio+pushback+espera+corrida): mediana',round(d.c.median()),'s')
for nome,m in (('todas com rota',ok),('SEM cadeia (ancora na pista)',u),('com cadeia',ok&(d.tem==1))):
    q=d[m]; e=q.y-q.y_fis0; print(f'{nome:30s} n={len(q):6d} | fisico RMSE {rm(e):6.1f} mediana|e| {e.abs().median():5.1f} | campeao {rm(q.y-q.champ):6.1f} mediana {(q.y-q.champ).abs().median():5.1f} | corr(fisico,y) {np.corrcoef(q.y_fis0,q.y)[0,1]:.3f}')
q=d[u]
for w in (.3,.5,.7,.9): print(f'  SEM cadeia, mistura w_campeao={w}: {rm(q.y-(w*q.champ+(1-w)*q.y_fis0)):.1f}',end='')
print('\n')

# ===== estimador completo: usa tudo (ADS-B encadeado ate o stand quando ha ping; ancora na pista quando nao) =====
d['chain']=(d.tem==1); v=d.v_bar.fillna(d.v_bar.median()).clip(1.5,12)
d['rR']=d.t_star-d.blk_rotulo; d['rM']=d.t_star-d.blk_rotulo-d.D_star/v
d['reached']=d.chain&(d.D_star<=150); d['meio']=d.chain&~(d.D_star<=150)
cR=pd.Series(np.nan,index=d.index); cM=pd.Series(np.nan,index=d.index)
for dd in days:
    te=d.d==dd; trn=d[(d.d<dd-pd.Timedelta(days=1))|(d.d>dd+pd.Timedelta(days=1))]
    cR[te]=trn[trn.reached].rR.median(); cM[te]=trn[trn.meio].rM.median()
yh=np.where(d.reached,d.mvt_s-(d.t_star-cR),np.where(d.meio,d.mvt_s-(d.t_star-d.D_star/v-cM),d.y_fis0))
d['yh']=yh; ok=d.L.notna()&np.isfinite(d.yh); q=d[ok]
print('\n=========== EGLL jan+jul 2025, todas as partidas com rota (n=%d) ==========='%len(q))
print('RMSE CAMPEAO: %.1f s'%rm(q.y-q.champ)); print('RMSE GRAFO (simulacao completa): %.1f s'%rm(q.y-q.yh))
for m,z in q.groupby('mes'): print('   %s  campeao %.1f | grafo %.1f  (n=%d)'%(m,rm(z.y-z.champ),rm(z.y-z.yh),len(z)))
d.to_parquet(os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/grafo_base_'+APT+'.parquet'))
