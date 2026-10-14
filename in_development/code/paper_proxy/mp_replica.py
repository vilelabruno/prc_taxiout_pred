"""Réplica (com proxies) do desenho de Mas-Pujol & Delgado 2024: features estáticas + demanda de aeroporto normalizada pela história da mesma hora + demanda de rede (OC/EC máx e média nas células cruzadas pela rota O-D) normalizada pela história da mesma hora.
Rota: círculo máximo O-D em células de 1° (proxy da rota histórica do PREDICT). Ocupação: aeronaves distintas por célula e bin de 15 min (enroute/occ_*.parquet, 62 dias, as duas faixas somadas).
Rótulo 'regulado-like': L15 = AOBT - EOBT_1 >= 15 min (atraso de pré-partida). Normalização: média e máximo dos OUTROS dias do mesmo mês, mesma hora (leave-one-day-out).  CV por dia."""
import glob, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/'; S='/private/tmp/claude-502/-Users-bruno-quirino-latam-com-Desktop-bruno-Projetos-prc-data-challenge/5e41970f-79c3-434a-8d98-364545fe860d/scratchpad/'
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','ARVT_1_flt','SCHED_TIME_UTC_mvt','WK_TBL_CAT_flt']
raw=pd.concat([pd.read_parquet(R+'data/raw/training_2025-0%d-01_2025-0%d-01.parquet'%(a,b),columns=cols) for a,b in [(1,2),(7,8)]]).reset_index(drop=True); d=raw[raw.PHASE_mvt=='DEP'].copy().reset_index(drop=True)
def ts(s):
    x=pd.to_datetime(s,utc=True,errors='coerce'); return pd.Series(x.astype('datetime64[s, UTC]').astype('int64').astype(float),index=x.index).where(x.notna()).values
for c,n in [('MVT_TIME_UTC_mvt','mvt'),('AOBT_3_flt','aobt'),('EOBT_1_flt','eobt'),('ARVT_1_flt','arr1'),('SCHED_TIME_UTC_mvt','sch')]: d[n]=ts(d[c])
d['ref']=d.eobt.where(d.eobt.notna(),d.sch); d=d[d.ref.notna()].reset_index(drop=True)
d['L15']=((d.aobt-d.eobt)>=900)&d.aobt.notna()&d.eobt.notna()
dt=pd.to_datetime(d.ref,unit='s',utc=True); d['hr']=dt.dt.hour; d['dow']=dt.dt.dayofweek; d['mes']=dt.dt.month; d['day']=dt.dt.strftime('%Y-%m-%d'); d['b15']=((d.ref%86400)//900).astype(int)
# aeroportos: tamanho por nº de voos
ap=pd.read_csv(R+'data/external/ourairports/airports.csv',usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident')
allf=raw.copy(); cnt=pd.concat([allf.ADEP_mvt,allf.ADES_mvt]).value_counts(); size=pd.cut(cnt,[0,500,5000,1e9],labels=[0,1,2]).astype(float)
d['sz_o']=d.ADEP_mvt.map(size); d['sz_d']=d.ADES_mvt.map(size)
# --- demanda de aeroporto normalizada (partidas planejadas na mesma hora no aeroporto de origem; chegadas planejadas no destino quando conhecido)
dep_cnt=d.groupby(['ADEP_mvt','day','hr']).size().rename('n_dep').reset_index()
dd=dep_cnt.merge(dep_cnt.groupby(['ADEP_mvt','hr']).n_dep.agg(['sum','count']).rename(columns={'sum':'s','count':'c'}).reset_index(),on=['ADEP_mvt','hr'])
dd['dep_rel']=dd.n_dep/np.where((dd.c>1),((dd.s-dd.n_dep)/(dd.c-1)).clip(lower=0.5),np.nan)
d=d.merge(dd[['ADEP_mvt','day','hr','n_dep','dep_rel']],on=['ADEP_mvt','day','hr'],how='left')
d['arr_h']=((d.arr1//3600)%24).fillna(-1).astype(int); d['arr_day']=pd.to_datetime(d.arr1,unit='s',utc=True).dt.strftime('%Y-%m-%d')
arr_cnt=d.groupby(['ADES_mvt','arr_day','arr_h']).size().rename('n_arr').reset_index(); aa=arr_cnt.merge(arr_cnt.groupby(['ADES_mvt','arr_h']).n_arr.agg(['sum','count']).rename(columns={'sum':'s','count':'c'}).reset_index(),on=['ADES_mvt','arr_h'])
aa['arr_rel']=aa.n_arr/np.where(aa.c>1,((aa.s-aa.n_arr)/(aa.c-1)).clip(lower=0.5),np.nan)
d=d.merge(aa[['ADES_mvt','arr_day','arr_h','n_arr','arr_rel']],on=['ADES_mvt','arr_day','arr_h'],how='left')
# --- ocupação por célula/bin15/dia
files=sorted(glob.glob(S+'enroute/occ_*.parquet')); days=[f.split('occ_')[1][:-8] for f in files]; di={x:i for i,x in enumerate(days)}
LAT0,LON0=28,-16; NLA,NLO=40,66; OC=np.zeros((len(days),96,NLA,NLO),np.float32)
for f,dday in zip(files,days):
    o=pd.read_parquet(f); b=(o.bin%96).astype(int).values; la=(o.clat-LAT0).astype(int).values; lo=(o.clon-LON0).astype(int).values; ok=(la>=0)&(la<NLA)&(lo>=0)&(lo<NLO)
    np.add.at(OC[di[dday]],(b[ok],la[ok],lo[ok]),o.n.values[ok])
tot=OC.sum(0); mx1=OC.max(0); nD=len(days)
# rota por par O-D: células cruzadas e fração
def cells(o,dst):
    a=ap.loc[o]; b=ap.loc[dst]; la1,lo1,la2,lo2=np.radians([a.latitude_deg,a.longitude_deg,b.latitude_deg,b.longitude_deg]); n=40
    f=np.linspace(0,1,n); dlt=2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2)); 
    if dlt<1e-6: return None
    A=np.sin((1-f)*dlt)/np.sin(dlt); B=np.sin(f*dlt)/np.sin(dlt); x=A*np.cos(la1)*np.cos(lo1)+B*np.cos(la2)*np.cos(lo2); y=A*np.cos(la1)*np.sin(lo1)+B*np.cos(la2)*np.sin(lo2); z=A*np.sin(la1)+B*np.sin(la2)
    la=np.degrees(np.arctan2(z,np.hypot(x,y))); lo=np.degrees(np.arctan2(y,x)); ci=np.floor(la-LAT0).astype(int); cj=np.floor(lo-LON0).astype(int); ok=(ci>=0)&(ci<NLA)&(cj>=0)&(cj<NLO)
    return f[ok],ci[ok],cj[ok]
feat_avg=np.full(len(d),np.nan); feat_max=np.full(len(d),np.nan); feat_raw=np.full(len(d),np.nan); has_day=d.day.map(di).values
for (o,dst),g in d.groupby(['ADEP_mvt','ADES_mvt']):
    if o not in ap.index or dst not in ap.index: continue
    c=cells(o,dst)
    if c is None: continue
    f,ci,cj=c; ix=g.index.values; dix=has_day[ix]; ok=~np.isnan(dix)
    if ok.sum()==0: continue
    ix=ix[ok]; dix=dix[ok].astype(int); T=(np.where(np.isfinite(d.arr1.values[ix]),d.arr1.values[ix],d.ref.values[ix]+7200)-d.ref.values[ix]).clip(1800,60000)
    ravg=np.zeros(len(ix)); rmax=np.zeros(len(ix)); raw_=np.zeros(len(ix))
    for k in range(len(f)):
        t=d.ref.values[ix]+900+f[k]*(T-900); b=((t%86400)//900).astype(int); dd_=d.ref.values[ix]; 
        # dia da célula pode mudar; usa o mesmo dia
        cur=OC[dix,b,ci[k],cj[k]]; mean_oth=(tot[b,ci[k],cj[k]]-cur)/max(nD-1,1); mxx=mx1[b,ci[k],cj[k]]
        r_avg=cur/np.maximum(mean_oth,1.0); r_max=cur/np.maximum(mxx,1.0)
        ravg=np.maximum(ravg,r_avg); rmax=np.maximum(rmax,r_max); raw_=np.maximum(raw_,cur)
    feat_avg[ix]=ravg; feat_max[ix]=rmax; feat_raw[ix]=raw_
d['oc_avg_rel_max']=feat_avg; d['oc_max_rel_max']=feat_max; d['oc_raw_max']=feat_raw
d['ap_o']=d.ADEP_mvt.astype('category'); d['ap_d']=d.ADES_mvt.astype('category')
STAT=['hr','dow','mes','sz_o','sz_d','ap_o']; AP=['dep_rel','arr_rel','n_dep','n_arr']; NET=['oc_avg_rel_max','oc_max_rel_max','oc_raw_max']
m=d.L15.notna(); fold=d.day.map(dict(zip(d.day.unique(),np.random.default_rng(0).integers(0,5,d.day.nunique())))).values
def auc(cols,mask=None):
    mk=np.ones(len(d),bool) if mask is None else mask; p=np.zeros(len(d))
    for k in range(5):
        tr=(fold!=k)&mk; te=(fold==k)&mk; p[te]=lgb.LGBMClassifier(n_estimators=250,learning_rate=0.05,num_leaves=15,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1).fit(d.loc[tr,cols],d.L15[tr]).predict_proba(d.loc[te,cols])[:,1]
    return round(roc_auc_score(d.L15[mk],p[mk]),4)
print('linhas',len(d),'| L15 %.2f%%'%(100*d.L15.mean()),'| cobertura rede',round(float(d.oc_raw_max.notna().mean()),3),'| dep_rel',round(float(d.dep_rel.notna().mean()),3),'| arr_rel',round(float(d.arr_rel.notna().mean()),3))
print('AUC (CV por dia) do rótulo L15 = AOBT−EOBT ≥ 15 min:')
res={'estático (hora, dia, mês, tamanho, aeroporto)':auc(STAT),'+ demanda do aeroporto normalizada':auc(STAT+AP),'+ demanda de rede (OC rota) normalizada':auc(STAT+NET),'+ aeroporto + rede (desenho de Mas-Pujol)':auc(STAT+AP+NET)}
for k,v in res.items(): print(f'  {k:52s}{v}')
for ap_ in ['EGLL','LFPG','EDDF','LIRF']:
    mk=(d.ADEP_mvt==ap_).values; print(f'  só {ap_} (n={mk.sum()}): estático {auc(STAT[:-1],mk)} | +aeroporto+rede {auc(STAT[:-1]+AP+NET,mk)}')
d['sj']=np.nan
d[['MVT_ID_mvt','dep_rel','arr_rel','oc_avg_rel_max','oc_max_rel_max','oc_raw_max','L15']].to_parquet(S+'mp_feats.parquet')
