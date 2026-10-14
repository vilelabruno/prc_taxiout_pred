"""Tabela de features de 2026 (ranking.parquet, 344.841 partidas) com as MESMAS colunas/grupos do year_table de 2025, exceto Rede_grade e Fila_simulada (sem 2026 / sem ganho). pn = previsão da v98."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0,os.path.expanduser('~/yearabl'))
import rwy_feats as RF, reorg_feats as RG, ctot_feats as CF
D=os.path.expanduser('~/yearabl/'); T0=pd.Timestamp('2025-01-01',tz='UTC').timestamp()
raw=pd.read_parquet(os.path.expanduser('~/prc/data/ranking.parquet')); dep=raw[raw.PHASE_mvt=='DEP'].copy().reset_index(drop=True); dep['MVT_ID_mvt']=dep.MVT_ID_mvt.astype('int64'); raw['MVT_ID_mvt']=raw.MVT_ID_mvt.astype('int64')
def ts(s):
    x=pd.to_datetime(s,utc=True,errors='coerce'); return pd.Series(x.astype('datetime64[s, UTC]').astype('int64').astype(float),index=x.index).where(x.notna()).values
d=dep[['MVT_ID_mvt','ADEP_mvt','ADES_mvt','WK_TBL_CAT_flt']].copy()
for c,n in [('MVT_TIME_UTC_mvt','mvt'),('AOBT_3_flt','aobt'),('EOBT_1_flt','eobt'),('SCHED_TIME_UTC_mvt','sch'),('ARVT_1_flt','arr1')]: d[n]=ts(dep[c])
d['ma']=d.mvt-d.aobt; d['me']=d.mvt-d.eobt; d['ms']=d.mvt-d.sch; d['ref']=d.eobt.where(d.eobt.notna(),d.sch); d['noaobt']=d.ma.isna().astype(float)
dt=pd.to_datetime(d.mvt,unit='s',utc=True); d['hr']=dt.dt.hour; d['dow']=dt.dt.dayofweek; d['day']=dt.dt.strftime('%Y-%m-%d'); d['mes']=dt.dt.month
v=pd.read_parquet(D+'v98.parquet'); v['MVT_ID_mvt']=v.MVT_ID_mvt.astype('int64'); d=d.merge(v.rename(columns={'TAXITIME_SEC_mvt':'pn'}),on='MVT_ID_mvt',how='left'); d['ma_m_pn']=d.ma-d.pn
pf=pd.read_parquet(D+'pair_features.parquet')
def merge(df,name):
    global d
    df=df.copy(); df['MVT_ID_mvt']=df.MVT_ID_mvt.astype('int64'); cols=[c for c in df.columns if c not in ('MVT_ID_mvt','apt','rwy') and c not in d.columns]; d=d.merge(df[['MVT_ID_mvt']+cols].drop_duplicates('MVT_ID_mvt'),on='MVT_ID_mvt',how='left'); return cols
G={}
G['Pista_fila']=merge(RF.build(raw),'rwy'); G['Pista_reorg']=merge(RG.build(raw),'reorg'); G['CTOT_proxy_trafego']=merge(CF.build(raw),'ctot')
gr=dep[['MVT_ID_mvt','ADEP_mvt','STAND_mvt','RUNWAY_mvt']].merge(pf[['ADEP_mvt','STAND_mvt','RUNWAY_mvt','stand_lvl','len_full','len_int','n_turns','n_cross','n_nodes','access_m']],on=['ADEP_mvt','STAND_mvt','RUNWAY_mvt'],how='left').drop(columns=['ADEP_mvt','STAND_mvt','RUNWAY_mvt'])
G['Dijkstra_stand_pista']=merge(gr,'dij')
ab=pd.read_parquet(os.path.expanduser('~/prc/data/adsb2026/adsb_2026_st.parquet')); b0=pd.read_parquet(D+'adsb_basic_allmonths.parquet'); ab=ab[list(b0.columns)]; G['ADSB_antigo_basico']=merge(ab,'adsb')
w=[]
for a_ in d.ADEP_mvt.unique():
    mm=pd.read_csv(D+'metar/%s.csv'%a_,comment='#',low_memory=False); mm['t']=pd.to_datetime(mm.valid,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float); mm=mm.sort_values('t')[['t','vsby','sknt','gust','tmpf']]
    s=d[d.ADEP_mvt==a_][['MVT_ID_mvt','ref']].sort_values('ref'); w.append(pd.merge_asof(s,mm,left_on='ref',right_on='t',direction='nearest',tolerance=3600).drop(columns=['t','ref']))
d=d.merge(pd.concat(w),on='MVT_ID_mvt',how='left'); G['Clima_origem']=['vsby','sknt','gust','tmpf']
d['apt']=d.ADEP_mvt; d['wk']=d.WK_TBL_CAT_flt.astype(str)
# demanda (mesma regra do combine.py, só com os dias do próprio ranking)
d['di']=((d.ref-T0)//86400).astype(int)
def rel30(df,key,tcol,hcol,out):
    g=df.groupby([key,tcol,hcol]).size().rename('n').reset_index(); res=[]
    for k,s in g.groupby([key,hcol]):
        s=s.sort_values(tcol); ds=s[tcol].values; n=s.n.values.astype(float); r=np.full(len(s),np.nan); rw=np.full(len(s),np.nan); dwk=ds%7
        for i in range(len(s)):
            m=(np.abs(ds-ds[i])<=15)&(ds!=ds[i]); r[i]=n[i]/max(n[m].mean(),0.5) if m.sum()>=5 else np.nan
            m2=(np.abs(ds-ds[i])<=28)&(ds!=ds[i])&(dwk==dwk[i]); rw[i]=n[i]/max(n[m2].mean(),0.5) if m2.sum()>=2 else np.nan
        res.append(pd.DataFrame({key:k[0],tcol:ds,hcol:k[1],out:r,out+'_wk':rw,'n_'+out:n}))
    return pd.concat(res)
r1=rel30(d,'ADEP_mvt','di','hr','dep_rel'); d=d.merge(r1,on=['ADEP_mvt','di','hr'],how='left')
d['arr_h']=((d.arr1//3600)%24).fillna(-1).astype(int); d['adi']=((d.arr1-T0)//86400)
dd=d.dropna(subset=['adi']).copy(); dd['adi']=dd.adi.astype(int); r2=rel30(dd,'ADES_mvt','adi','arr_h','arr_rel'); d['adi']=d.adi.astype('Int64'); r2['adi']=r2.adi.astype('Int64'); d=d.merge(r2,on=['ADES_mvt','adi','arr_h'],how='left')
G['Demanda_aeroporto']=['dep_rel','dep_rel_wk','n_dep_rel','arr_rel','arr_rel_wk','n_arr_rel']
# códigos categóricos iguais aos do treino
yt=pd.read_parquet(D+'year_table.parquet',columns=['apt','apt_c','wk','wk_c']).drop_duplicates(); am=dict(zip(yt.apt,yt.apt_c)); wm=dict(zip(yt.wk,yt.wk_c)); d['apt_c']=d.apt.map(am); d['wk_c']=d.wk.map(wm)
G['CLK']=['pn','ma','ms','me','hr','dow','ma_m_pn','noaobt','apt_c','wk_c']; d=d.drop(columns=['WK_TBL_CAT_flt'])
d.to_parquet(D+'rank_table.parquet'); json.dump(G,open(D+'rank_groups.json','w'),ensure_ascii=False)
print('linhas',len(d),'| únicas',d.MVT_ID_mvt.nunique(),'| pn NaN',int(d.pn.isna().sum()),'| wk_c NaN',int(d.wk_c.isna().sum()),'| apt_c NaN',int(d.apt_c.isna().sum()))
print({k:(len(v),round(float(d[v[0]].notna().mean()),2)) for k,v in G.items()})
