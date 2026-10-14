"""Espera escondida H = y - (MVT - 1o ping) por contexto (stand, dia da semana, hora, companhia, tipo) com encolhimento hierarquico (Ridge com escala por fator = media posterior de efeitos aleatorios gaussianos)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from sklearn.linear_model import Ridge
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store')
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/{APT}/cenario_b_oof.parquet')
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{S}/sincronia_grafo/fisico/{APT}_*.parquet'))]).drop(columns=['day','mvt'])
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt']).rename(columns={'MVT_ID_mvt':'id'})
d=o.merge(f,on='id',how='left').merge(tr,on='id',how='left'); d['d']=pd.to_datetime(d.day); t=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True)
d['hora']=t.dt.hour.astype(str); d['dow']=t.dt.dayofweek.astype(str); d['stand']=d.STAND_mvt.astype(str); d['oper']=d.AIRCRAFT_OPERATOR_flt.astype(str).str[:8]; d['acft']=d.AIRCRAFT_TYPE_mvt.astype(str); d['pier']=d.stand.str.extract(r'^(\D*\d?)')[0].fillna('?')
d['hxd']=d.hora+'_'+(d.dow.astype(int)>=5).astype(int).astype(str); d['rwy']=d.RUNWAY_mvt.astype(str)
mv=pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True)
for nm,c in (('ms','SCHED_TIME_UTC_mvt'),('ma','AOBT_3_flt'),('me','EOBT_1_flt')):
    d[nm]=(mv-pd.to_datetime(d[c],utc=True,errors='coerce')).dt.total_seconds(); d[nm+'_nan']=d[nm].isna().astype(float); d[nm]=d[nm].fillna(0)
d['tem']=d.tem.fillna(0); tk=d[(d.tem==1)&d.mvt_prim.notna()&(d.y>0)].reset_index(drop=True)
tk['H']=tk.y-tk.mvt_prim
print(APT,'voos com 1o ping:',len(tk),'| H: media',round(tk.H.mean()),'mediana',round(tk.H.median()),'desvio',round(tk.H.std()),'| RMSE do campeao neles',round(float(np.sqrt(((tk.y-tk.champ)**2).mean())),1))
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
FACT=['stand','dow','hora','oper','acft','pier','hxd','rwy']; NUM=['D_rota','tau_acc','gs_first','d0','esp_fora','esp_pat','mvt_ult']
for c in NUM: tk[c]=tk[c].fillna(tk[c].median())
days=np.array(sorted(tk.d.unique())); tk['blk']=tk.d.map({x:i//5 for i,x in enumerate(days)})
def design(df_tr,df_te,scales,numeric):
    Xs_tr=[];Xs_te=[]
    for k,s in scales.items():
        if s<=0: continue
        cats=pd.Categorical(df_tr[k]); idx={c:i for i,c in enumerate(cats.categories)}
        def oh(df):
            ii=df[k].map(idx); m=ii.notna().to_numpy(); r=np.where(m)[0]; return sparse.csr_matrix((np.ones(m.sum())*s,(r,ii[m].astype(int).to_numpy())),shape=(len(df),len(idx)))
        Xs_tr.append(oh(df_tr)); Xs_te.append(oh(df_te))
    if numeric:
        mu=df_tr[NUM].mean(); sd=df_tr[NUM].std().replace(0,1)
        Xs_tr.append(sparse.csr_matrix(((df_tr[NUM]-mu)/sd).to_numpy())); Xs_te.append(sparse.csr_matrix(((df_te[NUM]-mu)/sd).to_numpy()))
    return sparse.hstack(Xs_tr).tocsr(),sparse.hstack(Xs_te).tocsr()
def cv(scales,numeric=True,alpha=1.0):
    pred=np.zeros(len(tk))
    for b in tk.blk.unique():
        te=(tk.blk==b).to_numpy(); lo=tk.d[te].min()-pd.Timedelta(days=1); hi=tk.d[te].max()+pd.Timedelta(days=1); trn=((tk.d<lo)|(tk.d>hi)).to_numpy()
        Xtr,Xte=design(tk[trn],tk[te],scales,numeric); m=Ridge(alpha=alpha).fit(Xtr,tk.H[trn]); pred[te]=m.predict(Xte)
    return pred
def score(p): return rm(tk.H-p)

CLK=['ms','ma','me','ms_nan','ma_nan','me_nan']
SC={'stand':0.1,'dow':0.0,'hora':0.4,'oper':0.1,'acft':0.2,'pier':0.0,'hxd':0.0,'rwy':0.05}
def run(num,sc,nome):
    global NUM
    NUM=num
    for c in NUM: tk[c]=tk[c].fillna(tk[c].median())
    p=cv(sc); return p
print('\nRMSE de y nos voos com 1o ping (campeao 214.8), blocos fora:')
p0=run(['D_rota','tau_acc','gs_first','d0','esp_fora','esp_pat','mvt_ult'],SC,'a'); print('  numericos + contexto:           ',round(rm(tk.H-p0),1))
p1=run(['D_rota','tau_acc','gs_first','d0','esp_fora','esp_pat','mvt_ult']+CLK,SC,'b'); print('  + relogios (MVT-SCHED/AOBT/EOBT):',round(rm(tk.H-p1),1))
p2=run(['D_rota','tau_acc','gs_first','d0','esp_fora','esp_pat','mvt_ult']+CLK,{k:0.0 for k in SC},'c'); print('  + relogios, SEM contexto:        ',round(rm(tk.H-p2),1))
p3=run(CLK,{k:0.0 for k in SC},'d'); print('  so relogios:                     ',round(rm(tk.H-p3),1))
tk['yhat']=tk.mvt_prim+p1
for w in (0,.3,.5,.7,1): print(f'  mistura w_campeao={w}: RMSE {rm(tk.y-(w*tk.champ+(1-w)*tk.yhat)):.1f}')
