"""Tabela v0 de features do motor de grafo, por voo (partida), 1 aeroporto x 365 dias de 2025. Parametros FIXOS (EGLL 2025-01-17), sem guia de rota recente, sem calibracao. Chave MVT_ID_mvt.
Nunca usa BLOCK, TAXITIME, y, AOBT, ARVT_3. Usa MVT/stand/pista, trafego de terceiros (MVT dos outros, ADS-B) e o ADS-B do proprio voo ate o MVT."""
import os, sys, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
os.environ['SEM_GUIA']='1'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import sim_campo as SC
from multiprocessing import Pool
APT=SC.APT; OUT=os.path.expanduser('~/Projetos/prc_data_challenge/work/ablation/b')
import json
SUF=os.environ.get('SUF','v0'); _pj=f'{OUT}/params_{APT}.json'
PR=dict(X=3995.0,theta=10.55,tau=318.9,wads=0.3026,p=1.7576,q=1.4117,rev=True,stats=True,gm={},em={})
if SUF!='v0' and os.path.exists(_pj): PR.update(json.load(open(_pj))['params'])
def dia(day):
    r=SC.simula_dia((day,PR))
    if len(r)==0: return r
    Dia=pickle.load(open(f'{SC.R0}/{day}.pkl','rb')); D=Dia['D']; ex=[]
    for i,fim in zip(r.i,r.fim):
        d=D[int(i)]; e=dict(rota_ok=1,tem_ads=int(d['obs'] is not None),cadeia_ads=int(bool(d['chain'])),t_ads_obs=max(0.0,d['mvt']-SC.ROLL_DEP-fim) if d['chain'] else 0.0)
        if d['chain'] and d.get('pings') is not None and len(d['tp'])>=3:
            P=np.asarray(d['pings']); t=np.asarray(d['tp'],float); dt=np.diff(t); ok=dt>0
            v=np.hypot(*np.diff(P,axis=0).T)[ok]/dt[ok] if ok.any() else np.array([])
            if len(v): e.update(v_ads_med=float(np.median(v)),v_ads_dp=float(v.std()),t_parado_ads=float(dt[ok][v<1.0].sum()),maxgap_ads=float(dt.max()),n_pings_ads=int(len(t)))
        ex.append(e)
    r=pd.concat([r.reset_index(drop=True),pd.DataFrame(ex)],axis=1); r['dur_oculto']=r.dur; r['exot_graf']=r.dur+r.t_ads_obs; r['apt']=APT
    r['mvt_s']=[D[int(i)]['mvt'] for i in r.i]
    return r.rename(columns={'id':'MVT_ID_mvt','dur':'dur_sim_oculto'}).drop(columns=['i','fim','dur_oculto','guia'])
if __name__=='__main__':
    SC.graf(); os.makedirs(OUT,exist_ok=True)
    days=sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SC.R0}/20*.pkl'))
    with Pool(int(os.environ.get('NP','6'))) as p: rs=[x for x in p.imap(dia,days) if len(x)]
    df=pd.concat(rs,ignore_index=True); df.to_parquet(f'{OUT}/feat_b_{SUF}_{APT}.parquet'); print(APT,len(days),'dias',len(df),'voos','exot_graf med %.0f'%df.exot_graf.median(),'tem_ads %.2f'%df.tem_ads.mean(),flush=True)
