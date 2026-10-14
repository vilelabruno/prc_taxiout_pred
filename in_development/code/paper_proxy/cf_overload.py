"""Contrafactual de sobrecarga (algoritmo do artigo de Mas-Pujol & Delgado aplicado a decolagens deslocadas).
Ancora: MVT real. Para cada deslocamento d (min; negativo = sair mais cedo, positivo = mais tarde): o voo decolaria em MVT+d e, ao longo do círculo máximo O-D (40 amostras), estaria na célula c_k de 1° no instante MVT+d+f_k*T (T = duração planejada de voo = ARVT_1 − ref − 900 s).
 n_k = ocupação (aeronaves distintas na célula e janela de 15 min), interpolada linearmente no tempo entre bins vizinhos; no d=0 retira-se a presença do próprio voo (n−1); soma-se +1 (o voo hipotético).
 Algoritmo do artigo: razão contra a média (avg) e contra o máximo (max) dos 30 dias vizinhos na mesma hora; vale a célula MAIS CHEIA cruzada: A_avg(d) = max_k (n_k+1)/avg_k ; A_max(d) = max_k (n_k+1)/max_k ; só células com histórico >= 2 aeronaves/h.
Saída: ~/yearabl/cf/m{M}.parquet (chave MVT_ID_mvt; colunas A{avg|max}_{d}) para a mesma lista de voos da year_table.  Uso: python cf_overload.py M"""
import sys, os, numpy as np, pandas as pd
D=os.path.expanduser('~/yearabl/'); M=int(sys.argv[1]); T0=pd.Timestamp('2025-01-01',tz='UTC').timestamp(); ND=365; LAT0,LON0,NLA,NLO=28,-16,40,66
SH=[-30,-20,-15,-10,-5,-2,-1,0,1,2,5,10,15,20,30,45,60]
t=pd.read_parquet(D+'year_table.parquet',columns=['MVT_ID_mvt','mes','ADEP_mvt','ADES_mvt','mvt','ref','arr1']); t=t[t.mes==M].reset_index(drop=True)
OC=np.load(D+'OC.npy',mmap_mode='r').reshape(ND*96,NLA,NLO); MEAN=np.load(D+'MEAN.npy',mmap_mode='r'); MAXX=np.load(D+'MAXX.npy',mmap_mode='r')
ap=pd.read_csv(D+'airports.csv',usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident')
def cells(o,dst):
    a=ap.loc[o]; b=ap.loc[dst]; la1,lo1,la2,lo2=np.radians([a.latitude_deg,a.longitude_deg,b.latitude_deg,b.longitude_deg]); f=np.linspace(0,1,40); dlt=2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
    if dlt<1e-6: return None
    A=np.sin((1-f)*dlt)/np.sin(dlt); B=np.sin(f*dlt)/np.sin(dlt); x=A*np.cos(la1)*np.cos(lo1)+B*np.cos(la2)*np.cos(lo2); y=A*np.cos(la1)*np.sin(lo1)+B*np.cos(la2)*np.sin(lo2); z=A*np.sin(la1)+B*np.sin(la2)
    la=np.degrees(np.arctan2(z,np.hypot(x,y))); lo=np.degrees(np.arctan2(y,x)); ci=np.floor(la-LAT0).astype(int); cj=np.floor(lo-LON0).astype(int); ok=(ci>=0)&(ci<NLA)&(cj>=0)&(cj<NLO); return f[ok],ci[ok],cj[ok]
N=len(t); out={f'A{k}_{d}':np.full(N,np.nan,np.float32) for k in ('avg','max') for d in SH}
for (o,dst),g in t.groupby(['ADEP_mvt','ADES_mvt']):
    if o not in ap.index or dst not in ap.index: continue
    c=cells(o,dst)
    if c is None: continue
    f,ci,cj=c; ix=g.index.values; mvt=t.mvt.values[ix]; ref=t.ref.values[ix]; Tt=(np.where(np.isfinite(t.arr1.values[ix]),t.arr1.values[ix],ref+7200)-ref-900).clip(1200,60000)
    for d in SH:
        ra=np.zeros(len(ix)); rm=np.zeros(len(ix)); any_=np.zeros(len(ix),bool)
        for k in range(len(f)):
            tk=mvt+d*60+f[k]*Tt; x=(tk-T0)/900.0-0.5; b0=np.floor(x).astype(int); w=x-b0; okb=(b0>=0)&(b0+1<ND*96); b0c=np.clip(b0,0,ND*96-2)
            v=(1-w)*OC[b0c,ci[k],cj[k]]+w*OC[b0c+1,ci[k],cj[k]]
            if d==0: v=np.maximum(v-1.0,0.0)
            day=np.clip(((tk-T0)//86400).astype(int),0,ND-1); hr=np.clip((((tk-T0)%86400)//3600).astype(int),0,23); me=MEAN[day,hr,ci[k],cj[k]]; mx=MAXX[day,hr,ci[k],cj[k]]; cov=(me>=2.0)&okb
            ra=np.where(cov,np.maximum(ra,(v+1)/np.maximum(me,1.0)),ra); rm=np.where(cov,np.maximum(rm,(v+1)/np.maximum(mx,1.0)),rm); any_|=cov
        out[f'Aavg_{d}'][ix]=np.where(any_,ra,np.nan); out[f'Amax_{d}'][ix]=np.where(any_,rm,np.nan)
r=pd.DataFrame(out); r.insert(0,'MVT_ID_mvt',t.MVT_ID_mvt.values); os.makedirs(D+'cf',exist_ok=True); r.to_parquet(D+f'cf/m{M}.parquet'); print('mes',M,len(r),'ok',flush=True)
