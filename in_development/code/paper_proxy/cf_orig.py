"""Indicador de sobrecarga SÓ NA ORIGEM: rota O-D amostrada de 0 a 100 km do aeroporto a cada 5 km (21 pts), A_avg = max_k (n_k+1)/MEAN_k (algoritmo do artigo, voo real removido da contagem),
voo deslocado de d = -30..+30 min em passos de 0,5 min (âncora MVT). Saída: ~/yearabl/orig/m{M}.parquet (MVT_ID_mvt, Ao_-30.0 ... Ao_+30.0). Uso: python cf_orig.py MES"""
import os,sys,numpy as np,pandas as pd
M=int(sys.argv[1]); D=os.path.expanduser("~/yearabl/"); T0=pd.Timestamp("2025-01-01",tz="UTC").timestamp(); ND=365; LAT0,LON0,NLA,NLO=28,-16,40,66; R=6371.0
OC=np.load(D+"OC.npy",mmap_mode="r").reshape(ND*96,NLA,NLO); MEAN=np.load(D+"MEAN.npy",mmap_mode="r")
ap=pd.read_csv(D+"airports.csv",usecols=["ident","latitude_deg","longitude_deg"]).drop_duplicates("ident").set_index("ident")
t=pd.read_parquet(D+"year_table.parquet",columns=["MVT_ID_mvt","mes","ADEP_mvt","ADES_mvt","mvt","ref","arr1"]); t=t[t.mes==M].reset_index(drop=True)
DS=np.arange(-30,30+1e-9,0.5); S=np.arange(0,100.1,5.0)
def cells(o,dst):
    a=ap.loc[o]; b=ap.loc[dst]; la1,lo1,la2,lo2=np.radians([a.latitude_deg,a.longitude_deg,b.latitude_deg,b.longitude_deg]); dlt=2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
    if dlt<1e-6: return None
    f=np.minimum(S/R/dlt,1.0); A=np.sin((1-f)*dlt)/np.sin(dlt); B=np.sin(f*dlt)/np.sin(dlt); x=A*np.cos(la1)*np.cos(lo1)+B*np.cos(la2)*np.cos(lo2); y=A*np.cos(la1)*np.sin(lo1)+B*np.cos(la2)*np.sin(lo2); z=A*np.sin(la1)+B*np.sin(la2)
    la=np.degrees(np.arctan2(z,np.hypot(x,y))); lo=np.degrees(np.arctan2(y,x)); ci=np.floor(la-LAT0).astype(int); cj=np.floor(lo-LON0).astype(int); ok=(ci>=0)&(ci<NLA)&(cj>=0)&(cj<NLO); return f[ok],ci[ok],cj[ok]
N=len(t); out=np.full((N,len(DS)),np.nan,np.float32)
for (o,dst),g in t.groupby(["ADEP_mvt","ADES_mvt"]):
    if o not in ap.index or dst not in ap.index: continue
    c=cells(o,dst)
    if c is None: continue
    f,ci,cj=c; ix=g.index.values; mvt=t.mvt.values[ix]; ref=t.ref.values[ix]; Tt=(np.where(np.isfinite(t.arr1.values[ix]),t.arr1.values[ix],ref+7200)-ref-900).clip(1200,60000)
    for j,d in enumerate(DS):
        ra=np.zeros(len(ix)); any_=np.zeros(len(ix),bool)
        for k in range(len(f)):
            tk=mvt+d*60+f[k]*Tt; x=(tk-T0)/900.0-0.5; b0=np.floor(x).astype(int); w=x-b0; b0c=np.clip(b0,0,ND*96-2)
            v=(1-w)*OC[b0c,ci[k],cj[k]]+w*OC[b0c+1,ci[k],cj[k]]; br=np.floor((mvt+f[k]*Tt-T0)/900.0).astype(int); v=np.maximum(v-((1-w)*(b0c==br)+w*((b0c+1)==br)),0)
            day=np.clip(((tk-T0)//86400).astype(int),0,ND-1); hr=np.clip((((tk-T0)%86400)//3600).astype(int),0,23); me=MEAN[day,hr,ci[k],cj[k]]; cov=(me>=2.0)&(b0>=0)&(b0+1<ND*96)
            ra=np.where(cov,np.maximum(ra,(v+1)/np.maximum(me,1.0)),ra); any_|=cov
        out[ix,j]=np.where(any_,ra,np.nan)
r=pd.DataFrame(out,columns=["Ao_%+.1f"%d for d in DS]); r.insert(0,"MVT_ID_mvt",t.MVT_ID_mvt.values); os.makedirs(D+"orig",exist_ok=True); r.to_parquet(D+f"orig/m{M}.parquet"); print("mes",M,N,"ok",flush=True)
