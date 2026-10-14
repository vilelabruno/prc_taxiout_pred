"""Indicador (Aavg, algoritmo do artigo, voo real removido) para decolagens de uma pista, deslocamento d de -30 a +30 min em passos de 0,5 min. Saídas: ~/yearabl/rwy/APT_RW_k{40,4}_mMM.parquet (k40 = rota inteira, k4 = 4 primeiras amostras = origem).
Uso: python cf_rwy.py MES APT RW [PASSO_MIN]"""
import os,sys,glob,numpy as np,pandas as pd
M=int(sys.argv[1]); APT,RW=sys.argv[2],sys.argv[3]; STEP=float(sys.argv[4]) if len(sys.argv)>4 else 0.5
D=os.path.expanduser("~/yearabl/"); T0=pd.Timestamp("2025-01-01",tz="UTC").timestamp(); ND=365; LAT0,LON0,NLA,NLO=28,-16,40,66
OC=np.load(D+"OC.npy",mmap_mode="r").reshape(ND*96,NLA,NLO); MEAN=np.load(D+"MEAN.npy",mmap_mode="r")
ap=pd.read_csv(D+"airports.csv",usecols=["ident","latitude_deg","longitude_deg"]).drop_duplicates("ident").set_index("ident")
raw=pd.read_parquet(glob.glob("/home/bruno_quirino_latam_com/prc/data/training_2025-%02d-01_*.parquet"%M)[0],columns=["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","RUNWAY_mvt"]); raw=raw[(raw.PHASE_mvt=="DEP")&(raw.ADEP_mvt==APT)&(raw.RUNWAY_mvt==RW)]
t=pd.read_parquet(D+"year_table.parquet",columns=["MVT_ID_mvt","mes","ADEP_mvt","ADES_mvt","mvt","ref","arr1","y","en","ma"]); t["MVT_ID_mvt"]=t.MVT_ID_mvt.astype("int64"); raw["MVT_ID_mvt"]=raw.MVT_ID_mvt.astype("int64")
t=t[(t.mes==M)].merge(raw[["MVT_ID_mvt"]],on="MVT_ID_mvt").reset_index(drop=True)
DS=np.arange(-30,30+1e-9,STEP); KM=[40,4]
def cells(o,dst):
    a=ap.loc[o]; b=ap.loc[dst]; la1,lo1,la2,lo2=np.radians([a.latitude_deg,a.longitude_deg,b.latitude_deg,b.longitude_deg]); f=np.linspace(0,1,40); dlt=2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
    if dlt<1e-6: return None
    A=np.sin((1-f)*dlt)/np.sin(dlt); B=np.sin(f*dlt)/np.sin(dlt); x=A*np.cos(la1)*np.cos(lo1)+B*np.cos(la2)*np.cos(lo2); y=A*np.cos(la1)*np.sin(lo1)+B*np.cos(la2)*np.sin(lo2); z=A*np.sin(la1)+B*np.sin(la2)
    la=np.degrees(np.arctan2(z,np.hypot(x,y))); lo=np.degrees(np.arctan2(y,x)); ci=np.floor(la-LAT0).astype(int); cj=np.floor(lo-LON0).astype(int); ok=(ci>=0)&(ci<NLA)&(cj>=0)&(cj<NLO); return f[ok],ci[ok],cj[ok]
N=len(t); out={km:np.full((N,len(DS)),np.nan,np.float32) for km in KM}
for dst,g in t.groupby("ADES_mvt"):
    if dst not in ap.index: continue
    c=cells(APT,dst)
    if c is None: continue
    f,ci,cj=c; ix=g.index.values; mvt=t.mvt.values[ix]; ref=t.ref.values[ix]; Tt=(np.where(np.isfinite(t.arr1.values[ix]),t.arr1.values[ix],ref+7200)-ref-900).clip(1200,60000)
    for j,d in enumerate(DS):
        ra={km:np.zeros(len(ix)) for km in KM}; any_={km:np.zeros(len(ix),bool) for km in KM}
        for k in range(len(f)):
            tk=mvt+d*60+f[k]*Tt; x=(tk-T0)/900.0-0.5; b0=np.floor(x).astype(int); w=x-b0; b0c=np.clip(b0,0,ND*96-2)
            v=(1-w)*OC[b0c,ci[k],cj[k]]+w*OC[b0c+1,ci[k],cj[k]]; br=np.floor((mvt+f[k]*Tt-T0)/900.0).astype(int); v=np.maximum(v-((1-w)*(b0c==br)+w*((b0c+1)==br)),0)
            day=np.clip(((tk-T0)//86400).astype(int),0,ND-1); hr=np.clip((((tk-T0)%86400)//3600).astype(int),0,23); me=MEAN[day,hr,ci[k],cj[k]]; cov=(me>=2.0)&(b0>=0)&(b0+1<ND*96)
            val=np.where(cov,(v+1)/np.maximum(me,1.0),0)
            for km in KM:
                if k<km: ra[km]=np.maximum(ra[km],val); any_[km]|=cov
        for km in KM: out[km][ix,j]=np.where(any_[km],ra[km],np.nan)
for km in KM:
    r=pd.DataFrame(out[km],columns=["d%+.1f"%d for d in DS]); r.insert(0,"MVT_ID_mvt",t.MVT_ID_mvt.values); r["mvt"]=t.mvt.values; r["y"]=t.y.values; r["en"]=t.en.values; r["ma"]=t.ma.values; r["ADES"]=t.ADES_mvt.values
    os.makedirs(D+"rwy",exist_ok=True); r.to_parquet(D+"rwy/%s_%s_k%d_m%02d.parquet"%(APT,RW,km,M))
print("mes",M,N,"ok",flush=True)
