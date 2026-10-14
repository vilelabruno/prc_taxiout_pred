"""Janelas de decolagem na ORIGEM com ADS-B + METAR sincronizado (tempo, altitude QNH, posição).
Para cada decolagem (APT, DIA): trajetória = MOLDE (mediana das trilhas reais casadas por callsign/horário da mesma pista x setor de destino; fallback pista, depois aeroporto) ancorada em MVT+lag;
a trajetória é deslocada de d=-44..+30 min (passo 2) contra TODAS as outras aeronaves reais (exclui quem decolou a +-20 s); dv(d) = menor distância horizontal (NM) a aeronave com <1000 ft de diferença, nos 5 min iniciais.
METAR (estação do aeroporto): interpolado no tempo, QNH aplicado à altitude barométrica (altura AGL), relação com teto, vento relativo à pista e ao rumo da trajetória, distância à estação.
Uso: python win_all.py APT DIA TRKDIR RAWGLOB METARDIR AIRPORTS_CSV OUT.parquet"""
import sys, os, re, glob, numpy as np, pandas as pd
APT,DAY,TRK,RAWG,METD,APCSV,OUT=sys.argv[1:8]
AP={'EGLL':(51.4706,-0.4619,83),'EDDF':(50.0333,8.5706,364),'EDDM':(48.3538,11.7861,1487),'EHAM':(52.3086,4.7639,-11),'LEBL':(41.2971,2.0785,12),'LEMD':(40.4719,-3.5626,1998),'LFPG':(49.0097,2.5479,392),'LIRF':(41.8003,12.2389,14),'LSZH':(47.4647,8.5492,1417),'LTFM':(41.2753,28.7519,325)}
la0,lo0,ELEV=AP[APT]; KX=111.32*np.cos(np.radians(la0)); KY=110.57; NM=1.852; DT=5; K=61   # 61 passos de 5 s = 5 min
# ---------------- METAR ----------------
def load_metar(apt):
    m=pd.read_csv(f'{METD}/{apt}.csv',na_values=['null','M']); m['t']=pd.to_datetime(m.valid,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float); m=m.sort_values('t').drop_duplicates('t').reset_index(drop=True)
    txt=m.metar.astype(str)
    q=txt.str.extract(r'\bQ(\d{4})\b')[0].astype(float); a=txt.str.extract(r'\bA(\d{4})\b')[0].astype(float)*0.338639; m['qnh']=q.fillna(a).ffill().bfill()
    def ceil(s):
        L=[int(x)*100 for k,x in re.findall(r'\b(BKN|OVC|VV)(\d{3})',s)]; return min(L) if L else 99999.0
    m['ceil']=txt.map(ceil); m['vis']=m.vsby*1.609; m['wspd']=m.sknt; m['wdir']=m.drct; m['gust']=m.gust; wx=m.wxcodes.astype(str)
    m['ra']=wx.str.contains('RA|DZ').astype(float); m['sn']=wx.str.contains('SN|GR|PL').astype(float); m['fz']=wx.str.contains('FZ').astype(float); m['fg']=wx.str.contains('FG|BR').astype(float); m['ts']=wx.str.contains('TS').astype(float)
    m['temp']=(m.tmpf-32)/1.8; m['dew']=(m.dwpf-32)/1.8; m['lvp']=((m.vis<1.5)|(m.ceil<300)).astype(float)
    return m
def metar_at(m,t):
    i=np.clip(np.searchsorted(m.t.values,t,side='right')-1,0,len(m)-1); j=np.clip(i+1,0,len(m)-1); t0=m.t.values[i]; t1=m.t.values[j]; w=np.where(t1>t0,np.clip((t-t0)/np.maximum(t1-t0,1),0,1),0)
    o={'age_min':(t-t0)/60}
    for c in ('qnh','temp','dew','wspd','gust'): 
        v0=m[c].values[i]; v1=m[c].values[j]; o[c]=np.where(np.isfinite(v0)&np.isfinite(v1),v0*(1-w)+v1*w,np.where(np.isfinite(v0),v0,v1))
    th=np.radians(m.wdir.values); wu=-m.wspd.values*np.sin(th); wv=-m.wspd.values*np.cos(th)   # vetor do vento (para onde sopra)
    o['wu']=wu[i]*(1-w)+wu[j]*w; o['wv']=wv[i]*(1-w)+wv[j]*w
    for c in ('ceil','vis','ra','sn','fz','fg','ts','lvp'): o[c]=m[c].values[i]
    return o
# ---------------- trilhas ----------------
o=pd.read_parquet(f'{TRK}/trf_{DAY}.parquet'); o=o[o.apt==APT].sort_values(['icao','t']).reset_index(drop=True)
o['x']=((o.lon-lo0)*KX).astype('float32'); o['y']=((o.lat-la0)*KY).astype('float32')
T0=np.floor(o.t.min()/DT)*DT; NB=int((o.t.max()-T0)//DT)+2; NJ=o.icao.nunique(); P=np.full((NB,NJ,4),np.nan,np.float32); icl=[]
for j,(ic,g) in enumerate(o.groupby('icao',sort=False)):
    t=g.t.values; b0=int((t[0]-T0)//DT); b1=int((t[-1]-T0)//DT)+1; tb=T0+np.arange(b0,b1+1)*DT; icl.append(ic)
    for c,col in enumerate(('x','y','alt','gs')): P[b0:b1+1,j,c]=np.interp(tb,t,np.nan_to_num(g[col].values,nan=200.0) if col=='gs' else g[col].values)
    gp=np.diff(t)>30
    for a,b in zip(t[:-1][gp],t[1:][gp]): P[int((a-T0)//DT)+1:int((b-T0)//DT)+1,j,:]=np.nan
icl=np.array(icl); P[P[...,3]<90,0:3]=np.nan
FB=np.array([int(np.argmax(np.isfinite(P[:,j,0]))) if np.isfinite(P[:,j,0]).any() else -1 for j in range(NJ)]); ar=np.arange(NJ)
r0=np.hypot(P[np.clip(FB,0,None),ar,0],P[np.clip(FB,0,None),ar,1]); a0=P[np.clip(FB,0,None),ar,2]
dep=np.where((FB>=0)&(r0<25)&(a0<5000))[0]; toff=T0+FB*DT
csj={ic:cs.iloc[0].strip().upper() for ic,cs in o.dropna(subset=['callsign']).groupby('icao').callsign}
# ---------------- voos do dia ----------------
mon=DAY[5:7]; raw=pd.read_parquet(glob.glob(RAWG.replace('MM',mon))[0],columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','CALLSIGN_flt'])
raw=raw[(raw.PHASE_mvt=='DEP')&(raw.ADEP_mvt==APT)].copy(); raw['mvt']=pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
raw=raw[(raw.mvt>=T0-3600)&(raw.mvt<=T0+NB*DT+3600)&(pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m-%d')==DAY)].reset_index(drop=True); raw['MVT_ID_mvt']=raw.MVT_ID_mvt.astype('int64'); raw['cs']=raw.CALLSIGN_flt.astype(str).str.strip().str.upper()
apx=pd.read_csv(APCSV,usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident')
def brg(z):
    if z not in apx.index: return np.nan
    la2,lo2=np.radians(apx.latitude_deg[z]),np.radians(apx.longitude_deg[z]); l1,o1=np.radians(la0),np.radians(lo0); return (np.degrees(np.arctan2(np.sin(lo2-o1)*np.cos(la2),np.cos(l1)*np.sin(la2)-np.sin(l1)*np.cos(la2)*np.cos(lo2-o1)))+360)%360
raw['brg']=raw.ADES_mvt.map(brg); raw['sec']=(np.floor(((raw.brg+22.5)%360)/45)).astype('float')
mv=raw.mvt.values; match={}; usedj=set(); diffs=[]; depset=set(dep.tolist()); bycs={}
for j in dep:
    c=csj.get(icl[j]); 
    if c: bycs.setdefault(c,[]).append(j)
for i in range(len(raw)):
    c=raw.cs.iloc[i]
    if c in bycs:
        cand=sorted(bycs[c],key=lambda j:abs(toff[j]-mv[i]))
        if abs(toff[cand[0]]-mv[i])<900 and cand[0] not in usedj: match[i]=cand[0]; usedj.add(cand[0]); diffs.append(toff[cand[0]]-mv[i])
lag=float(np.median(diffs)) if len(diffs)>=10 else 28.0
pr=[]
for i in range(len(raw)):
    if i in match: continue
    for jj in dep:
        if jj in usedj: continue
        dd=abs(toff[jj]-lag-mv[i])
        if dd<40: pr.append((dd,i,jj))
pr.sort(); ui=set()
for dd,i,jj in pr:
    if i in ui or jj in usedj: continue
    ui.add(i); usedj.add(jj); match[i]=jj
# ---------------- moldes (pista x setor) ----------------
paths={}
for i,j in match.items():
    p=P[FB[j]:FB[j]+K,j,:3]
    if p.shape[0]==K and np.isfinite(p[:,0]).sum()>=0.8*K: paths[i]=p
def med(idx):
    if len(idx)<3: return None
    A=np.stack([paths[i] for i in idx]); return np.nanmedian(A,axis=0)
cell_t={}
for (rw,sc),g in raw.groupby(['RUNWAY_mvt','sec']): 
    t=med([i for i in g.index if i in paths]); 
    if t is not None: cell_t[(rw,sc)]=t
rw_t={rw:med([i for i in g.index if i in paths]) for rw,g in raw.groupby('RUNWAY_mvt')}; ap_t=med(list(paths))
# ---------------- METAR ----------------
M=load_metar(APT); rows=[]; DS=np.arange(-44,30.01,2.0); nd=len(DS)
def rwy_hdg(s):
    d=re.findall(r'\d+',str(s)); return (int(d[0])*10)%360 if d else np.nan
for i,f in raw.iterrows():
    tpl=paths.get(i) if False else None
    # doador real: o próprio voo se casado; senão o voo casado mais próximo no tempo da mesma pista x setor (depois pista, depois aeroporto)
    donor=None; lvl=''
    if i in paths: donor=i; lvl='own'
    else:
        for lv,sel in (('cell',lambda k:raw.RUNWAY_mvt[k]==f.RUNWAY_mvt and raw.sec[k]==f.sec),('rwy',lambda k:raw.RUNWAY_mvt[k]==f.RUNWAY_mvt),('apt',lambda k:True)):
            c=[k for k in paths if sel(k) and abs(raw.mvt[k]-f.mvt)<1800]
            if c: donor=min(c,key=lambda k:abs(raw.mvt[k]-f.mvt)); lvl=lv; break
    if donor is None: continue
    t=paths[donor]
    M0=f.mvt+lag; bm=int((M0-T0)//DT)
    # excluir quem decolou a +-20 s (o próprio voo e vizinhos de fila na mesma cabeceira)
    act=np.isfinite(P[max(0,bm-540):bm+K+420,:,0]).any(0); near=np.where(np.abs(toff-M0)<20)[0]; act[near]=False; act[match[donor]]=False; Oth=P[:,act,:]
    if Oth.shape[1]==0: continue
    idx=(bm+(DS*60/DT).round().astype(int))[:,None]+np.arange(K)[None,:]; ok=(idx>=0)&(idx<NB); idx=np.clip(idx,0,NB-1)
    O=Oth[idx]; dx=O[...,0]-t[None,:,0,None]; dy=O[...,1]-t[None,:,1,None]; h=np.hypot(dx,dy)/NM; v=np.abs(O[...,2]-t[None,:,2,None])
    h=np.where(np.isfinite(h)&ok[...,None]&np.isfinite(t[None,:,0,None]),h,np.inf); hv=np.where(v<1000,h,np.inf); dv=hv.min(axis=(1,2)); da=h.min(axis=(1,2))
    r={'MVT_ID_mvt':f.MVT_ID_mvt,'rwy':f.RUNWAY_mvt,'sec':f.sec,'tpl':lvl,'matched':int(i in match),'n_oth':int(Oth.shape[1]),'lag':lag}
    r.update({f'dv_{int(d):+d}':float(dv[k]) for k,d in enumerate(DS)}); r.update({'da_+0':float(da[int(np.argmin(np.abs(DS)))])})
    # METAR no instante MVT e ao longo da trajetória (60, 180, 300 s após decolagem): altitude QNH -> AGL, teto, vento relativo
    hd=rwy_hdg(f.RUNWAY_mvt); w0=metar_at(M,np.array([M0]))
    for k,(sec) in enumerate((0,60,180,300)):
        kk=min(sec//DT,K-1); tt=metar_at(M,np.array([M0+sec])); alt_b=t[kk,2]; qnh=float(tt['qnh'][0]); alt_q=alt_b+(1013.25-qnh)*27.0; agl=alt_q-ELEV
        r[f'agl_{sec}']=float(agl); r[f'incloud_{sec}']=float(agl>=float(tt['ceil'][0])); r[f'dist_{sec}']=float(np.hypot(t[kk,0],t[kk,1]))
        if kk>0 and np.isfinite(t[kk,0]) and np.isfinite(t[kk-1,0]): tr=np.degrees(np.arctan2(t[kk,0]-t[kk-1,0],t[kk,1]-t[kk-1,1]))%360
        else: tr=hd
        wu,wv=float(tt['wu'][0]),float(tt['wv'][0]); trr=np.radians(tr if np.isfinite(tr) else 0.0); r[f'hw_{sec}']=-(wu*np.sin(trr)+wv*np.cos(trr)); r[f'xw_{sec}']=(wu*np.cos(trr)-wv*np.sin(trr))
    for c in ('qnh','temp','dew','wspd','gust','ceil','vis','ra','sn','fz','fg','ts','lvp','age_min'): r['wx_'+c]=float(w0[c][0])
    th=np.radians(hd if np.isfinite(hd) else 0.0); r['wx_hw_rwy']=-(float(w0['wu'][0])*np.sin(th)+float(w0['wv'][0])*np.cos(th)); r['wx_xw_rwy']=float(w0['wu'][0])*np.cos(th)-float(w0['wv'][0])*np.sin(th)
    rows.append(r)
pd.DataFrame(rows).assign(day=DAY,apt=APT).to_parquet(OUT); print(APT,DAY,'voos',len(raw),'| casados',len(match),'| moldes cell/rwy',len(cell_t),len(rw_t),'| saída',len(rows),'lag %.0f s'%lag,flush=True)
