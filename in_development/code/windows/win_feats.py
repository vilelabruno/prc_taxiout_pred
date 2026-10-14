"""Features de JANELAS por decolagem (origem). Janela aberta no deslocamento d = a trajetória própria deslocada d min não chega a < THR NM de nenhuma outra aeronave com separação vertical < 1000 ft.
Uso: python win_feats.py APT DIA DIR_TRILHAS YEAR_TABLE OUT.parquet [STEP_MIN]"""
import sys, os, numpy as np, pandas as pd
APT,DAY,DIR,YT,OUT=sys.argv[1:6]; STEP=float(sys.argv[6]) if len(sys.argv)>6 else 1.0
AP={'EGLL':(51.4706,-0.4619),'EDDF':(50.0333,8.5706),'EDDM':(48.3538,11.7861),'EHAM':(52.3086,4.7639),'LEBL':(41.2971,2.0785),'LEMD':(40.4719,-3.5626),'LFPG':(49.0097,2.5479),'LIRF':(41.8003,12.2389),'LSZH':(47.4647,8.5492),'LTFM':(41.2753,28.7519)}
la0,lo0=AP[APT]; KX=111.32*np.cos(np.radians(la0)); KY=110.57; NM=1.852; DT=5
o=pd.read_parquet(f'{DIR}/trf_{DAY}.parquet'); o=o[o.apt==APT].sort_values(['icao','t']).reset_index(drop=True)
o['x']=((o.lon-lo0)*KX).astype('float32'); o['y']=((o.lat-la0)*KY).astype('float32')
T0=np.floor(o.t.min()/DT)*DT; NB=int((o.t.max()-T0)//DT)+2; icl=[]; P=np.full((NB,o.icao.nunique(),4),np.nan,np.float32)
for j,(ic,g) in enumerate(o.groupby('icao',sort=False)):
    t=g.t.values; b0=int((t[0]-T0)//DT); b1=int((t[-1]-T0)//DT)+1; tb=T0+np.arange(b0,b1+1)*DT; icl.append(ic)
    for c,col in enumerate(('x','y','alt','gs')): P[b0:b1+1,j,c]=np.interp(tb,t,np.nan_to_num(g[col].values,nan=200.0) if col=='gs' else g[col].values)
    for a,b in zip(t[:-1][np.diff(t)>30],t[1:][np.diff(t)>30]): P[int((a-T0)//DT)+1:int((b-T0)//DT)+1,j,:]=np.nan
icl=np.array(icl); GSMIN=float(os.environ.get('GSMIN',90)); sl=P[...,3]<GSMIN; P[sl,0:3]=np.nan
# decolagens = trilhas cujo 1º ponto no ar está a <25 km do aeroporto e <5000 ft (o solo foi excluído)
NJ=P.shape[1]; FB=np.array([int(np.argmax(np.isfinite(P[:,j,0]))) for j in range(NJ)]); r0=np.hypot(P[FB,np.arange(NJ),0],P[FB,np.arange(NJ),1]); a0=P[FB,np.arange(NJ),2]
dep=np.where((r0<25)&(a0<5000))[0]; toff=T0+FB*DT
csj={}
for ic,cs in o.dropna(subset=['callsign']).groupby('icao').callsign: csj[ic]=cs.iloc[0].strip().upper()
RAW=f'/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/data/raw/training_2025-{DAY[5:7]}-01_*.parquet'
import glob
raw=pd.read_parquet(glob.glob(RAW)[0],columns=['MVT_ID_mvt','CALLSIGN_flt']); raw['cs']=raw.CALLSIGN_flt.astype(str).str.strip().str.upper(); raw['MVT_ID_mvt']=raw.MVT_ID_mvt.astype('int64')
yt=pd.read_parquet(YT,columns=['MVT_ID_mvt','ADEP_mvt','mvt','day']); yt=yt[(yt.ADEP_mvt==APT)&(yt.day.astype(str)==DAY)].reset_index(drop=True); yt['MVT_ID_mvt']=yt.MVT_ID_mvt.astype('int64'); yt=yt.merge(raw[['MVT_ID_mvt','cs']],on='MVT_ID_mvt',how='left')
mv=yt.mvt.values.astype(float); match={}; usedj=set(); how={}
icj={ic:j for j,ic in enumerate(icl)}; depset=set(dep.tolist()); bycs={}
for ic,c in csj.items():
    if icj[ic] in depset: bycs.setdefault(c,[]).append(icj[ic])
diffs=[]
for i in range(len(yt)):
    c=yt.cs.iloc[i]
    if c in bycs:
        cand=sorted(bycs[c],key=lambda j: abs(toff[j]-mv[i]))
        if abs(toff[cand[0]]-mv[i])<900 and cand[0] not in usedj: match[i]=cand[0]; usedj.add(cand[0]); how[i]='cs'; diffs.append(toff[cand[0]]-mv[i])
lag=float(np.median(diffs)) if len(diffs)>=10 else 0.0
pr=[]
for i in range(len(yt)):
    if i in match: continue
    for jj in dep:
        if jj in usedj: continue
        dd=abs(toff[jj]-lag-mv[i])
        if dd<40: pr.append((dd,i,jj))
pr.sort(); usedi=set()
for dd,i,jj in pr:
    if i in usedi or jj in usedj: continue
    usedi.add(i); usedj.add(jj); match[i]=jj; how[i]='t'
print(DAY,'casados por indicativo',sum(v=='cs' for v in how.values()),'por horário',sum(v=='t' for v in how.values()),'| lag mediano (1º ponto ADS-B − MVT) %.0f s'%lag,'de',len(yt),flush=True)
K=int(900/DT)+1; DS=np.arange(-45,30+1e-9,STEP); nd=len(DS); rows=[]
for i,f in yt.iterrows():
    if i not in match: continue
    oj=int(match[i]); bm=int(FB[oj]); M=T0+bm*DT
    own=P[bm:bm+K,oj,:]
    if np.isfinite(own[:,0]).sum()<0.6*K: continue
    b_lo=max(0,bm-int(50*60/DT)); b_hi=min(NB,bm+K+int(35*60/DT)); act=np.isfinite(P[b_lo:b_hi,:,0]).any(0); act[oj]=False
    Oth=P[:,act,:]; 
    if Oth.shape[1]==0: continue
    idx=(bm+(DS*60/DT).round().astype(int))[:,None]+np.arange(K)[None,:]; ok=(idx>=0)&(idx<NB); idx=np.clip(idx,0,NB-1)
    O=Oth[idx]; dx=O[...,0]-own[None,:,0,None]; dy=O[...,1]-own[None,:,1,None]; h=np.hypot(dx,dy)/NM; v=np.abs(O[...,2]-own[None,:,2,None])
    h=np.where(np.isfinite(h)&ok[...,None]&np.isfinite(own[None,:,0,None]),h,np.inf); hv=np.where(v<1000,h,np.inf)
    if os.environ.get('DEBUG') and how[i]=='cs' and globals().setdefault('_nd',[0])[0]<8:
        _nd[0]+=1; i0_=int(np.argmin(np.abs(DS))); kk,nn=np.unravel_index(np.argmin(h[i0_]),h[i0_].shape); aj=np.where(act)[0][nn]; print('DBG',yt.cs.iloc[i],'| mais próximo:',icl[aj],csj.get(icl[aj],''),'dist %.2f NM'%h[i0_][kk,nn],'dz %.0f ft'%v[i0_][kk,nn],'em k=%d (%d s após decolagem)'%(kk,kk*DT),'| own alt %.0f | outro alt %.0f | n_trilha_outro_bins %d | own FB r0 %.1f km'%(own[kk,2],O[i0_,kk,nn,2],int(np.isfinite(P[:,aj,0]).sum()),r0[oj]))
    dmin_v=hv.min(axis=(1,2)); dmin_a=h.min(axis=(1,2)); r={'MVT_ID_mvt':f.MVT_ID_mvt,'how':how[i],'n_oth':int(Oth.shape[1]),'dmin_v0':float(dmin_v[np.argmin(np.abs(DS))]),'dmin_a0':float(dmin_a[np.argmin(np.abs(DS))])}
    r.update({f'dv_{d:+.0f}':float(dmin_v[k]) for k,d in enumerate(DS)}); r.update({f'da_{d:+.0f}':float(dmin_a[k]) for k,d in enumerate(DS)})
    for THR in (3.0,5.0):
        cl=dmin_v<THR; i0=int(np.argmin(np.abs(DS))); s=f'{int(THR)}'
        r['closed0_'+s]=bool(cl[i0])
        L=0
        while i0-L-1>=0 and not cl[i0-L-1]: L+=1
        R=0
        while i0+R+1<nd and not cl[i0+R+1]: R+=1
        r['L_'+s]=L*STEP; r['R_'+s]=R*STEP; r['width_'+s]=(L+R+1)*STEP
        # bloco fechado imediatamente antes da janela do MVT (quanto tempo antes estava fechado) e distância até a janela aberta anterior
        j=i0-L-1; C=0
        while j>=0 and cl[j]: C+=1; j-=1
        r['closed_run_before_'+s]=C*STEP; r['prev_open_gap_'+s]=(C*STEP if j>=0 else np.nan)
        pre=cl[:i0]; r['fclosed30_'+s]=float(pre[-int(30/STEP):].mean()); r['fclosed10_'+s]=float(pre[-int(10/STEP):].mean()); r['fclosed45_'+s]=float(pre.mean())
        # nº e largura média das janelas abertas e dos blocos fechados em [-45,0)
        ch=np.diff(np.r_[0,pre.astype(int),0]); st=np.where(ch==1)[0]; en=np.where(ch==-1)[0]; r['n_closed_blocks_'+s]=len(st); r['mean_closed_block_'+s]=float((en-st).mean()*STEP) if len(st) else 0.0
        po=~pre; ch=np.diff(np.r_[0,po.astype(int),0]); st=np.where(ch==1)[0]; en=np.where(ch==-1)[0]; r['n_open_windows_'+s]=len(st); r['mean_open_window_'+s]=float((en-st).mean()*STEP) if len(st) else 0.0
    rows.append(r)
pd.DataFrame(rows).assign(day=DAY).to_parquet(OUT); print(DAY,len(yt),'voos |',len(rows),'com trilha própria')
