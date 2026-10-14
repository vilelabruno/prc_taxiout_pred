"""Céu COMPLETO ao redor de cada aeroporto (origem e destino) do adsb.lol globe_history, por dia. Blocos 3D por aeroporto:
 anel (km do aeroporto): 0-15, 15-40, 40-80, 80-150 ; camada (ft acima da elevação do campo): solo<=50, 50-3000, 3000-7000, 7000-12000, 12000-24500, >24500 ;
 quadrante (rumo do aeroporto ao avião / 90): 0 NE,1 SE,2 SW,3 NW ; fase: 0 nível/solo, 1 sobe (vr>300), 2 desce (vr<-300).
 ceu_cells_<dia>: n = aeronaves distintas por (bin5, apt, anel, camada, quadrante, fase)
 ceu_hold_<dia>: aeronaves em espera (giro acumulado >= 270 graus em 10 min, variação de altitude <= 1500 ft, <= FL150) por (bin5, apt, anel)
 ceu_rate_<dia>: pousos (cruza elev+1500 ft descendo a <= 20 km) e decolagens (cruza subindo) por (bin5, apt)
Uso: python ceu_extract.py AAAA-MM-DD <saida_dir>"""
import glob, gzip, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np, pandas as pd
try: import orjson; loads=orjson.loads
except Exception: import json; loads=json.loads
R=Path('/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge'); S='/private/tmp/claude-502/-Users-bruno-quirino-latam-com-Desktop-bruno-Projetos-prc-data-challenge/5e41970f-79c3-434a-8d98-364545fe860d/scratchpad/'
RINGS=np.array([15.,40.,80.,150.]); LAYERS=np.array([50.,3000.,7000.,12000.,24500.])
def airports():
    top=pd.read_csv(S+'egll_dest_top100.csv').iloc[:,0].tolist()
    ids=list(dict.fromkeys(['EGLL','EDDF','EDDM','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']+top))
    ap=pd.read_csv(R/'data/external/ourairports/airports.csv',usecols=['ident','latitude_deg','longitude_deg','elevation_ft']).drop_duplicates('ident').set_index('ident')
    ids=[i for i in ids if i in ap.index]; a=ap.loc[ids].reset_index(); a['code']=range(len(a)); return a
def main(date,outdir):
    t0=time.time(); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); A=airports(); A.to_csv(outdir/'airports_ceu.csv',index=False)
    ALAT=A.latitude_deg.values; ALON=A.longitude_deg.values; AEL=A.elevation_ft.fillna(0).values; K=np.cos(np.radians(ALAT))*111.32; DLAT=150/110.54; DLON=150/K
    parts=sorted(glob.glob(str(R/'data/external/adsblol/raw'/date/'*.tar*'))); proc=subprocess.Popen(['cat']+parts,stdout=subprocess.PIPE); tf=tarfile.open(fileobj=proc.stdout,mode='r|')
    cells=[]; holds=[]; rates=[]; n=0
    for m in tf:
        if not(m.isfile() and '/traces/' in m.name and m.name.endswith('.json')): continue
        b=tf.extractfile(m).read()
        try: b=gzip.decompress(b)
        except Exception: pass
        try: d=loads(b)
        except Exception: continue
        tr=d.get('trace') or []; n+=1
        if len(tr)<2: continue
        lat=np.fromiter((p[1] for p in tr),float,len(tr)); lon=np.fromiter((p[2] for p in tr),float,len(tr))
        if lat.max()<A.latitude_deg.min()-1.4 or lat.min()>ALAT.max()+1.4: continue
        box=(np.abs(lat[:,None]-ALAT[None,:])<=DLAT)&(np.abs(lon[:,None]-ALON[None,:])<=DLON[None,:])
        base=d.get('timestamp',0.0)
        for j in np.where(box.any(0))[0]:
            idx=np.where(box[:,j])[0]
            t=base+np.fromiter((tr[i][0] for i in idx),float,len(idx))
            la=lat[idx]; lo=lon[idx]
            ar=[tr[i][3] for i in idx]; alt=np.array([0.0 if a=='ground' else (a if isinstance(a,(int,float)) else np.nan) for a in ar],float)
            ok=np.isfinite(alt)
            if ok.sum()<2: continue
            t,la,lo,alt=t[ok],la[ok],lo[ok],alt[ok]
            x=(lo-ALON[j])*K[j]; y=(la-ALAT[j])*110.54; dist=np.hypot(x,y); inr=dist<=150
            if not inr.any(): continue
            h=alt-AEL[j]
            dt=np.diff(t,prepend=t[0]); dh=np.diff(alt,prepend=alt[0]); vr=np.where((dt>0)&(dt<=120),dh/np.maximum(dt,1e-3)*60,np.nan)
            vr_rep=np.array([tr[i][7] if len(tr[i])>7 and isinstance(tr[i][7],(int,float)) else np.nan for i in idx],float)[ok]
            vr=np.where(np.isfinite(vr_rep),vr_rep,vr); ph=np.where(h<=50,0,np.where(vr>300,1,np.where(vr<-300,2,0)))
            ring=np.searchsorted(RINGS,dist,'left').clip(0,3); lay=np.searchsorted(LAYERS,h,'right').clip(0,5); lay=np.where(h<=50,0,np.maximum(lay,1))
            quad=((np.degrees(np.arctan2(x,y))%360)//90).astype(int)%4; b5=(t//300).astype(np.int64)
            code=((((b5*128+j)*4+ring)*6+lay)*4+quad)*3+ph
            cells.append(np.unique(code[inr]))
            # espera: giro acumulado em 10 min
            sel=np.where(inr&(h>1500)&(alt<=15000))[0]
            if len(sel)>=8:
                tt=t[sel]; xx=x[sel]; yy=y[sel]; hd=np.degrees(np.arctan2(np.diff(xx),np.diff(yy))); dpsi=(np.diff(hd,prepend=hd[0])+180)%360-180; S_=np.cumsum(np.r_[0,dpsi[1:]]) if len(dpsi)>1 else np.zeros(len(tt))
                S_=np.r_[0,np.cumsum(dpsi)][:len(tt)]; lo_i=np.searchsorted(tt,tt-600,'left'); turn=np.abs(S_-S_[lo_i]); dalt=np.abs(alt[sel]-alt[sel][lo_i])
                hit=np.where((turn>=270)&(dalt<=1500)&(tt-tt[lo_i]>=240))[0]
                if len(hit): holds.append(np.unique(((b5[sel][hit]*128+j)*4+ring[sel][hit])))
            # pousos e decolagens: cruzamento de elev+1500 ft a <= 20 km
            thr=AEL[j]+1500; a0=alt[:-1]; a1=alt[1:]; near=(dist[:-1]<=20)&(dist[1:]<=20)&((t[1:]-t[:-1])<=180)
            dn=near&(a0>thr)&(a1<=thr); up=near&(a0<=thr)&(a1>thr)
            for kind,mask in ((0,dn),(1,up)):
                for k in np.where(mask)[0]:
                    tc=t[k]+(a0[k]-thr)/max(a0[k]-a1[k],1e-6)*(t[k+1]-t[k]) if kind==0 else t[k]+(thr-a0[k])/max(a1[k]-a0[k],1e-6)*(t[k+1]-t[k])
                    rates.append((int(tc//300),j,kind))
    proc.kill()
    C=np.concatenate(cells) if cells else np.array([],dtype=np.int64); u,cn=np.unique(C,return_counts=True)
    ph=u%3; u1=u//3; quad=u1%4; u1//=4; lay=u1%6; u1//=6; ring=u1%4; u1//=4; ap=u1%128; b5=u1//128
    pd.DataFrame({'bin5':b5,'apt':ap,'ring':ring,'layer':lay,'quad':quad,'fase':ph,'n':cn}).to_parquet(outdir/f'ceu_cells_{date}.parquet',index=False)
    H=np.concatenate(holds) if holds else np.array([],dtype=np.int64); uh,ch=np.unique(H,return_counts=True)
    pd.DataFrame({'bin5':uh//512,'apt':(uh//4)%128,'ring':uh%4,'n_hold':ch}).to_parquet(outdir/f'ceu_hold_{date}.parquet',index=False)
    Rt=pd.DataFrame(rates,columns=['bin5','apt','kind']); Rt=Rt.groupby(['bin5','apt','kind']).size().unstack(fill_value=0).rename(columns={0:'arr',1:'dep'}).reset_index() if len(Rt) else pd.DataFrame(columns=['bin5','apt','arr','dep'])
    Rt.to_parquet(outdir/f'ceu_rate_{date}.parquet',index=False)
    print(date,'traces',n,'celulas',len(u),'espera',len(uh),'pousos',int(Rt.get('arr',pd.Series(dtype=int)).sum()),'decolagens',int(Rt.get('dep',pd.Series(dtype=int)).sum()),round(time.time()-t0),'s',flush=True)
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
