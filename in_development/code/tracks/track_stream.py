"""Posições individuais (>=5 s por aeronave) de aeronaves no ar < 12000 ft num raio ~55 km dos 10 aeroportos, de um dia do adsb.lol em STREAMING (sem gravar o bruto). Saída: ~/tracks/trk_<dia>.parquet
Uso: python track_stream.py AAAA-MM-DD   (reaproveita parts()/head() do grid_stream.py)"""
import gzip, subprocess, sys, tarfile, time, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/yearabl'))
from grid_stream import parts, loads
OUT=os.path.expanduser(os.environ.get('TRK_OUT','~/tracks')); os.makedirs(OUT,exist_ok=True)
ap=pd.read_csv(os.path.expanduser('~/yearabl/airports.csv'),usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident')
APTS=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
B=[(a,ap.latitude_deg[a],ap.longitude_deg[a],0.5,0.5/np.cos(np.radians(ap.latitude_deg[a]))) for a in APTS]
def main(day):
    t0=time.time(); ps=parts(day)
    if not ps: print(day,'SEM PARTES',flush=True); sys.exit(2)
    proc=subprocess.Popen(['curl','-sL','--retry','5','--retry-delay','5']+ps,stdout=subprocess.PIPE); tf=tarfile.open(fileobj=proc.stdout,mode='r|')
    rows=[]; n=0
    for m in tf:
        if not(m.isfile() and '/traces/' in m.name and m.name.endswith('.json')): continue
        b=tf.extractfile(m).read()
        try: b=gzip.decompress(b)
        except Exception: pass
        try: d=loads(b)
        except Exception: continue
        n+=1; icao=d.get('icao'); base=d.get('timestamp',0.0); last={}
        for pt in d.get('trace') or []:
            lat,lon=pt[1],pt[2]; alt=pt[3]
            if alt=='ground' or not isinstance(alt,(int,float)) or alt>12000: continue
            for a,la,lo,dl,dn in B:
                if abs(lat-la)<=dl and abs(lon-lo)<=dn:
                    t=base+pt[0]
                    if t-last.get(a,-1e9)<5: break
                    last[a]=t; rows.append((a,icao,t,lat,lon,alt,pt[4] if len(pt)>4 and pt[4] is not None else np.nan,pt[5] if len(pt)>5 and pt[5] is not None else np.nan)); break
    proc.stdout.close(); proc.wait()
    o=pd.DataFrame(rows,columns=['apt','icao','t','lat','lon','alt','gs','trk']).astype({'lat':'float32','lon':'float32','alt':'int32','gs':'float32','trk':'float32'})
    tmp=f'{OUT}/.trk_{day}.tmp.parquet'; o.to_parquet(tmp,index=False); os.replace(tmp,f'{OUT}/trk_{day}.parquet')
    print(day,'traces',n,'linhas',len(o),round(time.time()-t0),'s',flush=True)
if __name__=='__main__': main(sys.argv[1])
