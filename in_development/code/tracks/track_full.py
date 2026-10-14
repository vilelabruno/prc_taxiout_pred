"""TODOS os pontos ADS-B (sem reamostrar) de TODA aeronave (SÓ NO AR (sem solo), até 20000 ft) num raio ~80 km dos 10 aeroportos, de um dia do adsb.lol em STREAMING. Campos do trace: t, lat, lon, alt(ft), gs, trk, flags, vrate, callsign. Saída ~/tracks_full/trf_<dia>.parquet"""
import gzip, subprocess, sys, tarfile, time, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/yearabl'))
from grid_stream import parts, loads
OUT=os.path.expanduser(os.environ.get('TRF_OUT','~/tracks_full')); os.makedirs(OUT,exist_ok=True)
ap=pd.read_csv(os.path.expanduser('~/yearabl/airports.csv'),usecols=['ident','latitude_deg','longitude_deg']).drop_duplicates('ident').set_index('ident')
APTS=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
B=[(a,ap.latitude_deg[a],ap.longitude_deg[a],0.72,0.72/np.cos(np.radians(ap.latitude_deg[a]))) for a in APTS]
g=lambda pt,i: pt[i] if len(pt)>i else None
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
        n+=1; icao=d.get('icao'); base=d.get('timestamp',0.0); reg=d.get('r'); typ=d.get('t')
        for pt in d.get('trace') or []:
            lat,lon=pt[1],pt[2]; alt=pt[3]
            if alt=='ground' or not isinstance(alt,(int,float)) or alt>20000: continue
            for a,la,lo,dl,dn in B:
                if abs(lat-la)<=dl and abs(lon-lo)<=dn:
                    ex=g(pt,8); cs=ex.get('flight') if isinstance(ex,dict) else None
                    rows.append((a,icao,reg,typ,base+pt[0],lat,lon,alt,g(pt,4),g(pt,5),g(pt,6),g(pt,7),cs)); break
    proc.stdout.close(); proc.wait()
    o=pd.DataFrame(rows,columns=['apt','icao','reg','type','t','lat','lon','alt','gs','trk','flags','vrate','callsign'])
    for c in ('gs','trk','vrate'): o[c]=pd.to_numeric(o[c],errors='coerce').astype('float32')
    o['lat']=o.lat.astype('float32'); o['lon']=o.lon.astype('float32'); o['alt']=o.alt.astype('int32'); o['flags']=pd.to_numeric(o['flags'],errors='coerce').fillna(0).astype('int32')
    tmp=f'{OUT}/.trf_{day}.tmp.parquet'; o.to_parquet(tmp,index=False,compression='zstd'); os.replace(tmp,f'{OUT}/trf_{day}.parquet')
    print(day,'traces',n,'linhas',len(o),round(time.time()-t0),'s',flush=True)
if __name__=='__main__': main(sys.argv[1])
