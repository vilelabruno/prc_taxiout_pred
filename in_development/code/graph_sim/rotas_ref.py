"""Rota recente (guia) por voo, entre dias: a ULTIMA rota completa anterior (mesmo stand e pista; senao mesmo stand; senao stand a <=300 m) que passa perto do ponto de ancora
(ultimo ponto ADS, ou o no da pista), usada do ponto dela mais perto do MEU stand ate a ancora; ligada ao meu stand por Dijkstra. Pre-calculado."""
import sys, os, glob, pickle, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
import sim_dia as SD, foco as F
R0=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/rotas_'+os.environ.get('APT','EGLL')); APT=os.environ.get('APT','EGLL')
days=sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{R0}/*.pkl')); DAYS={x:pickle.load(open(f'{R0}/{x}.pkl','rb')) for x in days}
d=SD.Dia(APT,days[0]); sy=d.sy; g=d.g
refs=[]; 
for x in days:
    for i,r in enumerate(DAYS[x]['D']):
        if r['ok'] and r['chain'] and r['Pstar'] is not None and np.hypot(*(r['Pstar']-r['xy']))<=150 and len(r['tp'])>=40 and np.diff(r['tp']).max()<=60: refs.append(dict(day=x,i=i,mvt=r['mvt'],stand=r['stand'],rwy=r['rwy'],P=r['pings'],tp=r['tp'],cs=r['cs'],xy=r['xy']))
print('voos de referencia (ADS completo do stand ate o MVT):',len(refs),flush=True)
ref_sr={};ref_s={};ref_r={}
for k,r in enumerate(refs): ref_sr.setdefault((r['stand'],r['rwy']),[]).append(k); ref_s.setdefault(r['stand'],[]).append(k); ref_r.setdefault(r['rwy'],[]).append(k)
for dct in (ref_sr,ref_s,ref_r):
    for key in dct: dct[key].sort(key=lambda k:refs[k]['mvt'])
cache={}
def guia(stand,rwy,mvt,A_,Rr,xy,idj):
    cand_levels=[(1,ref_sr.get((stand,rwy),[])),(2,ref_s.get(stand,[])),(3,[k for k in ref_r.get(rwy,[]) if np.hypot(*(refs[k]['xy']-xy))<=300])]
    for nivel,lst in cand_levels:
        ant=[k for k in lst if refs[k]['mvt']<mvt][::-1][:12]; post=[k for k in lst if refs[k]['mvt']>=mvt][:3]
        for k in ant+post:
            r=refs[k]
            if r['day']+str(r['i'])==idj: continue
            dd=np.hypot(r['P'][:,0]-A_[0],r['P'][:,1]-A_[1]); iA=int(dd.argmin())
            if dd[iA]>Rr: continue
            iS=int(np.argmin(np.hypot(r['P'][:,0]-xy[0],r['P'][:,1]-xy[1])))
            if iS>=iA: continue
            key=(k,iS,iA,rwy)
            if key not in cache:
                hops=F.segs_de_pings(d,rwy,r['P'][iS:iA+1]); kS,fS=sy.snap_many(r['P'][iS:iS+1]); leg=[]
                si=d.stand_info(stand)
                if si is not None:
                    if kS[0]>=0: c,leg=d.caminho(rwy,si['opts'],to_edge=(kS[0],fS[0]))
                    else: leg=[(-1,float(np.hypot(*(si['xy']-r['P'][iS]))))]
                cache[key]=(leg+hops,float(r['tp'][iA]-r['tp'][iS]))
            return dict(ed_guide=cache[key][0],tau_ref=cache[key][1],nivel=nivel,ref_cs=r['cs'],ref_mvt=r['mvt'])
    return None
G={}; t0=time.time(); n=0; cont={0:0,1:0,2:0,3:0}
for x in days:
    for i,r in enumerate(DAYS[x]['D']):
        if not r['ok']: continue
        if r['chain'] and r['Pstar'] is not None: A_=r['Pstar']; Rr=60.0
        else:
            an=d.anchor(r['rwy'])
            if an is None: continue
            A_=g.xy[an]; Rr=150.0
        o=guia(r['stand'],r['rwy'],r['mvt'],A_,Rr,r['xy'],x+str(i)); G[(x,i)]=o; cont[0 if o is None else o['nivel']]+=1; n+=1
    print(x,'| voos %d | com rota recente: %.1f%% | %.0f s'%(n,100*(1-cont[0]/n),time.time()-t0),flush=True)
pickle.dump(G,open(f'{R0}/_guias.pkl','wb')); print('FIM | niveis: sem guia %d | mesmo stand e pista %d | mesmo stand %d | stand vizinho %d'%(cont[0],cont[1],cont[2],cont[3]))
