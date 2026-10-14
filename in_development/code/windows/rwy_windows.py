"""Janelas de PISTA. Linha do tempo de movimentos por pista (decolagens e chegadas, MVT exato). Janela fechada em t se há OUTRO movimento a menos de b_dep (decolagem) ou b_arr (pouso) de t.
Calibração por aeroporto x pista: reduz o quantil q até que >= 90% das decolagens reais estejam em janela ABERTA (excluindo o próprio voo)."""
import glob,numpy as np,pandas as pd
R='/Users/bruno.quirino@latam.com/Projetos/prc_data_challenge/data'; T=lambda s: pd.to_datetime(s,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
parts=[]
for m in range(1,13):
    r=pd.read_parquet(glob.glob(R+'/raw/training_2025-%02d-01_*.parquet'%m)[0],columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']); r['mvt']=T(r.MVT_TIME_UTC_mvt); parts.append(r.drop(columns='MVT_TIME_UTC_mvt'))
r=pd.concat(parts,ignore_index=True); r['MVT_ID_mvt']=r.MVT_ID_mvt.astype('int64'); r['apt']=np.where(r.PHASE_mvt=='DEP',r.ADEP_mvt,r.ADES_mvt)
DS=np.arange(-30,10.01,0.5); i0=int(np.where(DS==0)[0][0]); out=[]; cal=[]
def nearest_other(times,own_idx,q):
    """distância mínima de cada q aos eventos 'times' (ordenado), excluindo o evento own_idx (índice em times; -1 = nenhum). q é 1-D."""
    n=len(times); idx=np.searchsorted(times,q); best=np.full(len(q),np.inf)
    for off in (-2,-1,0,1):
        j=np.clip(idx+off,0,n-1); d=np.abs(times[j]-q); bad=(j==own_idx)
        best=np.where(bad,best,np.minimum(best,d))
    return best
for a,ga in r.groupby('apt'):
    for rw,g in ga.groupby('RUNWAY_mvt'):
        dep=g[g.PHASE_mvt=='DEP'].sort_values('mvt'); arr=g[g.PHASE_mvt=='ARR'].sort_values('mvt')
        if len(dep)<500: continue
        tD=dep.mvt.values; tA=arr.mvt.values; kD=np.arange(len(tD))
        d_dep=nearest_other(tD,kD,tD); d_arr=nearest_other(tA,-np.ones(len(tD),int),tD) if len(tA) else np.full(len(tD),np.inf)
        for q in (0.10,0.08,0.06,0.05,0.04,0.03,0.02,0.01):
            bd=np.quantile(d_dep,q); ba=np.quantile(d_arr[np.isfinite(d_arr)],q) if np.isfinite(d_arr).any() else 0.0
            bd=min(bd,120.0); ba=min(ba,120.0); op=(d_dep>=bd)&(d_arr>=ba)
            if op.mean()>=0.90: break
        cal.append(dict(apt=a,rwy=rw,n_dep=len(tD),n_arr=len(tA),q=q,b_dep=bd,b_arr=ba,aberto_em_d0=op.mean()))
        # janelas deslocadas: closed(d) para cada decolagem
        QD=tD[:,None]+DS[None,:]*60; qf=QD.ravel(); own=np.repeat(kD,len(DS))
        e1=nearest_other(tD,own,qf).reshape(QD.shape); e2=nearest_other(tA,-np.ones(len(qf),int),qf).reshape(QD.shape) if len(tA) else np.full(QD.shape,np.inf)
        closed=(e1<bd)|(e2<ba)
        n=len(tD); pre=closed[:,:i0]
        o={'MVT_ID_mvt':dep.MVT_ID_mvt.values,'apt':a,'rwy':rw,'closed0':closed[:,i0].astype(float)}
        for W,lab in ((10,5),(20,10),(30,15),(60,30)): o[f'rf_closed{lab}']=closed[:,i0-W:i0].mean(1)
        o['rf_closed_all']=pre.mean(1)
        L=np.zeros(n);Rr=np.zeros(n);Cb=np.zeros(n);no=np.zeros(n);mo=np.zeros(n);nb=np.zeros(n)
        for i in range(n):
            c=closed[i]; l=0
            while i0-l-1>=0 and not c[i0-l-1]: l+=1
            rr=0
            while i0+rr+1<len(c) and not c[i0+rr+1]: rr+=1
            j=i0-l-1;cc=0
            while j>=0 and c[j]: cc+=1;j-=1
            L[i]=l*0.5;Rr[i]=rr*0.5;Cb[i]=cc*0.5; p=c[:i0]
            ch=np.diff(np.r_[0,(~p).astype(int),0]); st=np.where(ch==1)[0];en=np.where(ch==-1)[0]; no[i]=len(st); mo[i]=(en-st).mean()*0.5 if len(st) else 0
            ch=np.diff(np.r_[0,p.astype(int),0]); nb[i]=len(np.where(ch==1)[0])
        o.update({'rf_L':L,'rf_R':Rr,'rf_width':L+Rr+0.5,'rf_closed_before':Cb,'rf_n_open_win':no,'rf_mean_open_win':mo,'rf_n_closed_blocks':nb,'rf_dep_gap_prev':np.r_[np.nan,np.diff(tD)],'rf_dep_gap_next':np.r_[np.diff(tD),np.nan]})
        out.append(pd.DataFrame(o))
C=pd.DataFrame(cal); print(C.round(2).to_string(index=False)); C.to_csv('rwy_calib.csv',index=False)
W=pd.concat(out,ignore_index=True); W.to_parquet('rwy_windows.parquet'); print(W.shape,'| decolagens em janela ABERTA no d=0: %.1f%%'%(100*(1-W.closed0.mean())))
