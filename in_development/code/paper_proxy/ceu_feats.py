"""Pressão de origem e destino a partir do céu completo (ceu_*.parquet), janelas SEMPRE para trás do instante de referência.
Referência do destino: ETA = MVT + T_od(rota, mês) ; da origem: MVT. Medidas por aeroporto e bin5:
 n_low  = aeronaves distintas em anéis <= 80 km e camadas 50-12000 ft ; n_desc / n_climb = idem por fase ; n_lvl = fase nível nas camadas 3000-12000 ft (espera/vetoração)
 n_hold = aeronaves em espera (qualquer anel) ; arr, dep = pousos e decolagens do bin ; qmax = maior ocupação de um quadrante (anel 1-2, <= 12000 ft)
Normalização: p95 de cada medida por aeroporto (e capacidade de chegadas = p99 de arr em hora móvel)."""
import glob, numpy as np, pandas as pd
def load(dirp):
    A=pd.read_csv(dirp+'airports_ceu.csv'); C=[];H=[];R=[]
    for f in sorted(glob.glob(dirp+'ceu_cells_*.parquet')):
        d=f.split('ceu_cells_')[1][:-8]
        if not glob.glob(dirp+f'ceu_rate_{d}.parquet'): continue
        C.append(pd.read_parquet(f)); H.append(pd.read_parquet(dirp+f'ceu_hold_{d}.parquet')); R.append(pd.read_parquet(dirp+f'ceu_rate_{d}.parquet'))
    return A,pd.concat(C),pd.concat(H),pd.concat(R)
def series(A,C,H,R):
    c=C[(C.ring<=2)&(C.layer.between(1,3))]
    g={'n_low':c.groupby(['apt','bin5']).n.sum(),'n_desc':c[c.fase==2].groupby(['apt','bin5']).n.sum(),'n_climb':c[c.fase==1].groupby(['apt','bin5']).n.sum(),
       'n_lvl':c[(c.fase==0)&c.layer.isin([2,3])].groupby(['apt','bin5']).n.sum(),'n_hold':H.groupby(['apt','bin5']).n_hold.sum(),
       'arr':R.groupby(['apt','bin5']).arr.sum(),'dep':R.groupby(['apt','bin5']).dep.sum(),'qmax':c[c.ring.isin([1,2])].groupby(['apt','bin5','quad']).n.sum().groupby(['apt','bin5']).max()}
    df=pd.DataFrame(g).fillna(0.0)
    out={}
    for j,sub in df.groupby(level='apt'):
        s=sub.droplevel('apt'); idx=np.arange(s.index.min(),s.index.max()+1); out[int(j)]=s.reindex(idx,fill_value=0.0)
    p95={j:s.quantile(.95).clip(lower=1) for j,s in out.items()}
    cap={j:max(float(s.arr.rolling(12).sum().quantile(.99)),1.0) for j,s in out.items()}
    return out,p95,cap
def window(out,p95,cap,j,ref_bin,W_bins,shift=0):
    """média das medidas nos W bins que TERMINAM em ref_bin+shift (janela para trás)."""
    if j not in out: return None
    s=out[j]; off=s.index.min(); v=s.values; cs=np.vstack([np.zeros((1,v.shape[1])),np.cumsum(v,0)])
    hi=np.clip(ref_bin+shift-off+1,0,len(v)); lo=np.clip(hi-W_bins,0,len(v)); n=np.maximum(hi-lo,1)
    m=(cs[hi]-cs[lo])/n[:,None]; cols=list(s.columns); r={}
    for k,c in enumerate(cols): r[c]=m[:,k]/p95[j][c]
    r['util_arr']=m[:,cols.index('arr')]*12/cap[j]
    return r
def feats(refs,apt_codes,ref_bins,out,p95,cap,prefix,wins=(2,4,8,12),shifts=(0,)):
    """refs: não usado; apt_codes: array de códigos (NaN se fora da lista); ref_bins: array de bins5. Devolve DataFrame."""
    n=len(ref_bins); res={}
    for W in wins:
        for sh in shifts:
            for nm in ['n_low','n_desc','n_climb','n_lvl','n_hold','arr','dep','qmax','util_arr']: res[f'{prefix}{nm}_w{W*5}_s{sh*5}']=np.full(n,np.nan)
    for j in np.unique(apt_codes[np.isfinite(apt_codes)]).astype(int):
        k=np.where(apt_codes==j)[0]
        for W in wins:
            for sh in shifts:
                r=window(out,p95,cap,j,ref_bins[k],W,sh)
                if r is None: continue
                for nm,v in r.items():
                    if f'{prefix}{nm}_w{W*5}_s{sh*5}' in res: res[f'{prefix}{nm}_w{W*5}_s{sh*5}'][k]=v
    return pd.DataFrame(res)
