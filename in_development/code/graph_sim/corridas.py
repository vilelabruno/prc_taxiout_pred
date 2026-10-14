"""Segmenta a trilha de voos que chegam ao stand em corridas: movimento (gs>=1 kt, juntando paradas <10 s) separadas por paradas."""
import os,sys,numpy as np,pandas as pd
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/regride')
def corridas(t,s,gs):
    mv=gs>=1; runs=[]; i=0; n=len(t)
    while i<n:
        if not mv[i]: i+=1; continue
        j=i
        while j+1<n and (mv[j+1] or (t[j+1]-t[j]<10 and np.any(mv[j+1:j+4]))): j+=1
        runs.append((i,j)); i=j+1
    return runs
for apt in sys.argv[1].split(','):
    d=pd.read_parquet(f'{O}/{apt}.parquet'); c=d[d.cat.isin(['cruzou','cruzou0'])&(d.gap<=30)&d.arr.notna()]; R=[]
    for r in c.itertuples():
        t,s,gs=np.array(r.arr).reshape(-1,3).T; rs=corridas(t,s,gs)
        rs=[x for x in rs if t[x[1]]-t[x[0]]>=5]
        row=dict(nrun=len(rs))
        for k,(i,j) in enumerate(rs[:3],1):
            row.update({f'ini{k}':t[i]-r.blk,f'dur{k}':t[j]-t[i],f'ds{k}':abs(s[j]-s[i]),f'v{k}':np.mean(gs[i:j+1])*0.5144,f'sf{k}':s[j]})
            if k>1: row[f'par{k}']=t[i]-t[rs[k-2][1]]
        R.append(row)
    R=pd.DataFrame(R); print(f'\n{apt} n={len(R)}  corridas por voo (>=5 s): {R.nrun.value_counts(normalize=True).sort_index().round(2).head(5).to_dict()}')
    for k in (1,2,3):
        cols=[x for x in (f'ini{k}',f'dur{k}',f'ds{k}',f'v{k}',f'sf{k}',f'par{k}') if x in R]
        print(f' corrida {k}:',{x:'/'.join(f'{v:.0f}' if x!=f'v{k}' else f'{v:.1f}' for v in R[x].quantile([.1,.5,.9])) for x in cols})
