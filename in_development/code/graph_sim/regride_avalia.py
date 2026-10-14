import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
APT=sys.argv[1]; S=os.path.expanduser('~/Projetos/oof_store')
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{S}/sincronia_grafo/regride/{APT}_*.parquet'))])
o=pd.read_parquet(f'{S}/grafo_por_aeroporto/{APT}/cenario_b_oof.parquet')[['id','y','champ','mes']]
d=o.merge(f,on='id',how='left'); d['d']=pd.to_datetime(d.day); d['tem']=d.tem.fillna(0)
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
print(APT,len(d),'| cadeia valida desde o MVT:',round(d.tem.mean(),3),'| jan',round(d[d.mes=='2025-01'].tem.mean(),3),'jul',round(d[d.mes=='2025-07'].tem.mean(),3))
t=d[d.tem==1].copy(); t['reached']=(t.D_star<=150)
print('das com cadeia: chegou na frente do stand (D*<=150 m):',round(t.reached.mean(),3),'| parou no meio:',round((~t.reached).mean(),3),'| cortou por passar do stand:',round(t.parou.mean(),3))
print('ponto de parada antes do BLOCK (so diagnostico, usa rotulo):',round((t.t_star<t.blk_rotulo).mean(),3))
vg=t.v_bar.median(); t['v']=t.v_bar.fillna(vg).clip(1.5,12)
t['r_R']=t.t_star-t.blk_rotulo                       # tempo do BLOCK ate o ponto de parada (patio+pushback+espera) quando chegou ao stand
t['r_M']=t.t_star-t.blk_rotulo-t.D_star/t.v          # idem, descontando a rota simulada D*/v
days=np.array(sorted(t.d.unique()))
def est(df,col):
    out=pd.Series(np.nan,index=df.index)
    for dd in days:
        te=df.d==dd; tr=df[(df.d<dd-pd.Timedelta(days=1))|(df.d>dd+pd.Timedelta(days=1))]
        out[te]=tr[col].median()
    return out
for cls,col,m in (('chegou ao stand (R)','r_R',t.reached),('no meio do taxi (M)','r_M',~t.reached)):
    q=t[m].copy(); c=est(q,col)
    q['blk_hat']=q.t_star-(q.D_star/q.v if col=='r_M' else 0)-c; q['yhat']=q.mvt-q.blk_hat; e=q.y-q.yhat
    print(f'\\n{cls}: n={len(q)} | constante mediana {q[col].median():.0f} s (quantis 10/50/90 {q[col].quantile([.1,.5,.9]).round(0).tolist()})')
    print(f'   RMSE de y fisico {rm(e):6.1f} | erro abs mediano {e.abs().median():5.1f} | q90 {e.abs().quantile(.9):6.1f} || campeao {rm(q.y-q.champ):6.1f} (mediana {(q.y-q.champ).abs().median():.1f})')
    ee=e[e.abs()<=e.abs().quantile(.9)]; print(f'   sem os 10% piores: fisico {rm(ee):.1f}')
    for w in (.3,.5,.7): print(f'   mistura w_campeao={w}: {rm(q.y-(w*q.champ+(1-w)*q.yhat)):.1f}',end='')
    print()
    if col=='r_M':
        for a,b in ((150,500),(500,1500),(1500,3000),(3000,9999)):
            z=q[(q.D_star>a)&(q.D_star<=b)]; ez=z.y-z.yhat; print(f'      D* {a}-{b} m: n={len(z)} fisico {rm(ez):6.1f} (mediana {ez.abs().median():5.1f}) campeao {rm(z.y-z.champ):6.1f}')
