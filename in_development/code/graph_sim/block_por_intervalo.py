import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
d=pd.read_csv(os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/componentes_EHAM.csv'))
d['day']=pd.to_datetime(d.mvt,unit='s').dt.strftime('%Y-%m-%d'); d=d[d.t_prev.notna()].copy()
d['gap']=d.t_io-d.t_prev
# offset (mediana de A) aprendido nos OUTROS dias (purga +-1 dia), so com onset bem medido
days=sorted(d.day.unique()); off={}
for dd in days:
    lo,hi=pd.Timestamp(dd)-pd.Timedelta(days=1),pd.Timestamp(dd)+pd.Timedelta(days=1)
    t=d[(pd.to_datetime(d.day)<lo)|(pd.to_datetime(d.day)>hi)]; t=t[t.gap<=12]
    off[dd]=float((t.t_io-t.blk).median())
d['off']=d.day.map(off)
res={}
for nome,f in [('midpoint',0.5),('ultimo ping parado',0.0),('1o ping em movimento',1.0),('30% do intervalo',0.3)]:
    on=d.t_prev+f*d.gap; d['e_'+nome]=d.blk-(on-d.off)          # erro = real - estimado (mesmo erro em y, sinal trocado)
def r(e): e=e.dropna(); return round(float(np.sqrt((e**2).mean())),1),round(float(e.abs().median()),1),round(float(e.abs().quantile(.9)),1)
print('voos (comecam parados, com intervalo):',len(d),'| intervalo mediano (s)',d.gap.median(),'q25/q75/q90',d.gap.quantile([.25,.75,.9]).tolist())
print('RMSE / erro abs mediano / q90 do BLOCK (=erro de y), por estimador de onset:')
for nome in ['ultimo ping parado','30% do intervalo','midpoint','1o ping em movimento']: print(f'  {nome:24s}',r(d['e_'+nome]))
d['faixa']=pd.cut(d.gap,[0,6,12,30,60,120,1e9],labels=['<=6','6-12','12-30','30-60','60-120','>120'])
print('\npor tamanho do intervalo (midpoint): n | RMSE | mediana | q90')
for k,q in d.groupby('faixa'): print(f'  {k:>7s}',len(q),*r(q['e_midpoint']))
print('\nconstante (o que o campeao-sem-ADS faria, media de y-block global): erro BLOCK por constante blk_taxi nao se aplica; referencia: desvio do BLOCK->1o ping',round((d.blk_first).std(),1))
print('\nsem a cauda: RMSE midpoint nos 90% menores erros', round(float(np.sqrt((d.e_midpoint[d.e_midpoint.abs()<=d.e_midpoint.abs().quantile(.9)]**2).mean())),1))
print('\n--- cauda (midpoint) ---')
e=d.e_midpoint; print('quantis do erro (real-estimado):',e.quantile([.01,.05,.1,.25,.5,.75,.9,.95,.99]).round(0).tolist())
print('|e|>120 s:',round((e.abs()>120).mean(),3),' |e|>300:',round((e.abs()>300).mean(),3),'| positivos(onset antes do BLOCK)',round((e>120).mean(),3),'negativos(onset depois)',round((e<-120).mean(),3))
b=d[e.abs()>300].head(8)[['blk_first','t_first','t_prev','t_io','blk','gap','A','B','C']]; b['t_prev-blk']=b.t_prev-b.blk; b['t_io-blk']=b.t_io-b.blk; print(b.round(0).to_string())
print('\n--- regra combinada: se (inicio do taxi - onset) > T, usa inicio do taxi - 240 s ---')
d['t_k']=d.blk+d.total; d['espera']=d.t_k-d.t_io
on=d.t_prev+0.5*d.gap-d.off
for T in (300,450,600,900,1e9):
    est=np.where(d.espera>T,d.t_k-240,on); e=d.blk-est
    print(f'T={T:>6.0f}  RMSE {np.sqrt((e**2).mean()):6.1f}  mediana|e| {np.median(np.abs(e)):5.1f}  q90 {np.quantile(np.abs(e),.9):6.1f}  |e|>120: {(np.abs(e)>120).mean():.3f}  trocados: {(d.espera>T).mean():.3f}')
print('espera (inicio do taxi - onset) quantis 50/75/90/95/99:',d.espera.quantile([.5,.75,.9,.95,.99]).round(0).tolist())
