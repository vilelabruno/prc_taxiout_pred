"""Avalia o simulador minimo (dias >= 8 do mes, para a janela de 30 dias ter historico)."""
import os,glob,numpy as np,pandas as pd
d=pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/sim/*.parquet'))])
d=d[pd.to_datetime(d.day).dt.day>=8]
d['own']=d.adsb & d.stand_start.astype('boolean').fillna(False).astype(bool)
e=d.dropna(subset=['y_sim','y_base'])
r=lambda a,b:np.sqrt(np.mean((a-b)**2))
print('RMSE y (s)')
for (apt,o),g in e.groupby(['apt','own']):
    print(f"  {apt} {'com trilha' if o else 'sem trilha'} n={len(g)}  base {r(g.y,g.y_base):.0f}  sim {r(g.y,g.y_sim):.0f}  sim+trilha {r(g.y,g.y_own):.0f}")
g=e[e.own].copy(); g['tm']=g.mvt-g.t_move; g['ex']=g.t_move-g.blk
print('Voos com trilha desde o stand: decomposicao')
for apt,h in g.groupby('apt'):
    vtrue=h.L/h.tm
    print(f"  {apt} sd(tm) {h.tm.std():.0f} | rmse mov L/v_hat {r(h.tm,h.L/h.v):.0f}  so-mes {r(h.tm,h.L/h.v_m):.0f}  med-pista {r(h.tm,h.tm.median()):.0f}"
          f" | corr(L/v,tm) {np.corrcoef(h.tm,h.L/h.v)[0,1]:.2f} | rmse saida {r(h.ex,h.ex_hat):.0f} sd(saida) {h.ex.std():.0f}")
