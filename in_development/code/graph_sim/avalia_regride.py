import os,sys,numpy as np,pandas as pd
O=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo/regride')
rm=lambda x:np.sqrt(np.mean(x**2))
for apt in sys.argv[1].split(','):
    d=pd.read_parquet(f'{O}/{apt}.parquet'); print(f'\n== {apt}  n={len(d)}'); print((d.cat.value_counts(normalize=True)).round(3).to_dict())
    d['y']=d.mvt-d.blk
    c=d[d.cat.isin(['cruzou','cruzou0'])&(d.gap<=30)].copy()
    c['e_exit']=c.t_exit-c.blk; c['e_move']=c.t_move-c.blk; c['e_first']=c.t_first-c.blk
    print(f"cruzou o stand (gap<=30 s) n={len(c)}: (saida-BLOCK) med {c.e_exit.median():.0f} sd {c.e_exit.std():.0f} MAE-med {abs(c.e_exit-c.e_exit.median()).median():.0f} | (1o mov-BLOCK) med {c.e_move.median():.0f} sd {c.e_move.std():.0f} | (1o ping-BLOCK) sd {c.e_first.std():.0f}")
    p=c.dropna(subset=['t_push']); 
    if len(p)>20: print(f"  com pushback detectado (pings em posicoes diferentes no stand) n={len(p)} ({len(p)/len(c):.0%}): (t_push-BLOCK) med {(p.t_push-p.blk).median():.0f} sd {(p.t_push-p.blk).std():.0f} | (saida-BLOCK) sd {p.e_exit.std():.0f} | (1o ping na zona-BLOCK) sd {(p.t_zona0-p.blk).std():.0f}")
    q=c[~c.index.isin(p.index)]; print(f"  sem pushback visivel n={len(q)}: (saida-BLOCK) sd {q.e_exit.std():.0f}")
    print('  s da ancora (m) por categoria:',{k:np.round(v.s_a.quantile([.1,.5,.9]).values).tolist() for k,v in d.groupby('cat') if 's_a' in v and v.s_a.notna().any()})
    late=d[d.cat=='comeca_tarde']; print(f"trilha comeca depois do stand: n={len(late)}  s_a med {late.s_a.median():.0f} m  v_a med {late.v_a.median():.1f} m/s")
    from regride import tacc
    for cut in (100,200,400,800):
        h=c.dropna(subset=[f's_cut{cut}']); h=h[(h[f'v_cut{cut}']>0.3)]
        if len(h)<50: continue
        print(f" validacao: escondo s<{cut} m (n={len(h)}); erro da saida estimada vs saida observada (s):")
        for a in (0.3,0.5,1.0):
            est=h[f't_cut{cut}']-np.array([tacc(s-30,v,a) for s,v in zip(h[f's_cut{cut}'],h[f'v_cut{cut}'])])
            e=est-h.t_exit; print(f"   aceleracao {a}: vies {e.median():.0f}  RMSE {rm(e):.0f}  MAE {abs(e).median():.0f}")
        e=h[f't_cut{cut}']-(h[f's_cut{cut}']-30)/h[f'v_cut{cut}'].clip(lower=1)-h.t_exit; print(f"   velocidade constante: vies {e.median():.0f} RMSE {rm(e):.0f} MAE {abs(e).median():.0f}")
        print(f"   (1o ping visivel - saida): med {(h[f't_cut{cut}']-h.t_exit).median():.0f} s")
