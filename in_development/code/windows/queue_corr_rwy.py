import numpy as np,pandas as pd,glob
cols=['MVT_ID_mvt','ADEP_mvt','mvt','aobt','hr','day','qd_300','qd_600','qd_900','qd_1800','bl_1','dw_1','wt_1','bl_2','dw_2','wt_2','sat_600','sat_1800','ap_qd_600']
yt=pd.read_parquet('../ablmac/year_table.parquet',columns=cols); yt['MVT_ID_mvt']=yt.MVT_ID_mvt.astype('int64')
rw=pd.concat([pd.read_parquet(f,columns=['MVT_ID_mvt','RUNWAY_mvt']) for f in glob.glob('slim/rawslim_*.parquet')]); rw['MVT_ID_mvt']=rw.MVT_ID_mvt.astype('int64'); yt=yt.merge(rw,on='MVT_ID_mvt',how='left')
yt['aobt_s']=yt.aobt.astype(float) if np.issubdtype(yt.aobt.dtype,np.number) else pd.to_datetime(yt.aobt,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float)
# ocupação do taxi-out no instante do MVT = voos já com pushback (AOBT) que ainda não decolaram (MVT_j > MVT_i); no aeroporto e na mesma pista
yt['q_taxi_apt']=np.nan; yt['q_taxi_rwy']=np.nan
for a,g in yt.groupby('ADEP_mvt'):
    ok=g.aobt_s.notna(); gg=g[ok]; A=np.sort(gg.aobt_s.values); M=np.sort(gg.mvt.values); yt.loc[gg.index,'q_taxi_apt']=np.searchsorted(A,gg.mvt.values,'right')-np.searchsorted(M,gg.mvt.values,'right')
    for r,h in gg.groupby('RUNWAY_mvt'):
        A2=np.sort(h.aobt_s.values); M2=np.sort(h.mvt.values); yt.loc[h.index,'q_taxi_rwy']=np.searchsorted(A2,h.mvt.values,'right')-np.searchsorted(M2,h.mvt.values,'right')
w=pd.read_parquet('rwy_windows.parquet').drop(columns=['apt','rwy']).merge(pd.read_parquet('rwy_ref.parquet'),on='MVT_ID_mvt',how='outer'); w['MVT_ID_mvt']=w.MVT_ID_mvt.astype('int64'); a=yt.merge(w,on='MVT_ID_mvt',how='inner')

Q=['q_taxi_rwy','q_taxi_apt','qd_900','bl_1','wt_1','sat_600']
IND=['rf_closed5','rf_closed15','rf_closed30','rf_width','rf_closed_before','rg_early_slack','rg_own_window','rg_pos_frac','rg_idle_open_s','rg_idle_frac','rg_idle_slots','rg_n_windows_ready']
def sp(x,y):
    m=x.notna()&y.notna()
    return x[m].rank().corr(y[m].rank()) if m.sum()>200 else np.nan
def sp_in(df,x,y):  # dentro de aeroporto x hora do dia (remove hora/aeroporto)
    out=[];wt=[]
    for _,g in df.groupby(['ADEP_mvt','hr']):
        if len(g)>=60 and g[x].nunique()>2 and g[y].nunique()>2:
            c=sp(g[x],g[y]); 
            if not np.isnan(c): out.append(c); wt.append(len(g))
    return np.average(out,weights=wt) if out else np.nan
print('voos',len(a))
print('\n== Spearman entre o INDICADOR (linhas) e a FILA (colunas) — todos os voos (Janela_P) / voos com trilha própria (Janela_obs)')
R=pd.DataFrame({q:{i:sp(a[i],a[q]) for i in IND} for q in Q}).round(3); print(R.to_string())
print('\n== o mesmo DENTRO de aeroporto x hora do dia (remove ciclo diário)')
R2=pd.DataFrame({q:{i:sp_in(a,i,q) for i in IND} for q in Q}).round(3); print(R2.to_string())
print('\n== controle: a própria fila contra o atraso observado (taxi MVT-AOBT) — para saber o teto de correlação possível')
a['taxi']=a.mvt-a.aobt_s; print({q:round(sp(a[q],a.taxi),3) for q in ('q_taxi_rwy','q_taxi_apt','qd_300','bl_1','wt_1')})

a['en_']=pd.read_parquet('../ablmac/year_table.parquet',columns=['en']).en.values[:0].tolist() if False else 0
