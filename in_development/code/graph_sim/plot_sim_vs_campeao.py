"""Figura: voo do EHAM com a simulacao reconstruida (meu pipeline) contra o MVT previsto pelo campeao (stack OOF) + RMSE nos mesmos voos."""
import os,sys,numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'src/physical')
from sincro import Sync
def corridas(t,gs):
    mv=gs>=1; runs=[]; i=0; n=len(t)
    while i<n:
        if not mv[i]: i+=1; continue
        j=i
        while j+1<n and (mv[j+1] or (t[j+1]-t[j]<10 and np.any(mv[j+1:j+4]))): j+=1
        runs.append((i,j)); i=j+1
    return [x for x in runs if t[x[1]]-t[x[0]]>=5]
S=os.path.expanduser('~/Projetos/oof_store/sincronia_grafo'); R='/Users/bruno.quirino@latam.com/Projetos/oof_store/clean_oof/chain/'; C='stack linear por aeroporto [componentes+hand]'
OUT=sys.argv[1]; apt='EHAM'
tr=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt'])
tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt==apt)].copy()
for c,n in (('MVT_TIME_UTC_mvt','mvt'),('BLOCK_TIME_UTC_mvt','blk')): tr[n]=pd.to_datetime(tr[c],utc=True).dt.as_unit('s').astype('int64')
tr=tr.drop_duplicates(['blk','mvt','STAND_mvt']).rename(columns={'STAND_mvt':'stand'})[['MVT_ID_mvt','blk','mvt','stand']]
sim=pd.read_parquet(f'{S}/sim/{apt}_anual.parquet').merge(tr,on=['blk','mvt','stand'],how='left')
oof=pd.read_parquet(R+'stack_oof_preds.parquet',columns=['MVT_ID_mvt',C]).rename(columns={C:'champ'}); sim=sim.merge(oof,on='MVT_ID_mvt',how='left')
e=sim.dropna(subset=['y','y_base','y_c','champ']); rm=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
r_ch,r_me,r_ba=rm(e.y,e.champ),rm(e.y,e.y_c),rm(e.y,e.y_base); n=len(e)
print(f'n={n}  RMSE campeao {r_ch:.1f} | meu pipeline {r_me:.1f} | base {r_ba:.1f}')
# voo representativo: erros absolutos mais proximos da mediana de cada um; trilha reconstruida do ADS-B com Sync (sincro.py)
import glob
an=pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(f'{S}/anual/{apt}_*.parquet'))],ignore_index=True)
tf=pd.read_parquet('data/processed/training_all_2025.parquet',columns=['CALLSIGN_flt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','STAND_mvt'])
tf=tf[(tf.PHASE_mvt=='DEP')&(tf.ADEP_mvt==apt)].copy()
for c,n_ in (('MVT_TIME_UTC_mvt','mvt'),('BLOCK_TIME_UTC_mvt','blk')): tf[n_]=pd.to_datetime(tf[c],utc=True).dt.as_unit('s').astype('int64')
tf['cs']=tf.CALLSIGN_flt.fillna(tf.FLIGHT_mvt).str.strip(); tf=tf.drop_duplicates(['blk','mvt','STAND_mvt']).rename(columns={'STAND_mvt':'stand'})[['blk','mvt','stand','cs']]
f=an.merge(sim[['blk','mvt','stand','L','ini2_c','taxi2_med','champ','y_c']],on=['blk','mvt','stand'],how='inner').merge(tf,on=['blk','mvt','stand'],how='left').dropna(subset=['L','ini2_c','taxi2_med','champ','cs'])
f['y']=f.mvt-f.blk; ec,em=abs(e.y-e.champ).median(),abs(e.y-e.y_c).median()
f['sc']=((abs(f.y-f.champ)-ec)/ec)**2+((abs(f.y-f.y_c)-em)/em)**2; f=f[f.dur1.between(40,140)&f.par.between(100,260)&(f.y>600)].sort_values('sc')
sy=Sync(apt); g=sy.g; r=None
for cand in f.head(40).itertuples():
    a_=pd.read_parquet(f'data/processed/adsb_surface/{cand.day}.parquet',filters=[('apt','==',apt)]).sort_values('t'); a_['cs']=a_.callsign.str.strip()
    v=a_[(a_.cs==cand.cs)&(a_.t>cand.mvt-5400)&(a_.t<=cand.mvt+60)]; gd=v[v.alt==0]
    if len(gd)<8: continue
    v=v[v.icao24==gd.icao24.iloc[-1]]; gd=v[(v.alt==0)&(v.t<=cand.mvt)].reset_index(drop=True)
    t=gd.t.to_numpy().astype(float); gs=gd.gs.fillna(0).to_numpy(); rs=corridas(t,gs)
    if len(rs)<2 or abs((t[rs[1][0]]-cand.blk)-cand.ini2)>5: continue      # reproduz o ini2 da tabela anual
    Pp=np.c_[(gd.lon-g.lon0)*g.kx,(gd.lat-g.lat0)*g.ky]; ed=np.array([x[0] for x in sy.snap(Pp)]); ok=ed>=0
    if ok.sum()<5: continue
    nodes=sy.route(list(ed[ok])); sS,_,_=sy.along(nodes,Pp); sS=np.maximum.accumulate(sS); s=sS-sS[0]; r=cand; break
if r is None: raise SystemExit('nenhum voo candidato reproduziu o ini2')
tr_=t-r.blk; y_true=r.y; y_ch=r.champ; y_me=r.ini2_c+r.taxi2_med
print(f'voo {r.cs} {r.day} pista {r.rwy} stand {r.stand}: y real {y_true:.0f} | campeao {y_ch:.0f} ({y_ch-y_true:+.0f}) | meu {y_me:.0f} ({y_me-y_true:+.0f}) | ini2 real {r.ini2:.0f} previsto {r.ini2_c:.0f} | taxi real {r.taxi2:.0f} previsto {r.taxi2_med:.0f}')
INK,INK2,GRID,BG='#0b0b0b','#52514e','#e4e3df','#fcfcfb'; BL,OR,AQ='#2a78d6','#eb6834','#1baf7a'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':GRID})
fig=plt.figure(figsize=(13.4,5.6),facecolor=BG); gs_=fig.add_gridspec(1,2,width_ratios=[2.35,1],wspace=0.16,left=0.06,right=0.985,top=0.80,bottom=0.17)
ax=fig.add_subplot(gs_[0]); bx=fig.add_subplot(gs_[1])
for a in (ax,bx):
    a.set_facecolor(BG); [a.spines[x].set_visible(False) for x in ('top','right')]; a.grid(axis='y',color=GRID,lw=.8); a.set_axisbelow(True)
# --- painel A: trilha observada, fases, simulacao, MVT previstos
i0,j0=rs[0]; i1,j1=rs[1]; p1=(tr_[i0],tr_[j0]); par=(tr_[j0],tr_[i1])
ax.axvspan(*p1,color='#d9d8d2',alpha=.65,lw=0); ax.axvspan(*par,facecolor='none',edgecolor='#b9b8b0',hatch='///',lw=0)
ymax=max(s.max(),r.L)*1.02
for x,lab,al in ((np.mean(p1),'pushback',1),(np.mean(par),'parada',1)): ax.text(x,ymax*.17,lab,ha='center',va='bottom',color=INK2,fontsize=9.5)
ax.plot(tr_,s,color=INK,lw=1.6,marker='o',ms=2.6,label='ADS-B observado (posição ao longo da rota)',zorder=3)
xs=[0,r.ini2_c,y_me]; ys=[0,0,r.L]; ax.plot(xs,ys,color=OR,lw=2.2,ls=(0,(5,2)),label=f'Simulação reconstruída: espera {r.ini2_c:.0f} s + taxi {r.taxi2_med:.0f} s',zorder=4)
ax.plot([r.ini2_c],[0],'o',color=OR,ms=7,mec=BG,mew=2,zorder=5)
for x,col,lab,ls_ in ((y_true,INK,'MVT real',(0,(1,0))),(y_ch,BL,'MVT previsto\npelo campeão',(0,(1,0))),(y_me,OR,'MVT previsto\npelo meu pipeline',(0,(1,0)))):
    ax.axvline(x,color=col,lw=2 if col!=INK else 1.4,ls=ls_ if col!=INK else (0,(3,3)),zorder=2)
pos={'MVT real':(y_true+8,.90,'left'),'MVT previsto\npelo campeão':(y_ch+8,.66,'left'),'MVT previsto\npelo meu pipeline':(y_me-8,.12,'right')}
for x,col,lab in ((y_true,INK,'MVT real'),(y_ch,BL,'MVT previsto\npelo campeão'),(y_me,OR,'MVT previsto\npelo meu pipeline')):
    d=x-y_true; txt=lab if col==INK else f'{lab}\n{d:+.0f} s'; px,py,ha=pos[lab]
    ax.text(px,ymax*py,txt,color=INK,fontsize=9.5,va='center',ha=ha,bbox=dict(boxstyle='round,pad=.25',fc=BG,ec=col,lw=1.2),zorder=6)
ax.annotate('fim da simulação:\nponto de espera (Dijkstra)',xy=(y_me,r.L),xytext=(y_me-20,r.L+560),ha='right',va='bottom',fontsize=9,color=INK2,arrowprops=dict(arrowstyle='-',color=INK2,lw=.8))
ax.axvline(0,color=INK2,lw=1,ls=':'); ax.text(0,-ymax*.095,'BLOCK',ha='center',color=INK2,fontsize=9,clip_on=False)
ax.set_xlim(-80,max(y_true,y_ch,y_me)*1.30); ax.set_ylim(0,ymax); ax.set_xlabel('segundos desde o BLOCK real',fontsize=10.5,labelpad=18); ax.set_ylabel('posição ao longo da rota stand→pista (m)',fontsize=10.5)
ax.legend(loc='upper left',bbox_to_anchor=(0.0,1.0),frameon=False,fontsize=9.5,labelcolor=INK)
ax.set_title(f'{r.cs} · EHAM · {r.day} · pista {r.rwy} · stand {r.stand}',loc='left',fontsize=11,color=INK2,pad=8)
# --- painel B: RMSE
names=['Campeão\n(stack OOF)','Meu pipeline\n(espera + taxi)','Base: mediana\nda pista, 30 d']; vals=[r_ch,r_me,r_ba]; cols=[BL,OR,AQ]
bars=bx.bar(names,vals,color=cols,width=.58,edgecolor='none')
for b,v in zip(bars,vals): bx.text(b.get_x()+b.get_width()/2,v+6,f'{v:.0f} s',ha='center',va='bottom',fontsize=12,fontweight='bold',color=INK)
bx.set_ylim(0,max(vals)*1.14); bx.set_ylabel('RMSE de y = MVT − BLOCK (s)',fontsize=10.5); bx.tick_params(axis='x',labelsize=9.5,length=0)
bx.set_title(f'RMSE nos mesmos {n:,} voos do EHAM'.replace(',','.'),loc='left',fontsize=11,color=INK2,pad=8)
fig.text(.06,.955,f'No EHAM o RMSE do campeão é {r_ch:.0f} s; o do meu pipeline, {r_me:.0f} s',fontsize=15.5,fontweight='bold',color=INK,ha='left',va='top')
fig.text(.06,.905,f'Esquerda: um voo típico (erro absoluto mediano do campeão e do meu pipeline), trilha reconstruída em fases. Direita: RMSE de jan+jul/2025, dias ≥ 8.',fontsize=10,color=INK2,ha='left',va='top')
fig.text(.06,.03,'Campeão = stack linear por aeroporto do OOF (jan+jul/2025), usado como proxy do campeão. A simulação termina no ponto de espera; a trilha real segue pela pista até o MVT.',fontsize=9,color=INK2,ha='left',va='bottom')
fig.savefig(OUT,dpi=170,facecolor=BG); print('salvo',OUT)
