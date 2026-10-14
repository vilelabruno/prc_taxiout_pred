import re, numpy as np, pandas as pd
ELEV={'EGLL':83,'EDDF':364,'EDDM':1487,'EHAM':-11,'LEBL':12,'LEMD':1998,'LFPG':392,'LIRF':14,'LSZH':1417,'LTFM':325}
def load_metar(path):
    m=pd.read_csv(path,na_values=['null','M']); m['t']=pd.to_datetime(m.valid,utc=True).astype('datetime64[s, UTC]').astype('int64').astype(float); m=m.sort_values('t').drop_duplicates('t').reset_index(drop=True)
    txt=m.metar.astype(str); q=txt.str.extract(r'\bQ(\d{4})\b')[0].astype(float); a=txt.str.extract(r'\bA(\d{4})\b')[0].astype(float)*0.338639; m['qnh']=q.fillna(a).ffill().bfill()
    def ceil(s):
        L=[int(x)*100 for k,x in re.findall(r'\b(BKN|OVC|VV)(\d{3})',s)]; return min(L) if L else 99999.0
    m['ceil']=txt.map(ceil); m['vis']=m.vsby*1.609; m['wspd']=m.sknt; m['wdir']=m.drct; wx=m.wxcodes.astype(str)
    m['ra']=wx.str.contains('RA|DZ').astype(float); m['sn']=wx.str.contains('SN|GR|PL').astype(float); m['fz']=wx.str.contains('FZ').astype(float); m['fg']=wx.str.contains('FG|BR').astype(float); m['ts']=wx.str.contains('TS').astype(float)
    m['temp']=(m.tmpf-32)/1.8; m['dew']=(m.dwpf-32)/1.8; m['lvp']=((m.vis<1.5)|(m.ceil<300)).astype(float); return m
def metar_at(m,t):
    i=np.clip(np.searchsorted(m.t.values,t,side='right')-1,0,len(m)-1); j=np.clip(i+1,0,len(m)-1); t0=m.t.values[i]; t1=m.t.values[j]; w=np.where(t1>t0,np.clip((t-t0)/np.maximum(t1-t0,1),0,1),0); o={'age_min':(t-t0)/60}
    for c in ('qnh','temp','dew','wspd','gust'):
        v0=m[c].values[i]; v1=m[c].values[j]; o[c]=np.where(np.isfinite(v0)&np.isfinite(v1),v0*(1-w)+v1*w,np.where(np.isfinite(v0),v0,v1))
    th=np.radians(m.wdir.values); wu=-m.wspd.values*np.sin(th); wv=-m.wspd.values*np.cos(th); o['wu']=wu[i]*(1-w)+wu[j]*w; o['wv']=wv[i]*(1-w)+wv[j]*w
    for c in ('ceil','vis','ra','sn','fz','fg','ts','lvp'): o[c]=m[c].values[i]
    return o
