import numpy as np
import pandas as pd
from prc.regime.constants import APTS, KEY, RAW_COLUMNS, WINDOWS


def window_agg(keys_ev, t_ev, v_ev, keys_q, t_q, own_v, width):
    total = np.zeros(len(t_q))
    count = np.zeros(len(t_q))
    v_ev = np.asarray(v_ev, float)
    ev = pd.DataFrame({'k': keys_ev, 't': t_ev, 'v': np.nan_to_num(v_ev), 'h': (~np.isnan(v_ev)).astype(float)})
    qk = pd.Series(keys_q)
    index = qk.groupby(qk.values).indices
    for k, g in ev.groupby('k', sort=False):
        if k not in index:
            continue
        rows = index[k]
        g = g.sort_values('t')
        ts = g.t.values
        cs = np.concatenate([[0], np.cumsum(g.v.values)])
        ch = np.concatenate([[0], np.cumsum(g.h.values)])
        tq = t_q[rows]
        lo = np.searchsorted(ts, tq - width, 'left')
        hi = np.searchsorted(ts, tq + width, 'right')
        total[rows] = cs[hi] - cs[lo]
        count[rows] = ch[hi] - ch[lo]
    if own_v is not None:
        own_v = np.asarray(own_v, float)
        has = ~np.isnan(own_v)
        total[has] -= own_v[has]
        count[has] -= 1
    return total, count


def seconds(s):
    v = s.values.astype('datetime64[s]').astype('int64').astype('float64')
    v[pd.isna(s).values] = np.nan
    return v


def prepare(raw):
    x = raw.copy()
    for c in ['MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']:
        x[c] = x[c].dt.tz_localize(None) if x[c].dt.tz is not None else x[c]
    x['apt'] = np.where(x.PHASE_mvt == 'DEP', x.ADEP_mvt, x.ADES_mvt)
    x = x[x.apt.isin(APTS)]
    x['rw'] = x.RUNWAY_mvt.fillna('NA').astype(str)
    dep = x[x.PHASE_mvt == 'DEP'].copy()
    arr = x[x.PHASE_mvt == 'ARR'].copy()
    dep['tm'] = seconds(dep.MVT_TIME_UTC_mvt)
    dep['ta'] = seconds(dep.AOBT_3_flt)
    dep['te'] = seconds(dep.EOBT_1_flt)
    arr['tm'] = seconds(arr.MVT_TIME_UTC_mvt)
    return dep.reset_index(drop=True), arr.reset_index(drop=True)


def neighbourhood(dep, arr):
    out = pd.DataFrame({KEY: dep[KEY].values})
    aobt_eobt = (dep.ta - dep.te).values
    late = np.where(np.isnan(aobt_eobt), np.nan, (aobt_eobt > 900).astype(float))
    no_aobt = np.isnan(dep.ta.values).astype(float)
    mvt_aobt = (dep.tm - dep.ta).values
    one = np.ones(len(dep))
    for name, kd, ka in [('rw', dep.apt + '|' + dep.rw, arr.apt + '|' + arr.rw), ('apt', dep.apt, arr.apt)]:
        kd = kd.values
        ka = ka.values
        for w in WINDOWS:
            width = w * 60
            for label, v in [('aoe', aobt_eobt), ('late', late), ('noa', no_aobt), ('ma', mvt_aobt)]:
                s, c = window_agg(kd, dep.tm.values, v, kd, dep.tm.values, v, width)
                out[f'{name}_{w}_{label}'] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            s, c = window_agg(kd, dep.tm.values, one, kd, dep.tm.values, one, width)
            out[f'{name}_{w}_ndep'] = c
            s, c = window_agg(ka, arr.tm.values, np.ones(len(arr)), kd, dep.tm.values, None, width)
            out[f'{name}_{w}_narr'] = c
    return out


def frame(raw):
    dep, arr = prepare(raw)
    f = neighbourhood(dep, arr)
    sec = lambda a, b: (dep[a] - dep[b]).dt.total_seconds().to_numpy()
    base = pd.DataFrame({
        KEY: dep[KEY].to_numpy(),
        'apt': dep.apt.to_numpy(),
        'mvt_aobt': sec('MVT_TIME_UTC_mvt', 'AOBT_3_flt'),
        'mvt_sched': sec('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'),
        'mvt_eobt': sec('MVT_TIME_UTC_mvt', 'EOBT_1_flt'),
        'hr': dep.MVT_TIME_UTC_mvt.dt.hour.to_numpy(),
        'month': dep.MVT_TIME_UTC_mvt.dt.month.to_numpy(),
    })
    return pd.concat([base, f.drop(columns=[KEY])], axis=1)


def feature_columns(df):
    return [c for c in df.columns if c.startswith('rw_') or c.startswith('apt_')]


def read_raw(path):
    return pd.read_parquet(path, columns=RAW_COLUMNS)
