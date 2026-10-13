import argparse
import numpy as np
import pandas as pd
from prc.paths import RANKING
from prc.daily.constants import FLOOR, KEY, LIRF, TARGET
from prc.daily.data import load_k

THRESHOLD = 60.0
SCALE = 1.0


def layer(base, months, ranking=RANKING, threshold=THRESHOLD, scale=SCALE, k_dir=None):
    rk = pd.read_parquet(ranking, columns=[KEY, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
    rk = rk[(rk.PHASE_mvt == 'DEP') & (rk.ADEP_mvt == LIRF)].copy()
    rk['key'] = rk[KEY].round().astype('int64')
    b = base.copy()
    b['key'] = b[KEY].round().astype('int64')
    m = b[['key', TARGET]].merge(rk[['key', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']], on='key')
    mvt = pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True)
    m['d'] = (mvt - pd.to_datetime(m.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    m['day'] = mvt.dt.tz_localize(None).dt.normalize()
    m = m[m.day.dt.month.isin(months)]
    k = load_k([2026]) if k_dir is None else load_k([2026], k_dir)
    k = k[k.apt == LIRF].drop_duplicates('day').set_index('day').k
    m['c'] = np.maximum(0.0, m.d - m[TARGET])
    m['pos'] = m.c > 0
    g = m.groupby('day').agg(total=('c', 'sum'), n=('pos', 'sum')).reset_index()
    g['k'] = g.day.map(k)
    g['b'] = (g.k - g.total) / g.n.clip(lower=1)
    g['on'] = g.b.abs() >= threshold
    m = m.merge(g[['day', 'b', 'on']], on='day')
    delta = np.where(m.pos & m.on, np.maximum(m[TARGET] - scale * m.b, FLOOR) - m[TARGET], 0.0)
    step = pd.Series(delta, index=m.key.to_numpy())
    out = base.copy()
    keys = out[KEY].round().astype('int64')
    out[TARGET] = out[TARGET].to_numpy(float) + keys.map(step).fillna(0.0).to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--months', default='1,2,6,7')
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    ap.add_argument('--scale', type=float, default=SCALE)
    a = ap.parse_args()
    base = pd.read_parquet(a.pred)
    out = layer(base, [int(x) for x in a.months.split(',')], RANKING, a.threshold, a.scale)
    out[KEY] = out[KEY].astype('float64')
    out.to_parquet(a.out, index=False)
    print(len(out), float(((out[TARGET] - base[TARGET]) ** 2).mean() ** 0.5))


if __name__ == '__main__':
    main()
