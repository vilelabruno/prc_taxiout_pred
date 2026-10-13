import json
import numpy as np
import pandas as pd
from prc.paths import ARTIFACTS, ensure
from prc.daily.constants import (BANDS, DITHER_HALF, DITHER_SEED, ERR_CLIP, HELD_GAP, KEY, LIRF, LONG, SIG_TAU_CAP,
                                 TC2_FLOOR, VAR_DEFAULT, Y_MAX)
from prc.daily.data import labelled_departures, load_k

PARAMS_PATH = ARTIFACTS / 'daily' / 'params.json'


def read_oof(path, pred_col):
    if str(path).endswith('.npz'):
        z = np.load(path)
        return pd.DataFrame({KEY: z['ids'], pred_col: z['pred']})
    return pd.read_parquet(path, columns=[KEY, pred_col])


def fit_frame(oof_path, pred_col='prediction', dither=False):
    lab = labelled_departures()
    oof = read_oof(oof_path, pred_col)
    oof[KEY] = oof[KEY].round().astype('int64')
    t = lab.merge(oof.drop_duplicates(KEY).rename(columns={pred_col: 'p'}), on=KEY, how='inner')
    t = t[t.y < Y_MAX].copy()
    t['p'] = t.p.clip(lower=0)
    if dither:
        t['d'] = t.d + np.random.default_rng(DITHER_SEED).uniform(-DITHER_HALF, DITHER_HALF, len(t))
    t = t.merge(load_k([2025])[['apt', 'day', 'k']], on=['apt', 'day'], how='left')
    t = t[t.k.notna()].copy()
    t['band'] = pd.cut(t.d, BANDS, labels=False)
    t['e'] = t.y - t.p
    return t


def fit_params(oof_path, pred_col='prediction', out_path=PARAMS_PATH, dither=False):
    t = fit_frame(oof_path, pred_col, dither)
    lirf = t[t.apt == LIRF]
    held = lirf[(lirf.d > LONG) & ((lirf.y - lirf.d).abs() > HELD_GAP)]
    c = t[t.p < t.d]
    ec = c.e.clip(-ERR_CLIP, ERR_CLIP)
    v = {f'{int(a)}:{int(b)}': float(g.var()) for (a, b), g in ec.groupby([(c.apt == LIRF).to_numpy(), c.band.to_numpy()])}
    tc2 = {}
    for a, x in c.groupby('apt'):
        day_mean = x.e.clip(-ERR_CLIP, ERR_CLIP).groupby(x.day).mean()
        n = x.groupby('day').size()
        idio = np.mean([v.get(f'{int(a == LIRF)}:{int(b)}', VAR_DEFAULT) for b in x.band]) / max(n.mean(), 1)
        tc2[a] = max(float(day_mean.var()) - idio, TC2_FLOOR)
    days = t.groupby(['apt', 'day']).agg(k=('k', 'first'), n=('d', 'size'))
    params = {
        'tau': float(held.y.median()),
        'sig_tau': min(float(held.y.std()), SIG_TAU_CAP),
        'cmed': float((days.k / days.n).median()),
        'tc2': tc2,
        'v': v,
    }
    if out_path is not None:
        ensure(ARTIFACTS / 'daily')
        with open(out_path, 'w') as f:
            json.dump(params, f, indent=1)
    return params
