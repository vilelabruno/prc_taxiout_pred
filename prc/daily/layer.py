import json
import numpy as np
import pandas as pd
from prc.paths import RANKING
from prc.daily.constants import (BANDS, DAMPING, DITHER_VAR, DMAX, EPS0, FLOOR, KAPPA, KEY, LIRF, LIRF_CLIP, LIRF_DOSE,
                                 LOGIT_CLIP, LONG, PRIOR_HI, PRIOR_LO, SWEEPS, TARGET, VAR_DEFAULT)
from prc.daily.data import departures, load_k
from prc.daily.fit import PARAMS_PATH


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return json.load(f)


def day_delta(is_lirf, d, p, band, k, dn, par, tc2, dither):
    n_rows = len(d)
    delta = np.zeros(n_rows)
    veps = EPS0 ** 2 + (KAPPA * dn * par['cmed']) ** 2
    if dither:
        veps += DITHER_VAR * float((p < d).sum())
    sig_tau = par['sig_tau']
    binary = np.zeros(n_rows, bool)
    extra = 0.0
    if is_lirf:
        binary = (d > LONG) & (d <= DMAX)
        a = np.maximum(d[binary] - par['tau'], 1.0)
        prior = np.clip((p[binary] - par['tau']) / a, PRIOR_LO, PRIOR_HI)
        rest_mask = ~binary
        cont = rest_mask & (p < d) & (d <= DMAX)
        known = np.maximum(0.0, d[rest_mask] - p[rest_mask]).sum()
        vi = np.array([par['v'].get(f'1:{int(b)}', VAR_DEFAULT) for b in band[cont]]) if cont.any() else np.array([])
        v0 = tc2 * cont.sum() ** 2 + vi.sum() + veps
        z0 = k - known
        pc = prior.copy()
        for _ in range(SWEEPS):
            ea = ((1 - pc) * a).sum()
            va = (pc * (1 - pc) * a ** 2 + (1 - pc) * sig_tau ** 2).sum()
            new = pc.copy()
            for i in range(len(a)):
                mu = ea - (1 - pc[i]) * a[i]
                v = v0 + va - (pc[i] * (1 - pc[i]) * a[i] ** 2 + (1 - pc[i]) * sig_tau ** 2)
                r = z0 - mu
                lo = np.log(prior[i] / (1 - prior[i])) + ((r - a[i]) ** 2 - r ** 2) / (2 * v)
                new[i] = 1 / (1 + np.exp(-np.clip(lo, -LOGIT_CLIP, LOGIT_CLIP)))
            pc = DAMPING * pc + (1 - DAMPING) * new
        ea = ((1 - pc) * a).sum()
        va = (pc * (1 - pc) * a ** 2 + (1 - pc) * sig_tau ** 2).sum()
        delta[binary] = (pc - prior) * a
        resid = z0 - ea
        extra = va
        rest = ~binary
    else:
        resid = k - np.maximum(0.0, d - p).sum()
        rest = np.ones(n_rows, bool)
    cont = rest & (p < d) & ((not is_lirf) | (d <= DMAX))
    if cont.any():
        n = cont.sum()
        vi = np.array([par['v'].get(f'{int(is_lirf)}:{int(b)}', VAR_DEFAULT) for b in band[cont]])
        vt = tc2 * n ** 2 + vi.sum() + extra + veps
        delta[cont] += (tc2 * n + vi) / vt * (-resid)
    return delta


def layer(base, months, par, ranking=RANKING, dither=False, k_dir=None):
    dep = departures(ranking, months)
    b = base[[KEY, TARGET]].copy()
    b[KEY] = b[KEY].round().astype('int64')
    fr = dep.merge(b.drop_duplicates(KEY), on=KEY, how='inner', validate='1:1')
    fr['p'] = fr[TARGET].clip(lower=0)
    years = sorted(set(pd.DatetimeIndex(fr.day).year))
    kk = load_k(years) if k_dir is None else load_k(years, k_dir)
    fr = fr.merge(kk, on=['apt', 'day'], how='left')
    fr['n'] = fr.groupby(['apt', 'day'])[KEY].transform('size')
    fr['dn'] = (fr['count'] - fr['n']).abs()
    fr['band'] = pd.cut(fr.d, BANDS, labels=False)
    d = fr.d.to_numpy(float)
    p = fr.p.to_numpy(float)
    band = fr.band.to_numpy()
    kv = fr.k.to_numpy(float)
    dn = fr.dn.to_numpy(float)
    apt = fr.apt.to_numpy()
    delta = np.zeros(len(fr))
    for (a, _), idx in fr.groupby(['apt', 'day']).indices.items():
        if not np.isfinite(kv[idx[0]]) or a not in par['tc2']:
            continue
        delta[idx] = day_delta(a == LIRF, d[idx], p[idx], band[idx], kv[idx[0]], dn[idx[0]], par, par['tc2'][a], dither)
    lirf = apt == LIRF
    part_b = np.where(~lirf, delta, 0.0)
    part_c = np.where(lirf & (d <= DMAX), np.clip(delta, -LIRF_CLIP, LIRF_CLIP) * LIRF_DOSE, 0.0)
    total = part_b + part_c
    orig = fr[TARGET].to_numpy(float)
    new = np.where(total != 0, np.maximum(p + total, FLOOR), orig)
    upd = pd.Series(new, index=fr[KEY].to_numpy())
    out = b.copy()
    hit = out[KEY].isin(upd.index)
    out.loc[hit, TARGET] = upd.reindex(out.loc[hit, KEY]).to_numpy()
    return out
