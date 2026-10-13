import argparse
import pandas as pd
from prc.daily.fit import PARAMS_PATH, fit_params
from prc.daily.layer import layer, load_params
from prc.paths import RANKING


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--oof')
    ap.add_argument('--oof-col', default='prediction')
    ap.add_argument('--pred')
    ap.add_argument('--out')
    ap.add_argument('--params', default=str(PARAMS_PATH))
    ap.add_argument('--months', default='1,2,6,7')
    ap.add_argument('--dither', action='store_true')
    a = ap.parse_args()
    if a.fit:
        p = fit_params(a.oof, a.oof_col, a.params, a.dither)
        print({k: p[k] for k in ('tau', 'sig_tau', 'cmed')}, len(p['v']), len(p['tc2']))
    if a.apply:
        base = pd.read_parquet(a.pred)
        out = layer(base, [int(x) for x in a.months.split(',')], load_params(a.params), RANKING, a.dither)
        out['MVT_ID_mvt'] = out['MVT_ID_mvt'].astype('float64')
        out.to_parquet(a.out, index=False)
        print(len(out), float(((out.TAXITIME_SEC_mvt - base.TAXITIME_SEC_mvt) ** 2).mean() ** 0.5))


if __name__ == '__main__':
    main()
