import argparse
import pandas as pd
from prc.paths import RANKING
from prc.regime.model import MODEL_DIR, fit, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--oof')
    ap.add_argument('--oof-col', default='pred')
    ap.add_argument('--pred')
    ap.add_argument('--out')
    ap.add_argument('--models', default=str(MODEL_DIR))
    ap.add_argument('--months', default='1,2,6,7')
    a = ap.parse_args()
    if a.fit:
        print(fit(a.oof, a.oof_col, a.models))
    if a.apply:
        base = pd.read_parquet(a.pred)
        out = layer(base, [int(x) for x in a.months.split(',')], RANKING, a.models)
        out['MVT_ID_mvt'] = out['MVT_ID_mvt'].astype('float64')
        out.to_parquet(a.out, index=False)
        print(len(out), float(((out.TAXITIME_SEC_mvt - base.TAXITIME_SEC_mvt) ** 2).mean() ** 0.5))


if __name__ == '__main__':
    main()
