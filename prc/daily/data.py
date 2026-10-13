from pathlib import Path
import numpy as np
import pandas as pd
from prc.paths import EUROCONTROL_DIR, TRAINING_DIR
from prc.daily.constants import APTS, KEY

COLS = [KEY, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']


def load_k(years, directory=EUROCONTROL_DIR):
    parts = []
    for y in years:
        f = pd.read_csv(Path(directory) / f'all_pre_departure_delays_{y}.csv', usecols=['FLT_DATE', 'APT_ICAO', 'FLT_DEP_1', 'DLY_ALL_PRE_2'])
        f = f[f.APT_ICAO.isin(APTS)]
        parts.append(pd.DataFrame({
            'apt': f.APT_ICAO.to_numpy(),
            'day': pd.to_datetime(f.FLT_DATE.astype(str).str[:10]).to_numpy(),
            'k': 60.0 * f.DLY_ALL_PRE_2.to_numpy(float),
            'count': f.FLT_DEP_1.to_numpy(float),
        }))
    return pd.concat(parts, ignore_index=True)


def frame(raw, extra=None):
    r = raw[(raw.PHASE_mvt == 'DEP') & raw.ADEP_mvt.isin(APTS)].drop_duplicates(KEY)
    mvt = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True)
    sch = pd.to_datetime(r.SCHED_TIME_UTC_mvt, utc=True)
    out = pd.DataFrame({
        KEY: r[KEY].round().astype('int64').to_numpy(),
        'apt': r.ADEP_mvt.to_numpy(),
        'day': mvt.dt.tz_localize(None).dt.normalize().to_numpy(),
        'month': mvt.dt.month.to_numpy(),
        'd': (mvt - sch).dt.total_seconds().to_numpy(),
    })
    if extra is not None:
        out['y'] = r[extra].to_numpy(float)
    return out.reset_index(drop=True)


def departures(path, months=None):
    out = frame(pd.read_parquet(path, columns=COLS))
    if months is not None:
        out = out[out.month.isin(list(months))].reset_index(drop=True)
    return out


def labelled_departures(directory=TRAINING_DIR):
    files = sorted(Path(directory).glob('training_2025-*.parquet'))
    parts = [frame(pd.read_parquet(f, columns=COLS + ['TAXITIME_SEC_mvt']), 'TAXITIME_SEC_mvt') for f in files]
    return pd.concat(parts, ignore_index=True)
