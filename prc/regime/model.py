import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from prc.paths import ARTIFACTS, RANKING, TRAINING_DIR, ensure
from prc.daily.fit import read_oof
from prc.regime.constants import (BASE_FEATURES, DOSE, FLOOR, KEY, LAMBDA, PARAMS, ROUNDS, SEEDS, TARGET, TRAIN_MONTHS,
                                  Y_MAX, ZERO_AIRPORTS)
from prc.regime.features import feature_columns, frame, read_raw

MODEL_DIR = ARTIFACTS / 'models' / 'regime'


def training_frame(oof_path, pred_col='pred'):
    raw = pd.concat([read_raw(Path(TRAINING_DIR) / f'training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet') for m in TRAIN_MONTHS])
    raw[KEY] = raw[KEY].round().astype('int64')
    f = frame(raw)
    f[KEY] = f[KEY].astype('int64')
    raw_y = pd.concat([pd.read_parquet(Path(TRAINING_DIR) / f'training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet',
                                       columns=[KEY, TARGET]) for m in TRAIN_MONTHS])
    raw_y[KEY] = raw_y[KEY].round().astype('int64')
    oof = read_oof(oof_path, pred_col).rename(columns={pred_col: 'p'})
    oof[KEY] = oof[KEY].round().astype('int64')
    t = f.merge(raw_y.drop_duplicates(KEY), on=KEY).merge(oof.drop_duplicates(KEY), on=KEY)
    t = t[t[TARGET] < Y_MAX].reset_index(drop=True)
    t['en'] = t[TARGET] - t.p.clip(lower=0)
    return t


def design(df, cats):
    d = df.copy()
    d['aptc'] = pd.Categorical(d.apt, categories=cats)
    return d


def fit(oof_path, pred_col='pred', out_dir=MODEL_DIR):
    t = training_frame(oof_path, pred_col)
    cats = sorted(t.apt.unique())
    t = design(t, cats)
    x = feature_columns(t)
    ensure(out_dir)
    for seed in SEEDS:
        p = dict(PARAMS, seed=seed, num_threads=8)
        lgb.train(p, lgb.Dataset(t[BASE_FEATURES + x], t.en), ROUNDS).save_model(str(Path(out_dir) / f'wide_{seed}.txt'))
        lgb.train(p, lgb.Dataset(t[BASE_FEATURES], t.en), ROUNDS).save_model(str(Path(out_dir) / f'base_{seed}.txt'))
    with open(Path(out_dir) / 'meta.json', 'w') as f:
        json.dump({'cats': cats, 'features': x}, f)
    return len(t)


def increment(df, model_dir=MODEL_DIR):
    with open(Path(model_dir) / 'meta.json') as f:
        meta = json.load(f)
    d = design(df, meta['cats'])
    wide = np.zeros(len(d))
    narrow = np.zeros(len(d))
    for seed in SEEDS:
        wide += lgb.Booster(model_file=str(Path(model_dir) / f'wide_{seed}.txt')).predict(d[BASE_FEATURES + meta['features']]) / len(SEEDS)
        narrow += lgb.Booster(model_file=str(Path(model_dir) / f'base_{seed}.txt')).predict(d[BASE_FEATURES]) / len(SEEDS)
    inc = DOSE * (wide - narrow)
    inc[d.apt.isin(ZERO_AIRPORTS).to_numpy()] = 0.0
    return pd.DataFrame({KEY: df[KEY].astype('int64').to_numpy(), 'month': df.month.to_numpy(), 'inc': inc})


def layer(base, months, ranking=RANKING, model_dir=MODEL_DIR):
    raw = read_raw(ranking)
    raw[KEY] = raw[KEY].round().astype('int64')
    inc = increment(frame(raw), model_dir).drop_duplicates(KEY).set_index(KEY)
    out = base.copy()
    ids = out[KEY].round().astype('int64')
    m = ids.map(inc.month)
    step = ids.map(inc.inc).fillna(0.0) * m.map(LAMBDA).fillna(0.0) * m.isin(months)
    new = out[TARGET].to_numpy(float) + step.to_numpy()
    changed = step.to_numpy() != 0
    new[changed] = np.maximum(new[changed], FLOOR)
    out[TARGET] = new
    return out
