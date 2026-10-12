import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from prc.paths import RANKING
from prc.otimista.inputs import FINAL_DIR, OTIMISTA_MODELS
from prc.otimista.lib import load_holdout, holdout_cats
from prc.otimista import frames as OF
from prc.otimista.final_ctx import CtxFinal, FIN, POOL, POOL_APPLY, PARAMS_S4, PARAMS_POOL, SEEDS

def read_pred(path):
    d = pd.read_parquet(path); d['MVT_ID_mvt'] = d.MVT_ID_mvt.astype('int64'); assert d.MVT_ID_mvt.is_unique
    return d.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)


def build_frame(a, pred, cats, log):
    t0 = time.time(); months = [int(x) for x in a.months.split(',')]
    rk = OF.build_final(a.rank, a.own, a.pts, a.arr, months, pred, cats, a.rt, a.legrt)
    assert rk.pred.notna().all(), 'pred does not cover all DEP rows of the requested months'
    log(f'2026 frame {len(rk)} rows, months {sorted(rk.mes.unique().tolist())}, {time.time()-t0:.0f}s'); return rk, months


def compose(cx, cS, eS, cP, eP, T, lebl_chain=None):
    rk = cx.rk; pred = rk.pred.to_numpy(float); apt = cx.aptr; n = len(rk)
    c = np.where(eS, cS, np.where(eP, cP, 0.0)); elig = eS | eP; camada = np.full(n, '', dtype=object); dose = np.zeros(n); novo = pred.copy()
    s4 = eS & np.isin(apt, FIN); camada[s4] = 'S4'; dose[s4] = 1.0; novo[s4] = np.maximum(pred[s4] + c[s4], 0.0)
    lb = eP & (apt == 'LEBL'); camada[lb] = 'LEBL_50_50'; dose[lb] = 1.0
    chain = pred.copy()
    if lebl_chain is not None:
        ch = rk.MVT_ID_mvt.astype('int64').map(lebl_chain).to_numpy(float); ok = np.isfinite(ch); chain[ok] = ch[ok]
    novo[lb] = 0.5 * chain[lb] + 0.5 * np.maximum(pred[lb] + c[lb], 0.0)
    eg = eP & (apt == 'EGLL'); camada[eg] = 'EGLL_x0.5'; dose[eg] = 0.5; novo[eg] = np.maximum(pred[eg] + 0.5 * c[eg], 0.0)
    lm = eP & (apt == 'LEMD')
    if T in (1, 2):
        new = np.maximum(pred[lm] + 0.5 * c[lm], 0.0); novo[lm] = np.where(np.abs(new - pred[lm]) >= 0.5, new, pred[lm]); camada[lm] = 'LEMD_jan_like_x0.5'; dose[lm] = 0.5
    else:
        novo[lm] = np.maximum(pred[lm] + 0.5 * c[lm], 0.0); camada[lm] = 'LEMD_jul_like_x0.5'; dose[lm] = 0.5
    novo = np.maximum(novo, 60.0)
    return pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.astype('int64').to_numpy(), 'apt': apt, 'mes': rk.mes.to_numpy(), 'elig': elig, 'c': c, 'camada': camada, 'dose': dose, 'pred': pred, 'novo': novo,
                         'gate': rk.gate.to_numpy(), 'adsb_cov_raw': rk.adsb_cov_raw.to_numpy(), 'dia': rk.dia.to_numpy()})


def rms_table(out):
    x = out[out.elig].copy(); x['d'] = x.novo - x.pred
    g = x.groupby(['apt', 'mes', 'camada']).agg(n=('c', 'size'), media_c=('c', 'mean'), rms_c=('c', lambda s: np.sqrt((s ** 2).mean())), media_d=('d', 'mean'), rms_d=('d', lambda s: np.sqrt((s ** 2).mean())),
                                                 p5_d=('d', lambda s: s.quantile(.05)), p95_d=('d', lambda s: s.quantile(.95)), n_alt=('d', lambda s: int((s.abs() > 1e-9).sum())))
    return g.round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='parquet MVT_ID_mvt, TAXITIME_SEC_mvt (v36 do final)'); ap.add_argument('--out', required=True)
    ap.add_argument('--months', default='2,6'); ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--seeds', default='3,11,29')
    ap.add_argument('--rank', default=str(RANKING)); ap.add_argument('--own', default=str(FINAL_DIR / 'adsb_own_cs_final.parquet')); ap.add_argument('--pts', default=str(FINAL_DIR / 'e2_pts_final.parquet'))
    ap.add_argument('--arr', default=str(FINAL_DIR / 'e2_arr_final.parquet')); ap.add_argument('--rt', default=str(FINAL_DIR / 'rt_final.parquet')); ap.add_argument('--legrt', default=str(FINAL_DIR / 'leg_rt_final.parquet')); ap.add_argument('--ox', default=str(FINAL_DIR / 'ox_final.parquet'))
    ap.add_argument('--lebl-chain', default=None, help='parquet MVT_ID_mvt, TAXITIME_SEC_mvt com a cadeia LEBL (lebl_chain) de 2026; sem ele a metade "cadeia" do LEBL 50/50 e o proprio pred')
    ap.add_argument('--gate-eddf', type=float, default=120.0, help='EDDF: eligible only with adsb_first_off >= this value (s before MVT); negative disables'); ap.add_argument('--zero-days', default='2026-06-11', help='dias tratados como adsb_cov=0 (virgula); vazio desliga')
    ap.add_argument('--thr', type=float, default=0.15); ap.add_argument('--tag', default=None, help='suffix of the drift tables (default: pred file name)')
    ap.add_argument('--drift-dir', default=None, help='default: directory of --out'); ap.add_argument('--model-dir', default=str(OTIMISTA_MODELS)); ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--s4-only', action='store_true'); ap.add_argument('--pool-only', action='store_true')
    a = ap.parse_args(); t0 = time.time(); seeds = tuple(int(x) for x in a.seeds.split(',')); tag = a.tag or Path(a.pred).stem; drift_dir = Path(a.drift_dir or Path(a.out).parent)
    out_path = Path(a.out); assert not out_path.exists(), f'{out_path} already exists: not overwriting (choose another --out)'
    log = lambda *x, **k: print(*x, **{**k, 'flush': True})
    pred = read_pred(a.pred); lebl_chain = read_pred(a.lebl_chain) if a.lebl_chain else None
    hold = load_holdout(); cats = holdout_cats(hold)
    rk, months = build_frame(a, pred, cats, log); rep = dict(pred=str(a.pred), months=months, seeds=list(seeds), jobs=a.jobs, gate_eddf=a.gate_eddf, zero_days=a.zero_days, thr=a.thr, lebl_chain=a.lebl_chain, T={}); parts = []
    for T in months:
        tT = time.time(); cx = CtxFinal(T, rk, hold=hold, ox_final=a.ox, thr=a.thr, gate_eddf=(a.gate_eddf if a.gate_eddf >= 0 else None), zero_days=[d for d in a.zero_days.split(',') if d], log=log)
        cx.save_drift(drift_dir, tag); r = dict(n=int(len(cx.rk)), gates=cx.rk.gate.value_counts().to_dict(), keeps_n={g: {ap: len(v) for ap, v in k.items()} for g, k in cx.keeps.items()}, drift_csv=str(drift_dir / f'drift_T{T}_{tag}.csv'))
        n = len(cx.rk); cS = np.zeros(n); eS = np.zeros(n, bool); cP = np.zeros(n); eP = np.zeros(n, bool)
        if not a.pool_only:
            bs, key, info, ntr, cached = cx.fit_or_load('s4', FIN, False, PARAMS_S4, seeds, a.jobs, a.model_dir, not a.no_cache); cS, eS = cx.apply26(bs, FIN, False)
            r['s4'] = dict(key=key, cache=cached, n_treino=ntr, fontes={str(k): v['n'] for k, v in info.items()}, zeradas={str(k): v['zeradas'] for k, v in info.items() if v['zeradas']}, aplicado=int(eS.sum())); cx.clear()
        if not a.s4_only:
            bs, key, info, ntr, cached = cx.fit_or_load('pool', POOL, True, PARAMS_POOL, seeds, a.jobs, a.model_dir, not a.no_cache); cP, eP = cx.apply26(bs, POOL_APPLY, True)
            r['pool'] = dict(key=key, cache=cached, n_treino=ntr, fontes={str(k): v['n'] for k, v in info.items()}, zeradas={str(k): v['zeradas'] for k, v in info.items() if v['zeradas']}, aplicado=int(eP.sum())); cx.clear()
        parts.append(compose(cx, cS, eS, cP, eP, T, lebl_chain)); r['seconds'] = round(time.time() - tT, 1); rep['T'][str(T)] = r
        log(f'[T={T}] done: eligible {int(parts[-1].elig.sum())} de {n}, {r["seconds"]} s')
    out = pd.concat(parts, ignore_index=True)
    assert out.MVT_ID_mvt.is_unique and np.isfinite(out.c).all() and np.isfinite(out.novo).all() and (out.novo >= 60).all() and len(out) == len(rk)
    out_path.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(out_path, index=False); rms = rms_table(out); rms.to_csv(str(out_path) + '.rms.csv')
    rep['n_linhas'] = int(len(out)); rep['n_eligible'] = int(out.elig.sum()); rep['n_changed'] = int((np.abs(out.novo - out.pred) > 1e-9).sum()); rep['seconds_total'] = round(time.time() - t0, 1); rep['out'] = str(out_path)
    json.dump(rep, open(str(out_path) + '.report.json', 'w'), indent=1, default=str)
    log(rms.to_string()); log(f'OK {len(out)} rows, {rep["n_eligible"]} elegiveis, {rep["n_changed"]} alteradas -> {out_path} ({rep["seconds_total"]} s)')


if __name__ == '__main__':
    main()
