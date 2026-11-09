"""Analyses requested by the reviewers of the npj Digital Medicine revision.

Used by ``notebooks/04_revision_analyses.ipynb`` (and the thin CLI
``src/run_revision.py`` for the long refits). Everything builds on the public
modules (``cohort_io``, ``seasonality_lib``, ``stage_competition``,
``lme_binned``, ``publication_figures``) so that predictors, residualisation and
the fitter are exactly those of ``02_modelling.ipynb``.

Sections (reviewer comment IDs in brackets):

1. Data preparation identical to 02_modelling.
2. Fitting / prediction helpers.
3. Out-of-sample validation [R1-M2, R3-M1].
4. Within- vs between-person temperature [R1-M6].
5. Country clustering and regional robustness [R1-M3].
6. Monthly-threshold robustness and adherence [R1-m5].
7. HRV lower bound [R1-M5] and daily-data helpers [R1-M5, R1-m5, R2-3].
8. Extra model combinations [R3-M2].
9. Post-hoc statistics [R3-M5, R1-m15, R1-m9, R1-m3].
10. Figures [R3-M6, R3-M7, R3-m1, Figs 2-4 on held-out data].

Every judgement call that needs the authors' sign-off is marked ``REVIEW:``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import cohort_io as cio
import file_io as fio
import lme_binned
import seasonality_lib as sl
import stage_competition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_RAW = dict(sl.METRIC_TARGET_COLS)  # rhr -> bpm column, hrv -> ms column
# Column the models are fitted on (HRV on the log scale, as in 02_modelling).
TARGET_FIT = {'rhr': 'avg_resting_heart_rate_bpm', 'hrv': 'target_log'}
UNITS = {'rhr': 'bpm', 'hrv': 'ms'}
SUMMARY_RANGE = {'rhr': (30.0, 150.0), 'hrv': (5.0, 250.0)}  # as 02_modelling
DEFAULT_RUN_TAG = {'rhr': 'rhr_rerun', 'hrv': 'hrv_rerun'}  # as 02_modelling
NEEDED_COLS = [
    'user_id', 'activity_period', 'gender', 'age', 'lat_e7', 'lng_e7',
    'un_region', 'country_2_alpha_cd', 'dl_monthly', 'avg_temp_mean',
    'avg_temp_min', 'avg_temp_max',
]
LAT_BAND_BINS = [-90, -60, -30, 0, 30, 60, 90]
LAT_BAND_LABELS = ['South Polar', 'Mid South', 'S. Tropics', 'N. Tropics',
                   'Mid North', 'Far North']  # = publication_figures bands
AGE_BINS = [18, 30, 40, 50, 60, 70, 120]
AGE_LABELS = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
RES_TOKEN = re.compile(r'\b(avg_temp_(?:mean|max|min)(?:_sq)?_c_res)\b')
MONTH_HARMONICS = 'month_sin + month_cos + month_sin_2 + month_cos_2'


def revision_dir(data_dir: str, metric: str, *parts: str) -> str:
  """``<data_dir>/results/revision/<metric>/<parts>`` (created)."""
  path = os.path.join(data_dir, fio.RESULTS_SUBDIR, 'revision', metric, *parts)
  fio.make_dirs(path)
  return path


def _write_json(obj: Any, path: str) -> None:
  with fio.open_file(path, 'w') as f:
    json.dump(obj, f, indent=2, default=float)


def _read_json(path: str) -> Any:
  with fio.open_file(path, 'r') as f:
    return json.load(f)


def write_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
  with fio.open_file(path, 'w') as f:
    df.to_csv(f, index=index)
  return path


def read_csv(path: str, **kwargs) -> pd.DataFrame:
  with fio.open_file(path, 'r') as f:
    return pd.read_csv(f, **kwargs)


# ===========================================================================
# 1. Data preparation (mirrors 02_modelling.ipynb)
# ===========================================================================
def add_groups(df: pd.DataFrame) -> pd.DataFrame:
  """Adds lat_band, hemisphere, age_band and age_sex (in place)."""
  lat_deg = df['lat_e7'] / 1e7
  df['lat_band'] = pd.cut(lat_deg, LAT_BAND_BINS, labels=LAT_BAND_LABELS,
                          right=False).astype(str)
  df['hemisphere'] = np.where(lat_deg >= 0, 'Northern', 'Southern')
  df['age_band'] = pd.cut(df['age'], AGE_BINS, labels=AGE_LABELS,
                          right=False).astype(str)
  df['age_sex'] = df['gender'].astype(str) + ' ' + df['age_band']
  return df


def load_prepared(
    metric: str,
    data_dir: Optional[str] = None,
    subsample_frac: float = 0.10,
    thin: Optional[bool] = None,
    random_state: int = 42,
    extra_cols: Sequence[str] = (),
    residualise: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
  """Loads the cohort and builds the predictors exactly like 02_modelling.

  Steps: load -> (RHR) density thinning of the age-22 artefact -> sex /
  longitude features -> Gelman standardisation on the FULL cohort -> month
  harmonics -> log target for HRV -> user subsample (``subsample_frac`` of
  users, seed ``random_state``) -> users with >= 2 months -> temperature
  residualised on the subsample.

  Args:
    metric: 'rhr' | 'hrv'.
    data_dir: Data root (default ``file_io.get_data_dir()``).
    subsample_frac: Share of users kept (02_modelling used 0.10).
    thin: Apply the age-22 thinning (default: RHR only, as 02_modelling).
    random_state: Seed (02_modelling used 42, so the same users are drawn).
    extra_cols: Further cohort columns to keep when present (e.g. a valid-days
      count); absent ones are ignored.
    residualise: Residualise temperature on the returned frame. Set False when
      the caller residualises on a training split itself.

  Returns:
    (frame, run_stats) where run_stats holds the standardisation, the
    temperature OLS, the male share (``seasonality_lib.feature_stats`` layout)
    and the scaling factors.
  """
  data_dir = data_dir or fio.get_data_dir()
  target = TARGET_RAW[metric]
  path = cio.resolve_input_path(data_dir, metric)
  df = cio.load_cohort(data_dir, metric, path=path)
  keep = [c for c in NEEDED_COLS + [target] + list(extra_cols) if c in df]
  df = df[keep].copy()
  if thin is None:
    thin = metric == 'rhr'
  if thin:
    lo, hi = SUMMARY_RANGE[metric]
    df, _ = sl.apply_density_based_thinning(
        df, target_col=target, val_min=lo, val_max=hi,
        random_state=random_state, verbose=False)
  add_groups(df)
  male_share = sl.add_sex_and_longitude_features(df)
  summary = sl.standardise_model_features(df)
  sl.add_month_harmonics(df)
  df['target_log'] = (np.log(df[target]) if metric == 'hrv'
                      else df[target].astype(float))
  df['user_id_n'] = df['user_id'].astype('category').cat.codes
  if subsample_frac < 1.0:
    users = pd.Series(df['user_id_n'].unique()).sample(
        frac=subsample_frac, random_state=random_state)
    df = df[df['user_id_n'].isin(users)].copy()
  counts = df['user_id_n'].value_counts()
  df = df[df['user_id_n'].isin(counts[counts > 1].index)].copy()
  ols = sl.residualise_temperature(df, 'harmonic') if residualise else {}
  run_stats = sl.feature_stats(summary, ols, male_share)
  run_stats['scaling_factors'] = sl.scaling_factors_from_summary(summary)
  df = df.reset_index(drop=True)
  return df, run_stats


def load_raw(metric: str, data_dir: Optional[str] = None,
             thin: Optional[bool] = None, random_state: int = 42
             ) -> pd.DataFrame:
  """Full clean cohort as in 01_cohort_stats (RHR thinned), no subsampling."""
  data_dir = data_dir or fio.get_data_dir()
  target = TARGET_RAW[metric]
  df = cio.load_cohort(data_dir, metric,
                       path=cio.resolve_input_path(data_dir, metric))
  df = df[[c for c in NEEDED_COLS + [target] if c in df]].copy()
  if thin if thin is not None else metric == 'rhr':
    lo, hi = SUMMARY_RANGE[metric]
    df, _ = sl.apply_density_based_thinning(
        df, target_col=target, val_min=lo, val_max=hi,
        random_state=random_state, verbose=False)
  return df.reset_index(drop=True)


def run_stats_from_run(metric: str, data_dir: Optional[str] = None,
                       run_tag: Optional[str] = None
                       ) -> Optional[Dict[str, Any]]:
  """feature_stats.json of the 02_modelling run (+ scaling factors)."""
  data_dir = data_dir or fio.get_data_dir()
  run_dir = os.path.join(data_dir, fio.RESULTS_SUBDIR,
                         run_tag or DEFAULT_RUN_TAG[metric])
  st = sl.read_feature_stats(run_dir)
  if st is not None:
    st['scaling_factors'] = sl.scaling_factors_from_summary(
        st['standardisation'])
  return st


def winner_coefficients(metric: str, data_dir: Optional[str] = None,
                        run_tag: Optional[str] = None) -> Optional[pd.Series]:
  """REML coefficients of the paper's winner (model_<metric>_coefficients_
  interpreted.csv), or None if the run folder is absent."""
  data_dir = data_dir or fio.get_data_dir()
  run_dir = os.path.join(data_dir, fio.RESULTS_SUBDIR,
                         run_tag or DEFAULT_RUN_TAG[metric])
  path = sl.coefficients_csv_path(run_dir, metric)
  if not fio.file_exists(path):
    return None
  c = read_csv(path, index_col=0)['coef']
  return c[~c.index.astype(str).str.endswith(' Var')]


def winner_formula(
    metric: str,
    data_dir: Optional[str] = None,
    run_tag: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
  """Formula of the overall BIC winner of the 02_modelling run.

  REVIEW: reads ``results/<run_tag>/lme-models-results-<metric>.csv``; check
  the printed formula is the paper's Model 5.3 / Stage 6 winner. ``fallback``
  is used when the run table is absent (e.g. on dummy data).
  """
  data_dir = data_dir or fio.get_data_dir()
  run_dir = os.path.join(data_dir, fio.RESULTS_SUBDIR,
                         run_tag or DEFAULT_RUN_TAG[metric])
  path = sl.results_csv_path(run_dir, metric)
  if fio.file_exists(path):
    best = stage_competition.overall_winner(read_csv(path))
    return str(best['formula'])
  if fallback is None:
    raise FileNotFoundError(f'{path} not found and no fallback formula given')
  print(f'[winner_formula] {path} not found; using the fallback formula')
  return fallback


# A reasonable stand-in for the Stage 6 winner, used only when no run table
# exists (dummy data). REVIEW: never used on the real data if the run exists.
FALLBACK_RHS = (
    '(age_c + age_sq_c) * gender_n + (' + MONTH_HARMONICS + ') * (lat_e7_c)'
    ' + (' + MONTH_HARMONICS + ') * (avg_temp_max_c_res + avg_temp_min_c_res)'
)


def fallback_formula(metric: str) -> str:
  return f'{TARGET_FIT[metric]} ~ {FALLBACK_RHS}'


# ===========================================================================
# 2. Fitting / prediction helpers
# ===========================================================================
def fit_lme(formula: str, df: pd.DataFrame, reml: bool = False,
            groups: str = 'user_id_n'):
  """Random-intercept LME with the binned fitter of 02_modelling."""
  model = lme_binned.BinnedMixedLM.from_formula(formula, data=df,
                                                groups=groups)
  return model.fit(reml=reml, method='lbfgs')


def variance_components(result) -> Tuple[float, float]:
  cov_re = result.cov_re
  var_u = float(cov_re.iloc[0, 0] if hasattr(cov_re, 'iloc') else cov_re[0, 0])
  return var_u, float(result.scale)


def fixed_prediction(df: pd.DataFrame, fe_params: pd.Series) -> np.ndarray:
  return sl.predict_from_coefficients(df, fe_params).to_numpy()


def shrunken_intercepts(resid: pd.Series, groups: pd.Series, var_u: float,
                        var_e: float) -> pd.Series:
  """BLUP of a random intercept from fixed-effect residuals."""
  g = resid.groupby(groups)
  return var_u * g.sum() / (var_u * g.count() + var_e)


def coef_table(result) -> pd.DataFrame:
  fe = pd.Series(result.fe_params)
  k = len(fe)
  se = np.asarray(result.bse_fe, float)[:k]
  coef = fe.to_numpy(float)
  z = coef / se
  out = pd.DataFrame({
      'coef': coef, 'std_err': se,
      'ci_lower': coef - 1.959964 * se, 'ci_upper': coef + 1.959964 * se,
      'p': 2 * stats.norm.sf(np.abs(z)),
  }, index=fe.index)
  out.index.name = 'term'
  return out


def fit_summary(result, df: pd.DataFrame, target: str) -> Dict[str, float]:
  var_u, var_e = variance_components(result)
  fe = fixed_prediction(df, result.fe_params)
  var_f = float(np.var(fe))
  total = var_f + var_u + var_e
  return dict(
      n_rows=len(df), n_users=int(df['user_id_n'].nunique()),
      llf=float(result.llf), aic=float(result.aic), bic=float(result.bic),
      var_u=var_u, var_e=var_e, marginal_R2=var_f / total,
      conditional_R2=(var_f + var_u) / total,
      rmse_marginal=float(np.sqrt(np.mean((df[target] - fe) ** 2))),
  )


def residual_columns(formula: str) -> List[str]:
  """Residualised temperature columns used by a formula."""
  return sorted(set(RES_TOKEN.findall(formula)))


# ===========================================================================
# 3. Out-of-sample validation [R1-M2, R3-M1]
# ===========================================================================
def point_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
  y, yhat = np.asarray(y, float), np.asarray(yhat, float)
  err = y - yhat
  sst = np.sum((y - y.mean()) ** 2)
  return dict(
      n=len(y), rmse=float(np.sqrt(np.mean(err ** 2))),
      mae=float(np.mean(np.abs(err))), bias=float(np.mean(err)),
      r2=float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
      pearson_r=float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 2 else np.nan,
  )


def aggregate_metrics(pred: pd.DataFrame, group_col: Optional[str],
                      obs: str = 'y', hat: str = 'pred_marginal',
                      min_rows: int = 30) -> Dict[str, float]:
  """Agreement of monthly mean trajectories (the quantity in Figs 2-4).

  Monthly means of observed and predicted values per (group, month) cell with
  at least ``min_rows`` rows; returns Pearson r, RMSE, NRMSE (RMSE / SD of
  observed cell means: scale free, comparable between RHR and HRV) and the
  ratio of predicted to observed peak-to-trough amplitude (averaged over
  groups). REVIEW: min_rows.
  """
  keys = ['month'] if group_col is None else [group_col, 'month']
  cell = pred.groupby(keys, observed=True).agg(
      o=(obs, 'mean'), p=(hat, 'mean'), n=(obs, 'size')).reset_index()
  cell = cell[cell['n'] >= min_rows]
  if len(cell) < 3:
    return dict(agg_r=np.nan, agg_rmse=np.nan, agg_nrmse=np.nan,
                agg_level_bias=np.nan, agg_shape_rmse=np.nan,
                amplitude_ratio=np.nan, n_cells=len(cell))
  rmse = float(np.sqrt(np.mean((cell['o'] - cell['p']) ** 2)))
  # Level vs shape: a held-out country/region can have a non-zero mean random
  # intercept (a constant offset), which the fixed effects cannot know. Split
  # the error into that offset and the error left after removing it.
  if group_col is None:
    d = cell['o'] - cell['p']
    bias = float(d.mean())
    shape_rmse = float(np.sqrt(np.mean((d - bias) ** 2)))
  else:
    d = cell['o'] - cell['p']
    off = d.groupby(cell[group_col]).transform('mean')
    bias = float(d.mean())
    shape_rmse = float(np.sqrt(np.mean((d - off) ** 2)))
  if group_col is None:
    amp = (cell['p'].max() - cell['p'].min()) / (cell['o'].max() -
                                                 cell['o'].min())
  else:
    rng = cell.groupby(group_col, observed=True).agg(
        po=('p', lambda s: s.max() - s.min()),
        oo=('o', lambda s: s.max() - s.min()))
    amp = float((rng['po'] / rng['oo'].replace(0, np.nan)).mean())
  return dict(agg_r=float(np.corrcoef(cell['o'], cell['p'])[0, 1]),
              agg_rmse=rmse, agg_nrmse=rmse / float(cell['o'].std()),
              agg_level_bias=bias, agg_shape_rmse=shape_rmse,
              amplitude_ratio=float(amp), n_cells=len(cell))


def modal_user_value(df: pd.DataFrame, col: str) -> pd.Series:
  """Most frequent value of ``col`` per user (users can move between groups)."""
  return (df.groupby('user_id_n')[col]
          .agg(lambda s: s.value_counts().index[0]))


def user_split(df: pd.DataFrame, test_frac: float = 0.2,
               random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
  """Participant-level split: every user is entirely in train or in test."""
  users = df['user_id_n'].unique()
  rng = np.random.default_rng(random_state)
  test_users = rng.choice(users, size=int(round(test_frac * len(users))),
                          replace=False)
  test = df['user_id_n'].isin(test_users).to_numpy()
  return ~test, test


def holdout_group_split(df: pd.DataFrame, group_col: str, level: Any,
                        user_level: Optional[pd.Series] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
  """Leave-one-group-out by user: users whose modal group is ``level`` form
  the test set; the model never sees them (no user in both sets).

  REVIEW: users are assigned to their modal group. A user who moved between
  countries / latitude bands, or crossed an age-band boundary during 2022,
  keeps all their months on one side.
  """
  if user_level is None:
    user_level = modal_user_value(df, group_col)
  test_users = user_level.index[user_level == level]
  test = df['user_id_n'].isin(test_users).to_numpy()
  return ~test, test


def evaluate_fold(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    formula: str,
    metric: str,
    calib_frac: float = 0.5,
    random_state: int = 0,
    reml: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
  """Fits on train, evaluates on held-out users.

  * Temperature is re-residualised on the training rows only and the training
    OLS is applied to the test rows (no leakage through the residualisation).
  * Marginal prediction: fixed effects only (what the model can say about a
    new person from their demographics, location, month and temperature).
  * Conditional prediction: each held-out user's months are split at random
    into calibration months (``calib_frac``) used only to estimate that user's
    intercept (BLUP with the training variance components) and evaluation
    months on which both predictions are scored. Every user has >= 2 months,
    so both parts are non-empty. REVIEW: calib_frac = 0.5; an alternative is a
    temporal split (first half of the year -> second half), but that confounds
    the season with the split.
  * HRV is scored on the fitted log scale and, back-transformed with Duan's
    smearing factor from the training residuals, in ms.

  Returns:
    (metrics, test predictions on the evaluation months, coefficient table).
  """
  target = TARGET_FIT[metric]
  train = df.loc[train_mask].copy()
  test = df.loc[test_mask].copy()
  ols = sl.residualise_temperature(train, 'harmonic')
  sl.residualise_temperature(test, params=ols)
  res = fit_lme(formula, train, reml=reml)
  var_u, var_e = variance_components(res)
  fe = res.fe_params

  train_fe = fixed_prediction(train, fe)
  train_resid = train[target].to_numpy() - train_fe
  blup_tr = shrunken_intercepts(pd.Series(train_resid), train['user_id_n']
                                .reset_index(drop=True), var_u, var_e)
  train_cond = train_fe + train['user_id_n'].map(blup_tr).to_numpy()

  rng = np.random.default_rng(random_state)
  test = test.reset_index(drop=True)
  test['_r'] = rng.random(len(test))
  rank = test.groupby('user_id_n')['_r'].rank(method='first')
  n_u = test.groupby('user_id_n')['_r'].transform('size')
  n_cal = np.clip(np.floor(n_u * calib_frac), 1, n_u - 1)
  calib = (rank <= n_cal).to_numpy()
  test['pred_marginal'] = fixed_prediction(test, fe)
  resid = test[target] - test['pred_marginal']
  blup = shrunken_intercepts(resid[calib], test.loc[calib, 'user_id_n'],
                             var_u, var_e)
  test['re_hat'] = test['user_id_n'].map(blup).fillna(0.0)
  test['pred_conditional'] = test['pred_marginal'] + test['re_hat']
  ev = test.loc[~calib].copy()
  ev['y'] = ev[target]

  out: Dict[str, Any] = dict(
      n_train_users=int(train['user_id_n'].nunique()),
      n_test_users=int(test['user_id_n'].nunique()),
      n_eval_rows=int(len(ev)), var_u=var_u, var_e=var_e)
  for name, y, hat in [
      ('train_marginal', train[target], train_fe),
      ('train_conditional', train[target], train_cond),
      ('test_marginal', ev['y'], ev['pred_marginal']),
      ('test_conditional', ev['y'], ev['pred_conditional']),
  ]:
    for k, v in point_metrics(y, hat).items():
      out[f'{name}_{k}'] = v
  if metric == 'hrv':
    smear_m = float(np.mean(np.exp(train_resid)))
    smear_c = float(np.mean(np.exp(train[target].to_numpy() - train_cond)))
    ev['y_ms'] = np.exp(ev['y'])
    ev['pred_marginal_ms'] = np.exp(ev['pred_marginal']) * smear_m
    ev['pred_conditional_ms'] = np.exp(ev['pred_conditional']) * smear_c
    for name, hat in [('test_marginal_ms', 'pred_marginal_ms'),
                      ('test_conditional_ms', 'pred_conditional_ms')]:
      for k, v in point_metrics(ev['y_ms'], ev[hat]).items():
        out[f'{name}_{k}'] = v
  for k, v in aggregate_metrics(ev, None).items():
    out[f'test_global_{k}'] = v
  keep = ['user_id_n', 'month', 'activity_period', 'gender', 'age',
          'age_band', 'age_sex', 'lat_band', 'hemisphere', 'un_region',
          'country_2_alpha_cd', 'y', 'pred_marginal', 'pred_conditional']
  keep += [c for c in ('country_size',) if c in ev]
  keep += [c for c in ('y_ms', 'pred_marginal_ms', 'pred_conditional_ms')
           if c in ev]
  return out, ev[keep], coef_table(res)


SPARSE_LEVEL = 'data-sparse countries'


def add_sparse_country_group(df: pd.DataFrame, max_users: int) -> List[str]:
  """Adds ``country_size``: 'data-sparse countries' for users whose modal
  country has fewer than ``max_users`` users in ``df``, else 'other'.
  Returns the sparse countries. REVIEW: the threshold."""
  uc = modal_user_value(df, 'country_2_alpha_cd')
  n = uc.value_counts()
  sparse = sorted(n.index[n < max_users].astype(str))
  df['country_size'] = np.where(
      df['country_2_alpha_cd'].astype(str).isin(sparse), SPARSE_LEVEL, 'other')
  return sparse


def default_folds(df: pd.DataFrame,
                  countries: Sequence[str] = (),
                  min_users: int = 200,
                  sparse_country_max_users: Optional[int] = None
                  ) -> List[Tuple[str, str, Any]]:
  """Fold list: (fold_id, kind, level).

  * ``user_split``: one participant-level 80/20 split.
  * ``age_sex`` / ``lat_band`` / ``un_region``: leave one level out, for every
    level with at least ``min_users`` users.
  * ``country``: leave out each of ``countries``.
  * ``country_size``: all users from data-sparse countries (fewer than
    ``sparse_country_max_users`` users each) held out together, so the model
    is scored on exactly the countries it has little data for, with enough
    pooled users for stable metrics.

  REVIEW: which countries to hold out (default none; suggest e.g. the
  largest southern-hemisphere country, a tropical one and a far-north one,
  such as AU, BR/SG and NO/FI) and min_users. Holding out a whole level
  (e.g. 'Mid North', ~88% of the HRV cohort) leaves a small training set:
  that fold mostly tests extrapolation and is reported for completeness.
  """
  folds: List[Tuple[str, str, Any]] = [('user_split', 'user_split', None)]
  for kind in ('age_sex', 'lat_band', 'un_region'):
    lev = modal_user_value(df, kind).value_counts()
    for level in lev.index[lev >= min_users]:
      if kind == 'un_region' and str(level) in ('unknown', 'nan'):
        continue
      folds.append((f'{kind}={level}', kind, level))
  if countries:
    cn = modal_user_value(df, 'country_2_alpha_cd').value_counts()
    for c in countries:
      n = int(cn.get(c, 0))
      print(f'[folds] hold-out country {c}: {n:,} users in the fitting sample')
      if n < min_users:
        print(f'[folds]   skipped ({n} < min_users={min_users})')
        continue
      folds.append((f'country={c}', 'country_2_alpha_cd', c))
  if sparse_country_max_users:
    sparse = add_sparse_country_group(df, sparse_country_max_users)
    n_sparse = int(modal_user_value(df, 'country_size').eq(SPARSE_LEVEL).sum())
    print(f'[folds] data-sparse countries (< {sparse_country_max_users:,} '
          f'users each): {len(sparse)} countries, {n_sparse:,} users')
    if n_sparse >= min_users:
      folds.append((f'country_size={SPARSE_LEVEL}', 'country_size',
                    SPARSE_LEVEL))
  return folds


def run_oos_validation(
    df: pd.DataFrame,
    formula: str,
    metric: str,
    out_dir: str,
    folds: Optional[Sequence[Tuple[str, str, Any]]] = None,
    test_frac: float = 0.2,
    random_state: int = 0,
    force: bool = False,
) -> pd.DataFrame:
  """Runs every fold; resumable (a fold with ``metrics.json`` is skipped).

  Writes ``<out_dir>/<fold>/{metrics.json, predictions.csv.gz,
  coefficients.csv}`` and returns the fold metrics table (also written to
  ``<out_dir>/oos_metrics.csv``).
  """
  folds = folds or default_folds(df)
  user_level_cache: Dict[str, pd.Series] = {}
  rows = []
  for fold_id, kind, level in folds:
    safe = re.sub(r'[^A-Za-z0-9=._-]+', '_', fold_id)
    fdir = os.path.join(out_dir, safe)
    mpath = os.path.join(fdir, 'metrics.json')
    if fio.file_exists(mpath) and not force:
      m = _read_json(mpath)
      print(f'[oos] {fold_id}: done earlier, loaded')
    else:
      if kind == 'user_split':
        tr, te = user_split(df, test_frac, random_state)
      else:
        if kind not in user_level_cache:
          user_level_cache[kind] = modal_user_value(df, kind)
        tr, te = holdout_group_split(df, kind, level, user_level_cache[kind])
      if te.sum() == 0 or tr.sum() == 0:
        print(f'[oos] {fold_id}: empty split, skipped')
        continue
      print(f'[oos] {fold_id}: fitting on {tr.sum():,} rows, '
            f'testing on {te.sum():,} rows')
      m, pred, coefs = evaluate_fold(df, tr, te, formula, metric,
                                     random_state=random_state)
      m.update(fold=fold_id, kind=kind, level=None if level is None
               else str(level))
      fio.make_dirs(fdir)
      with fio.open_file(os.path.join(fdir, 'predictions.csv.gz'), 'wb') as f:
        pred.to_csv(f, index=False, compression='gzip')
      write_csv(coefs, os.path.join(fdir, 'coefficients.csv'), index=True)
      _write_json(m, mpath)
    rows.append(m)
  table = pd.DataFrame(rows)
  write_csv(table, os.path.join(out_dir, 'oos_metrics.csv'))
  return table


def load_fold_predictions(out_dir: str, fold: str = 'user_split'
                          ) -> pd.DataFrame:
  safe = re.sub(r'[^A-Za-z0-9=._-]+', '_', fold)
  with fio.open_file(os.path.join(out_dir, safe, 'predictions.csv.gz'),
                     'rb') as f:
    return pd.read_csv(f, compression='gzip')


def oos_summary_table(table: pd.DataFrame, metric: str) -> pd.DataFrame:
  """Compact table for the supplement (one row per fold)."""
  sfx = '_ms' if metric == 'hrv' else ''
  cols = {
      'fold': 'Held-out set', 'n_test_users': 'Test users',
      'train_marginal_rmse': 'Train RMSE (marginal)',
      'train_conditional_rmse': 'Train RMSE (conditional)',
      f'test_marginal{sfx}_rmse': f'Test RMSE (marginal{", ms" if sfx else ""})',
      f'test_conditional{sfx}_rmse':
          f'Test RMSE (conditional{", ms" if sfx else ""})',
      'test_marginal_r2': 'Test R² (marginal)',
      'test_conditional_r2': 'Test R² (conditional)',
      'test_global_agg_r': 'Monthly-mean r',
      'test_global_agg_nrmse': 'Monthly-mean NRMSE',
      'test_global_agg_level_bias': 'Monthly-mean level offset',
      'test_global_agg_shape_rmse': 'Monthly-mean RMSE after offset',
      'test_global_amplitude_ratio': 'Amplitude ratio (pred/obs)',
  }
  cols = {k: v for k, v in cols.items() if k in table}
  return table[list(cols)].rename(columns=cols)


def group_agreement(pred: pd.DataFrame, group_col: str, metric: str,
                    log_scale: bool = True, min_rows: int = 30
                    ) -> pd.DataFrame:
  """Per-group monthly-trajectory agreement on held-out data [R3-M7]."""
  obs, hat = ('y', 'pred_marginal')
  if metric == 'hrv' and not log_scale:
    obs, hat = 'y_ms', 'pred_marginal_ms'
  rows = []
  for level, g in pred.groupby(group_col, observed=True):
    m = aggregate_metrics(g, None, obs, hat, min_rows)
    m.update({group_col: level, 'n_users': g['user_id_n'].nunique()})
    rows.append(m)
  return pd.DataFrame(rows)


# ===========================================================================
# 4. Within- vs between-person temperature [R1-M6]
# ===========================================================================
def add_within_between(df: pd.DataFrame, res_cols: Sequence[str]
                       ) -> List[str]:
  """Person-mean centring of the residualised temperatures (in place).

  For each linear residual ``x`` (e.g. ``avg_temp_max_c_res``):
  ``x_b`` = the user's mean of ``x`` over their months (between-person) and
  ``x_w = x - x_b`` (within-person). Squared terms
  (``avg_temp_max_sq_c_res``) become ``(x_w)**2`` and ``(x_b)**2``.
  REVIEW: the squared within term is the square of the within deviation (not
  re-centred); this keeps the quadratic shape interpretable as a function of
  the deviation from the person's own mean.

  Returns the list of new column names.
  """
  new = []
  linear = {c for c in res_cols if '_sq_' not in c}
  for sq in (c for c in res_cols if '_sq_' in c):
    linear.add(sq.replace('_sq_c_res', '_c_res'))
  for c in sorted(linear):
    b = df.groupby('user_id_n')[c].transform('mean')
    df[f'{c}_b'] = b
    df[f'{c}_w'] = df[c] - b
    new += [f'{c}_b', f'{c}_w']
  for sq in (c for c in res_cols if '_sq_' in c):
    base = sq.replace('_sq_c_res', '_c_res')
    df[f'{sq}_w'] = df[f'{base}_w'] ** 2
    df[f'{sq}_b'] = df[f'{base}_b'] ** 2
    new += [f'{sq}_w', f'{sq}_b']
  return new


def within_between_formula(formula: str) -> str:
  """Every residualised temperature term -> its within part (keeping the
  season interactions), plus the between parts as main effects.

  REVIEW: between parts enter as main effects only (they are constant within
  a person, so their interaction with season would be identified only
  through who is observed in which month).
  """
  cols = residual_columns(formula)
  wb = RES_TOKEN.sub(lambda m: m.group(1) + '_w', formula)
  return wb + ''.join(f' + {c}_b' for c in cols)


def run_within_between(df: pd.DataFrame, formula: str, metric: str,
                       out_dir: str, force: bool = False
                       ) -> Dict[str, Any]:
  """Refits the winner with the within/between split (ML, same rows).

  Writes ``within_between_coefficients.csv`` (original, within, between
  coefficients side by side, plus a Wald test of within = between) and
  ``within_between_fit.json`` (BIC of both models).
  """
  cpath = os.path.join(out_dir, 'within_between_coefficients.csv')
  jpath = os.path.join(out_dir, 'within_between_fit.json')
  if fio.file_exists(cpath) and fio.file_exists(jpath) and not force:
    return dict(coefficients=read_csv(cpath), fit=_read_json(jpath))
  target = TARGET_FIT[metric]
  df = df.copy()
  add_within_between(df, residual_columns(formula))
  wb_formula = within_between_formula(formula)
  base = fit_lme(formula, df)
  wb = fit_lme(wb_formula, df)
  cb, cw = coef_table(base), coef_table(wb)
  k = len(wb.fe_params)
  cov = pd.DataFrame(np.asarray(wb.cov_params())[:k, :k],
                     index=wb.fe_params.index, columns=wb.fe_params.index)
  rows = []
  for term in cb.index:
    w_term = RES_TOKEN.sub(lambda m: m.group(1) + '_w', term)
    row = dict(term=term, coef_original=cb.loc[term, 'coef'],
               se_original=cb.loc[term, 'std_err'])
    if w_term in cw.index:
      row.update(coef_within=cw.loc[w_term, 'coef'],
                 se_within=cw.loc[w_term, 'std_err'])
    b_term = term + '_b' if RES_TOKEN.fullmatch(term) else None
    if b_term and b_term in cw.index:
      row.update(coef_between=cw.loc[b_term, 'coef'],
                 se_between=cw.loc[b_term, 'std_err'])
      d = cw.loc[w_term, 'coef'] - cw.loc[b_term, 'coef']
      v = (cov.loc[w_term, w_term] + cov.loc[b_term, b_term]
           - 2 * cov.loc[w_term, b_term])
      row.update(diff_w_minus_b=d, z_diff=d / np.sqrt(v),
                 p_diff=2 * stats.norm.sf(abs(d / np.sqrt(v))))
    rows.append(row)
  table = pd.DataFrame(rows)
  fit = dict(formula=formula, within_between_formula=wb_formula,
             original=fit_summary(base, df, target),
             within_between=fit_summary(wb, df, target))
  write_csv(table, cpath)
  _write_json(fit, jpath)
  write_csv(cw, os.path.join(out_dir, 'within_between_full_coefficients.csv'),
            index=True)
  return dict(coefficients=table, fit=fit)


def within_temperature_curve(coefs: pd.DataFrame, run_stats: Mapping[str, Any],
                             metric: str, deltas=np.arange(-10, 10.5, 0.5),
                             column: str = 'coef_within') -> pd.DataFrame:
  """Model-implied change for a within-person shift of ``delta`` degC applied
  to all residualised temperature terms, averaged over the year (the season
  interactions average out because the harmonics have zero annual mean).

  REVIEW: assumes the monthly mean, max and min temperatures move together by
  ``delta``; HRV is converted to % change.
  """
  factors = run_stats['scaling_factors']
  c = coefs.set_index('term')[column].dropna()
  out = []
  for d in deltas:
    tot = 0.0
    for term, beta in c.items():
      if not RES_TOKEN.fullmatch(term):
        continue  # interactions with the season average to zero
      if '_sq_' in term:
        base = term.replace('_sq_c_res', '_c')
        tot += beta * (d / factors[base]) ** 2
      else:
        tot += beta * d / factors[term.replace('_c_res', '_c')]
    out.append(dict(delta_c=d, effect=(np.exp(tot) - 1) * 100
                    if metric == 'hrv' else tot))
  return pd.DataFrame(out)


# ===========================================================================
# 5. Country clustering and regional robustness [R1-M3]
# ===========================================================================
def country_counts(df: pd.DataFrame) -> pd.DataFrame:
  """Users / user-months per country (supplementary table)."""
  user_country = modal_user_value(df, 'country_2_alpha_cd')
  region = df.groupby('country_2_alpha_cd')['un_region'].agg(
      lambda s: s.value_counts().index[0])
  t = pd.DataFrame({
      'users': user_country.value_counts(),
      'user_months': df['country_2_alpha_cd'].value_counts(),
  }).fillna(0).astype(int)
  t['un_region'] = region.reindex(t.index)
  t['share_users_%'] = 100 * t['users'] / t['users'].sum()
  t.index.name = 'country'
  return t.sort_values('users', ascending=False)


def country_variance_share(df: pd.DataFrame, result, metric: str
                           ) -> Dict[str, float]:
  """Country share of the variance left after the fixed effects.

  Two estimates on users' mean marginal residuals (one row per user, modal
  country):
  * a random-intercept model ``mean_resid ~ 1 | country`` (binned fitter),
    giving the between-country variance;
  * the same expressed as a share of the total marginal residual variance
    (country + person + month-to-month).
  REVIEW: user means carry sampling noise var_e / n_i, which ends up in the
  within-country part, not in the country part, so the country variance is
  not inflated by it.
  """
  target = TARGET_FIT[metric]
  r = df[target].to_numpy() - fixed_prediction(df, result.fe_params)
  u = pd.DataFrame({'user_id_n': df['user_id_n'], 'r': r}).groupby(
      'user_id_n')['r'].mean().rename('mean_resid').to_frame()
  u['country'] = modal_user_value(df, 'country_2_alpha_cd')
  u['country_n'] = u['country'].astype('category').cat.codes
  res_c = lme_binned.BinnedMixedLM.from_formula(
      'mean_resid ~ 1', data=u, groups='country_n').fit(reml=True,
                                                         method='lbfgs')
  var_c, var_within_c = variance_components(res_c)
  var_u, var_e = variance_components(result)
  total = float(np.var(r))
  return dict(var_country=var_c, var_user_means_within_country=var_within_c,
              var_user_model=var_u, var_month_model=var_e,
              total_marginal_resid_var=total,
              country_share_of_marginal_resid=var_c / total,
              country_share_of_between_person=var_c / (var_c + var_within_c),
              n_countries=int(u['country'].nunique()), n_users=len(u))


def capped_subsample(df: pd.DataFrame, cap: int = 300, random_state: int = 0
                     ) -> pd.DataFrame:
  """At most ``cap`` users per (modal) country. REVIEW: cap."""
  uc = modal_user_value(df, 'country_2_alpha_cd').rename('country_modal')
  keep = (uc.reset_index().groupby('country_modal', group_keys=False)
          .apply(lambda g: g.sample(min(cap, len(g)),
                                    random_state=random_state))['user_id_n'])
  out = df[df['user_id_n'].isin(keep)].copy()
  out['country_modal'] = out['user_id_n'].map(uc)
  return out


def fit_country_nested(df: pd.DataFrame, formula: str):
  """Users nested in countries: country random intercept + user variance
  component (statsmodels MixedLM; slow, so use on a capped subsample).

  Users are assigned to their modal country, so the user and country
  intercepts are nested; for users who never move country this is the same
  as crossed random effects. REVIEW: the earlier plan said 'crossed'; with
  modal-country assignment the two coincide, and nested is what statsmodels
  can fit.
  """
  import statsmodels.formula.api as smf  # pylint: disable=g-import-not-at-top
  model = smf.mixedlm(formula, data=df, groups='country_modal',
                      re_formula='1',
                      vc_formula={'user': '0 + C(user_id_n)'})
  return model.fit(reml=True, method=['bfgs', 'lbfgs', 'cg'])


def run_country_robustness(df: pd.DataFrame, formula: str, metric: str,
                           out_dir: str, cap: int = 300,
                           fit_nested: bool = True, force: bool = False
                           ) -> Dict[str, Any]:
  """R1-M3: country table, variance share, capped refits, country FE check.

  Fits on the capped subsample:
  (a) the winner with user intercepts only (baseline on the same rows);
  (b) users nested in countries (country random intercept), if fit_nested;
  and on the full fitting sample:
  (c) the winner plus country fixed effects ``C(country_2_alpha_cd)``.
  Writes ``country_*.csv/json`` and returns the pieces.
  """
  jpath = os.path.join(out_dir, 'country_variance_share.json')
  cpath = os.path.join(out_dir, 'country_coefficients_compare.csv')
  if fio.file_exists(jpath) and fio.file_exists(cpath) and not force:
    return dict(variance=_read_json(jpath), coefficients=read_csv(cpath),
                counts=read_csv(os.path.join(out_dir, 'country_counts.csv')))
  counts = country_counts(df)
  write_csv(counts, os.path.join(out_dir, 'country_counts.csv'), index=True)
  base_full = fit_lme(formula, df)
  var = country_variance_share(df, base_full, metric)
  _write_json(var, jpath)
  cols = {'full_user_RI': coef_table(base_full)['coef']}
  se = {'full_user_RI': coef_table(base_full)['std_err']}
  fe_formula = formula + ' + C(country_2_alpha_cd)'
  fe = coef_table(fit_lme(fe_formula, df))
  cols['full_country_FE'], se['full_country_FE'] = fe['coef'], fe['std_err']
  capped = capped_subsample(df, cap)
  b = coef_table(fit_lme(formula, capped))
  cols['capped_user_RI'], se['capped_user_RI'] = b['coef'], b['std_err']
  if fit_nested:
    try:
      nres = fit_country_nested(capped, formula)
      cols['capped_nested_country'] = nres.fe_params
      se['capped_nested_country'] = nres.bse_fe
      var['nested_country_var'] = float(nres.cov_re.iloc[0, 0])
      var['nested_user_var'] = float(nres.vcomp[0])
      var['nested_resid_var'] = float(nres.scale)
      _write_json(var, jpath)
    except Exception as e:  # pylint: disable=broad-except
      print(f'[country] nested fit failed: {e!r}')
  table = pd.DataFrame(cols).join(pd.DataFrame(se), rsuffix='_se')
  table = table[~table.index.str.startswith('C(country')]
  table.index.name = 'term'
  write_csv(table, cpath, index=True)
  return dict(variance=var, coefficients=table.reset_index(), counts=counts)


def fit_by_group(df: pd.DataFrame, formula: str, group_col: str,
                 min_users: int = 2000) -> pd.DataFrame:
  """Winner refitted separately in each group (e.g. UN region) [R1-M3].

  Returns long table (group, term, coef, std_err). Terms that cannot be
  estimated within a group (e.g. latitude in a narrow region) are still
  reported but will have wide CIs. REVIEW: min_users.
  """
  user_level = modal_user_value(df, group_col)
  out = []
  for level, n in user_level.value_counts().items():
    if n < min_users or str(level) in ('unknown', 'nan'):
      continue
    sub = df[df['user_id_n'].isin(user_level.index[user_level == level])]
    sub = sub.copy()
    sl.residualise_temperature(sub, 'harmonic')
    try:
      t = coef_table(fit_lme(formula, sub)).reset_index()
    except Exception as e:  # pylint: disable=broad-except
      print(f'[by_group] {level}: {e!r}')
      continue
    t.insert(0, group_col, level)
    t['n_users'] = n
    out.append(t)
  return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ===========================================================================
# 6. Monthly-threshold robustness and adherence [R1-m5]
# ===========================================================================
VALID_DAYS_COL = 'n_valid_days'


def attach_valid_days(df: pd.DataFrame, path: Optional[str]) -> bool:
  """Merges a per-user-month valid-day count if available (in place).

  The public schema has no valid-day count. REVIEW / DATA NEEDED: export a
  CSV with columns ``id, month, n_valid_days`` (same ids and YYYY-MM months as
  the cohort file) from the internal pipeline and pass its path here.
  Returns True when the column is available.
  """
  if VALID_DAYS_COL in df:
    return True
  if not path or not fio.file_exists(path):
    return False
  v = read_csv(path).rename(columns={'id': 'user_id', 'month':
                                     'activity_period'})
  v['activity_period'] = pd.to_datetime(v['activity_period'])
  v['user_id'] = v['user_id'].astype(df['user_id'].dtype)
  merged = df[['user_id', 'activity_period']].merge(
      v, on=['user_id', 'activity_period'], how='left')
  df[VALID_DAYS_COL] = merged[VALID_DAYS_COL].to_numpy()
  return True


def adherence_by_month(df: pd.DataFrame) -> pd.DataFrame:
  """Adherence per calendar month and hemisphere.

  Always: number of users contributing a month and that number as a share of
  all users of the hemisphere (proxy for seasonal drop-out). With a valid-day
  column also: mean / median valid days and share of user-months with >= 7,
  14, 20 valid days.
  """
  g = df.groupby(['hemisphere', 'month'])
  out = g['user_id_n'].nunique().rename('active_users').to_frame()
  total = df.groupby('hemisphere')['user_id_n'].nunique()
  out['share_of_users_%'] = 100 * out['active_users'] / out.index.get_level_values(
      'hemisphere').map(total).to_numpy()
  if VALID_DAYS_COL in df:
    out['mean_valid_days'] = g[VALID_DAYS_COL].mean()
    out['median_valid_days'] = g[VALID_DAYS_COL].median()
    for t in (7, 14, 20):
      out[f'share_>={t}d_%'] = 100 * g[VALID_DAYS_COL].apply(
          lambda s, t=t: (s >= t).mean())
  return out.reset_index()


def run_threshold_sensitivity(df: pd.DataFrame, formula: str, metric: str,
                              out_dir: str, thresholds=(7, 14, 20),
                              force: bool = False) -> pd.DataFrame:
  """Refits the winner keeping user-months with >= t valid days.

  Temperature is re-residualised on each filtered sample; users left with a
  single month are dropped (as in 02_modelling). Returns a long table
  (threshold, term, coef, std_err) plus fit statistics per threshold.
  """
  path = os.path.join(out_dir, 'threshold_sensitivity.csv')
  if fio.file_exists(path) and not force:
    return read_csv(path)
  if VALID_DAYS_COL not in df:
    raise KeyError(f'{VALID_DAYS_COL} missing: see attach_valid_days')
  target = TARGET_FIT[metric]
  out = []
  for t in (None,) + tuple(thresholds):
    sub = df if t is None else df[df[VALID_DAYS_COL] >= t]
    cnt = sub['user_id_n'].value_counts()
    sub = sub[sub['user_id_n'].isin(cnt[cnt > 1].index)].copy()
    sl.residualise_temperature(sub, 'harmonic')
    res = fit_lme(formula, sub)
    tab = coef_table(res).reset_index()
    for k, v in fit_summary(res, sub, target).items():
      tab[k] = v
    tab.insert(0, 'min_valid_days', 'baseline' if t is None else t)
    out.append(tab)
  table = pd.concat(out, ignore_index=True)
  write_csv(table, path)
  return table


# ===========================================================================
# 7. HRV lower bound [R1-M5] and daily-data helpers [R1-M5, R1-m5, R2-3]
# ===========================================================================
def hrv_low_tail(df: pd.DataFrame, cutoffs=(20, 25, 30)) -> pd.DataFrame:
  """Monthly HRV near the 20 ms bound, by sex x age band.

  With monthly data we can only show how much of each group sits close to the
  bound (monthly means of daily values that were already >= 20 ms). The share
  of DAILY values removed at 20 ms needs the daily data
  (``daily_bound_exclusions``). REVIEW.
  """
  col = TARGET_RAW['hrv']
  rows = []
  for (sex, age), g in df.groupby(['gender', 'age_band'], observed=True):
    um = g.groupby('user_id')[col].median()
    row = dict(sex=sex, age_band=age, users=len(um), user_months=len(g),
               median_ms=float(g[col].median()),
               p05_ms=float(g[col].quantile(0.05)))
    for c in cutoffs:
      row[f'user_months_<{c}ms_%'] = 100 * float((g[col] < c).mean())
      row[f'users_median_<{c}ms_%'] = 100 * float((um < c).mean())
    rows.append(row)
  return pd.DataFrame(rows)


def plot_hrv_distribution_by_age(df: pd.DataFrame, bins=np.arange(10, 151, 2)
                                 ) -> plt.Figure:
  """Histogram of monthly HRV by age band, per sex, with the 20 ms bound."""
  col = TARGET_RAW['hrv']
  fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
  for ax, sex in zip(axes, ['Female', 'Male']):
    g = df[df['gender'] == sex]
    for age in AGE_LABELS:
      v = g.loc[g['age_band'] == age, col]
      if len(v):
        ax.hist(v, bins=bins, density=True, histtype='step', label=age)
    ax.axvline(20, color='k', ls='--', lw=0.8)
    ax.set_title(sex, fontsize=9)
    ax.set_xlabel('Monthly HRV (ms)')
  axes[0].set_ylabel('Density')
  axes[1].legend(title='Age', fontsize=7, frameon=False)
  fig.tight_layout()
  return fig


# --- Daily data (NOT in the public schema) ---------------------------------
# REVIEW / DATA NEEDED: these functions expect a daily extract with columns
# ``user_id, date, value`` (value = daily RHR in bpm or daily HRV rMSSD in ms,
# BEFORE any range filtering) for the same users and year. They cover the
# bound sensitivity (R1-M5), the valid-day thresholds (R1-m5) and the
# weekday/weekend analysis (R2-3).
def daily_bound_exclusions(daily: pd.DataFrame, lo: float = 20.0,
                           hi: float = 200.0,
                           user_info: Optional[pd.DataFrame] = None
                           ) -> pd.DataFrame:
  """% of daily values below ``lo`` / above ``hi`` overall and by sex x age
  band (``user_info``: user_id, gender, age)."""
  d = daily.copy()
  d['below'] = d['value'] < lo
  d['above'] = d['value'] > hi
  if user_info is not None:
    d = d.merge(user_info, on='user_id', how='left')
    d['age_band'] = pd.cut(d['age'], AGE_BINS, labels=AGE_LABELS,
                           right=False).astype(str)
    keys = ['gender', 'age_band']
  else:
    keys = []
  overall = pd.DataFrame([dict(group='all', days=len(d),
                               below_lo_pct=100 * d['below'].mean(),
                               above_hi_pct=100 * d['above'].mean())])
  if not keys:
    return overall
  g = d.groupby(keys).agg(days=('value', 'size'),
                          below_lo_pct=('below', 'mean'),
                          above_hi_pct=('above', 'mean')).reset_index()
  g[['below_lo_pct', 'above_hi_pct']] *= 100
  return pd.concat([overall, g], ignore_index=True)


def monthly_from_daily(daily: pd.DataFrame, lo: Optional[float] = 20.0,
                       hi: Optional[float] = 200.0, min_days: int = 1,
                       days: str = 'all') -> pd.DataFrame:
  """Rebuilds user-month means from daily values.

  Args:
    daily: user_id, date, value.
    lo, hi: Range filter (None = no bound) [R1-M5 sensitivity].
    min_days: Minimum valid days per user-month [R1-m5].
    days: 'all' | 'weekday' | 'weekend' [R2-3].

  Returns:
    user_id, activity_period, value (mean), n_valid_days. Merge onto the
    cohort (replacing the target) and refit with the usual pipeline.
    REVIEW: check that the internal monthly values are plain means of the
    filtered daily values (as described in Methods).
  """
  d = daily.copy()
  d['date'] = pd.to_datetime(d['date'])
  if lo is not None:
    d = d[d['value'] >= lo]
  if hi is not None:
    d = d[d['value'] <= hi]
  wd = d['date'].dt.dayofweek
  if days == 'weekday':
    d = d[wd < 5]
  elif days == 'weekend':
    d = d[wd >= 5]
  d['activity_period'] = d['date'].dt.to_period('M').dt.to_timestamp()
  m = d.groupby(['user_id', 'activity_period'])['value'].agg(
      value='mean', n_valid_days='size').reset_index()
  return m[m['n_valid_days'] >= min_days]


def weekday_weekend_difference(daily: pd.DataFrame,
                               user_country: pd.Series) -> pd.DataFrame:
  """Weekend minus weekday mean per user-month, summarised by country and
  month [R2-3]. ``user_country``: user_id -> country.
  REVIEW: weekend = Saturday/Sunday everywhere; some countries (e.g. parts of
  the Middle East) have Friday/Saturday weekends.
  """
  d = daily.copy()
  d['date'] = pd.to_datetime(d['date'])
  d['weekend'] = d['date'].dt.dayofweek >= 5
  d['month'] = d['date'].dt.month
  um = d.groupby(['user_id', 'month', 'weekend'])['value'].mean().unstack()
  um = um.dropna()
  um['diff'] = um[True] - um[False]
  um = um.reset_index()
  um['country'] = um['user_id'].map(user_country)
  return (um.groupby(['country', 'month'])['diff']
          .agg(['mean', 'std', 'count']).reset_index())


# ===========================================================================
# 8. Extra model combinations [R3-M2]
# ===========================================================================
def extra_model_candidates(formula: str) -> Dict[str, str]:
  """Combinations the stage-wise selection never tried.

  * age x temperature: age_c interacting with every residualised
    temperature term;
  * sex x season and age x season;
  * sex x temperature;
  * latitude x temperature.
  REVIEW: add / remove candidates; a reversed stage order (temperature before
  latitude) can be run with stage_competition.run_competition and a custom
  StageSpec if the authors want it.
  """
  res = residual_columns(formula)
  temp = ' + '.join(c for c in res if '_sq_' not in c) or 'avg_temp_mean_c_res'
  return {
      'winner': formula,
      'winner + age x temperature': f'{formula} + age_c:({temp})',
      'winner + sex x temperature': f'{formula} + gender_n:({temp})',
      'winner + sex x season': f'{formula} + gender_n:({MONTH_HARMONICS})',
      'winner + age x season': f'{formula} + age_c:({MONTH_HARMONICS})',
      'winner + latitude x temperature': f'{formula} + lat_e7_c:({temp})',
  }


def run_extra_models(df: pd.DataFrame, formula: str, metric: str,
                     out_dir: str, force: bool = False) -> pd.DataFrame:
  """ML fits of ``extra_model_candidates``; BIC relative to the winner.
  Resumable per candidate."""
  path = os.path.join(out_dir, 'extra_models.csv')
  done = read_csv(path) if fio.file_exists(path) and not force else None
  target = TARGET_FIT[metric]
  rows = [] if done is None else done.to_dict('records')
  have = {r['candidate'] for r in rows}
  for name, f in extra_model_candidates(formula).items():
    if name in have:
      continue
    print(f'[extra] {name}')
    res = fit_lme(f, df)
    rows.append(dict(candidate=name, formula=f, **fit_summary(res, df,
                                                              target)))
    write_csv(pd.DataFrame(rows), path)
  t = pd.DataFrame(rows)
  t['delta_BIC_vs_winner'] = t['bic'] - t.loc[t['candidate'] == 'winner',
                                              'bic'].iloc[0]
  write_csv(t, path)
  return t


# ===========================================================================
# 9. Post-hoc statistics
# ===========================================================================
def user_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
  """One row per user: mean raw metric, modal sex / band / country."""
  col = TARGET_RAW[metric]
  u = df.groupby('user_id_n').agg(value=(col, 'mean'), age=('age', 'mean'))
  for c in ('gender', 'lat_band', 'hemisphere', 'age_band', 'un_region',
            'country_2_alpha_cd'):
    u[c] = modal_user_value(df, c)
  if metric == 'hrv':
    u['log_value'] = df.groupby('user_id_n')['target_log'].mean()
  return u


def two_group_test(a: pd.Series, b: pd.Series, label_a: str, label_b: str
                   ) -> Dict[str, Any]:
  """Welch t-test and Mann-Whitney U on user-level means, with the
  difference, its 95% CI and Cohen's d [R3-M5]."""
  a, b = a.dropna(), b.dropna()
  t = stats.ttest_ind(a, b, equal_var=False)
  u = stats.mannwhitneyu(a, b, alternative='two-sided')
  diff = a.mean() - b.mean()
  se = np.sqrt(a.var() / len(a) + b.var() / len(b))
  sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
               / (len(a) + len(b) - 2))
  return dict(group_a=label_a, group_b=label_b, n_a=len(a), n_b=len(b),
              mean_a=a.mean(), mean_b=b.mean(), diff=diff,
              diff_ci_lower=diff - 1.96 * se, diff_ci_upper=diff + 1.96 * se,
              cohens_d=diff / sp, welch_t=t.statistic, welch_p=t.pvalue,
              mannwhitney_p=u.pvalue)


def overall_test(values: pd.Series, groups: pd.Series, label: str,
                 order: Optional[Sequence[str]] = None) -> Dict[str, Any]:
  """Kruskal-Wallis test across all groups, with epsilon-squared effect size
  (share of rank variance explained by the grouping)."""
  levels = [g for g in (order or sorted(groups.dropna().unique()))
            if (groups == g).sum() > 1]
  samples = [values[groups == g].dropna() for g in levels]
  h = stats.kruskal(*samples)
  n, k = sum(len(x) for x in samples), len(samples)
  return dict(test='Kruskal-Wallis (all groups)', group_a=label,
              group_b=', '.join(map(str, levels)), n_a=n, n_b=np.nan,
              kruskal_H=h.statistic, kruskal_p=h.pvalue,
              epsilon_sq=(h.statistic - k + 1) / (n - k))


def group_comparisons(df: pd.DataFrame, metric: str) -> pd.DataFrame:
  """Tests for the comparisons named by R3-M5 (user-level means).

  * sex: Male vs Female;
  * hemisphere: Northern vs Southern;
  * age: Kruskal-Wallis across all age bands, then each band vs the next
    one up (adjacent pairs), matching how the text describes age trends;
  * latitude: Kruskal-Wallis across all bands; tropics (N. + S. Tropics) vs
    Far North/South Polar (the reviewer's 'tropical vs polar'); tropics vs
    each other band (the 'tropical RHR is highest' claim); Mid North vs
    Far North (the HRV range reported in Results 2.1.2).
  Pairwise rows give the difference, its 95% CI, Cohen's d, Welch's t-test
  and Mann-Whitney U. With N in the millions all p-values will be tiny; the
  reply should point to effect sizes and CIs.
  REVIEW: the tropics-vs-band and Mid North vs Far North rows are computed for
  both metrics; keep the ones that match a claim in the text.
  """
  u = user_means(df, metric)
  val = u['value']
  rows = []

  def pair(mask_a, mask_b, la, lb, test):
    r = two_group_test(val[mask_a], val[mask_b], la, lb)
    r['test'] = test
    rows.append(r)

  pair(u['gender'] == 'Male', u['gender'] == 'Female', 'Male', 'Female',
       'sex')
  pair(u['hemisphere'] == 'Northern', u['hemisphere'] == 'Southern',
       'Northern', 'Southern', 'hemisphere')
  # Age: overall + adjacent bands.
  rows.append(dict(overall_test(val, u['age_band'], 'age bands', AGE_LABELS),
                   test='age: Kruskal-Wallis'))
  present = [a for a in AGE_LABELS if (u['age_band'] == a).sum() > 1]
  for a0, a1 in zip(present[:-1], present[1:]):
    pair(u['age_band'] == a1, u['age_band'] == a0, a1, a0,
         'age: adjacent bands')
  # Latitude: overall + named contrasts.
  bands = [b for b in LAT_BAND_LABELS if (u['lat_band'] == b).sum() > 1]
  rows.append(dict(overall_test(val, u['lat_band'], 'latitude bands', bands),
                   test='latitude: Kruskal-Wallis'))
  trop = u['lat_band'].isin(['N. Tropics', 'S. Tropics'])
  polar = u['lat_band'].isin(['Far North', 'South Polar'])
  pair(trop, polar, 'Tropics', 'Far North/Polar', 'latitude: tropics vs polar')
  for b in bands:
    if b in ('N. Tropics', 'S. Tropics'):
      continue
    pair(trop, u['lat_band'] == b, 'Tropics', b,
         'latitude: tropics vs other band')
  pair(u['lat_band'] == 'Mid North', u['lat_band'] == 'Far North',
       'Mid North', 'Far North', 'latitude: Mid North vs Far North')
  out = pd.DataFrame(rows)
  out.insert(0, 'metric', metric)
  first = ['metric', 'test', 'group_a', 'group_b']
  return out[first + [c for c in out.columns if c not in first]]


def rhr_hrv_correlation(df_rhr: pd.DataFrame, df_hrv: pd.DataFrame
                        ) -> Dict[str, Any]:
  """Within- and between-person RHR-HRV correlation [R1-m15].

  Users present in both cohorts, matched on user and month. Between-person:
  correlation of user means (Pearson on ln HRV and Spearman). Within-person:
  repeated-measures correlation (Bakdash & Marusich 2017): Pearson
  correlation of person-centred values with df = N - users - 1.
  REVIEW: the two cohorts must share user ids (true if both CSVs come from
  the same export); thinning removed some RHR users.
  """
  r = df_rhr[['user_id', 'activity_period', TARGET_RAW['rhr']]]
  h = df_hrv[['user_id', 'activity_period', TARGET_RAW['hrv']]]
  m = r.merge(h, on=['user_id', 'activity_period'])
  m = m[m.groupby('user_id')['user_id'].transform('size') > 1].copy()
  if m.empty:
    return dict(n_users=0)
  m['ln_hrv'] = np.log(m[TARGET_RAW['hrv']])
  x, y = TARGET_RAW['rhr'], 'ln_hrv'
  um = m.groupby('user_id')[[x, y]].mean()
  rb = stats.pearsonr(um[x], um[y])
  sb = stats.spearmanr(um[x], um[y])
  xc = m[x] - m.groupby('user_id')[x].transform('mean')
  yc = m[y] - m.groupby('user_id')[y].transform('mean')
  rw = float(np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
  n, k = len(m), um.shape[0]
  dfree = n - k - 1
  tw = rw * np.sqrt(dfree / (1 - rw ** 2))
  z = np.arctanh(rw)
  se = 1 / np.sqrt(dfree - 1)
  zb = np.arctanh(rb.statistic)
  seb = 1 / np.sqrt(k - 3)
  return dict(
      n_users=int(k), n_user_months=int(n),
      between_pearson_r=float(rb.statistic),
      between_ci=(float(np.tanh(zb - 1.96 * seb)),
                  float(np.tanh(zb + 1.96 * seb))),
      between_spearman_rho=float(sb.statistic),
      within_rmcorr_r=rw,
      within_ci=(float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))),
      within_p=float(2 * stats.t.sf(abs(tw), dfree)))


def effect_sizes(fe: pd.Series, run_stats: Mapping[str, Any], metric: str,
                 ref_lat_deg: float = 45.0) -> pd.DataFrame:
  """Effect sizes in intuitive units [R1-m9], from the fixed effects.

  Contrasts are differences of the fixed-effect prediction between two
  synthetic profiles that differ in one variable, averaged over the 12
  months (so season interactions average out where they should). Temperature
  residuals are 0 (= the usual temperature for that latitude and month)
  except in the temperature contrast, which adds +5 degC to the monthly
  mean, max and min. HRV is reported as % change.
  REVIEW: reference profile (age 40, sex-balanced, 45 deg N, longitude 0,
  day length from latitude and month), and the contrasts listed.
  """
  summ = run_stats['standardisation']
  male_share = run_stats['male_share']
  factors = run_stats['scaling_factors']

  def profile(age=40.0, male=None, lat=ref_lat_deg, dtemp=0.0):
    months = np.arange(1, 13)
    p = pd.DataFrame({
        'activity_period': pd.to_datetime(
            [f'2022-{m:02d}-01' for m in months]),
        'age': age, 'lat_e7': lat * 1e7, 'lng_e7': 0.0,
    })
    p['dl_monthly'] = sl.mid_month_day_length_hours(lat, months)
    for c in ('avg_temp_mean', 'avg_temp_max', 'avg_temp_min'):
      p[c] = summ.get(f'{c}_mean', 0.0)
    sl.add_month_harmonics(p)
    p['gender_n'] = (0.0 if male is None else float(male)) - (
        0.0 if male is None else male_share)
    p['abs_lat_e7'] = p['lat_e7'].abs()
    lng = np.deg2rad(p['lng_e7'] / 1e7)
    p['sin_lng'], p['cos_lng'] = np.sin(lng), np.cos(lng)
    sl.standardise_model_features(p, summary=summ)
    for c in ('avg_temp_mean', 'avg_temp_max', 'avg_temp_min'):
      key = f'{c}_c'
      shift = dtemp / factors[key] if key in factors else 0.0
      p[f'{c}_c_res'] = shift
      p[f'{c}_sq_c_res'] = shift ** 2
    return p

  def pred(p):
    return float(sl.predict_from_coefficients(p, fe).mean())

  def fmt(d):
    return (np.exp(d) - 1) * 100 if metric == 'hrv' else d

  unit = '%' if metric == 'hrv' else 'bpm'
  rows = []
  for a0, a1 in [(30, 40), (40, 50), (50, 60), (60, 70)]:
    for sex, male in [('Female', 0), ('Male', 1)]:
      rows.append((f'Age {a0} -> {a1} ({sex})',
                   fmt(pred(profile(a1, male)) - pred(profile(a0, male)))))
  rows.append(('Male vs Female (age 40)',
               fmt(pred(profile(male=1)) - pred(profile(male=0)))))
  rows.append(('+5 degC residual temperature (annual average)',
               fmt(pred(profile(dtemp=5)) - pred(profile()))))
  for lat in (ref_lat_deg, -35.0):
    p = profile(lat=lat)
    s = sl.predict_from_coefficients(p, fe)
    rows.append((f'Seasonal peak-to-trough at {lat:+.0f} deg',
                 fmt(float(s.max() - s.min()))))
    rows.append((f'Peak month at {lat:+.0f} deg', float(p['month'][s.idxmax()])))
  return pd.DataFrame(rows, columns=['contrast', f'effect ({unit})'])


def country_residual_correlation(df: pd.DataFrame, result, metric: str,
                                 external_csv: str,
                                 min_users: int = 200) -> pd.DataFrame:
  """[R1-m3, optional] Country mean marginal residual vs external indicators.

  ``external_csv``: columns ``country`` (ISO alpha-2) and one column per
  indicator (e.g. ``obesity_pct`` from WHO GHO, ``pm25_ugm3`` from WHO /
  World Bank). Returns Pearson and Spearman r with 95% CI and the number of
  countries. REVIEW / DATA NEEDED: download the indicators (same year, 2022
  or nearest).
  """
  target = TARGET_FIT[metric]
  r = df[target].to_numpy() - fixed_prediction(df, result.fe_params)
  u = pd.DataFrame({'user_id_n': df['user_id_n'], 'r': r}).groupby(
      'user_id_n')['r'].mean().to_frame()
  u['country'] = modal_user_value(df, 'country_2_alpha_cd')
  c = u.groupby('country')['r'].agg(['mean', 'size'])
  c = c[c['size'] >= min_users]
  ext = read_csv(external_csv).set_index('country')
  j = c.join(ext, how='inner')
  rows = []
  for ind in ext.columns:
    v = j[['mean', ind]].dropna()
    if len(v) < 5:
      continue
    pr = stats.pearsonr(v['mean'], v[ind])
    z, se = np.arctanh(pr.statistic), 1 / np.sqrt(len(v) - 3)
    rows.append(dict(indicator=ind, n_countries=len(v),
                     pearson_r=pr.statistic, pearson_p=pr.pvalue,
                     ci_lower=np.tanh(z - 1.96 * se),
                     ci_upper=np.tanh(z + 1.96 * se),
                     spearman_rho=stats.spearmanr(v['mean'], v[ind]).statistic))
  return pd.DataFrame(rows)


# ===========================================================================
# 10. Figures
# ===========================================================================
MONTH_TICKS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


def plot_monthly_trajectories(
    pred: pd.DataFrame,
    metric: str,
    group_col: Optional[str] = None,
    levels: Optional[Sequence[Any]] = None,
    log_scale: bool = False,
    show_conditional: bool = False,
    ncols: int = 3,
    min_rows: int = 30,
    title: Optional[str] = None,
    sharey: bool = False,
) -> plt.Figure:
  """Observed vs predicted monthly means (95% CI of the observed mean).

  Used for Figs 2-4 on held-out users [R1-M2, R3-M1], Fig 2 by hemisphere
  [R3-M6] and the HRV fits on the ln scale [R3-M7]. ``pred`` is the output
  of ``evaluate_fold`` (columns y, pred_marginal[, y_ms, pred_marginal_ms]).
  """
  if metric == 'hrv' and not log_scale:
    obs, hat, hatc, ylab = ('y_ms', 'pred_marginal_ms',
                            'pred_conditional_ms', 'HRV (ms)')
  elif metric == 'hrv':
    obs, hat, hatc, ylab = 'y', 'pred_marginal', 'pred_conditional', 'ln HRV'
  else:
    obs, hat, hatc, ylab = 'y', 'pred_marginal', 'pred_conditional', 'RHR (bpm)'
  if group_col is None:
    groups = [('All', pred)]
  else:
    levels = levels or list(pred[group_col].value_counts().index)
    groups = [(lv, pred[pred[group_col] == lv]) for lv in levels]
  n = len(groups)
  ncols = min(ncols, n)
  nrows = int(np.ceil(n / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows),
                           squeeze=False, sharey=sharey)
  for ax, (lv, g) in zip(axes.flat, groups):
    cell = g.groupby('month').agg(o=(obs, 'mean'), s=(obs, 'std'),
                                  n=(obs, 'size'), p=(hat, 'mean'),
                                  pc=(hatc, 'mean')).reindex(range(1, 13))
    cell = cell[cell['n'] >= min_rows]
    ci = 1.96 * cell['s'] / np.sqrt(cell['n'])
    ax.fill_between(cell.index, cell['o'] - ci, cell['o'] + ci, alpha=0.25,
                    color='C0', lw=0)
    ax.plot(cell.index, cell['o'], 'o-', ms=2.5, color='C0', label='Observed')
    ax.plot(cell.index, cell['p'], '-', color='C3', label='Predicted (fixed)')
    if show_conditional:
      ax.plot(cell.index, cell['pc'], '--', color='C2',
              label='Predicted (+ intercept)')
    users = g['user_id_n'].nunique()
    ax.set_title(f'{lv}  (n={users:,})', fontsize=8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_TICKS, fontsize=7)
    ax.tick_params(labelsize=7)
  for ax in list(axes.flat)[n:]:
    ax.axis('off')
  for ax in axes[:, 0]:
    ax.set_ylabel(ylab, fontsize=8)
  axes.flat[0].legend(fontsize=6, frameon=False)
  if title:
    fig.suptitle(title, fontsize=9)
  fig.tight_layout()
  return fig


def figure_1_shared_axes(df_rhr: pd.DataFrame, df_hrv: pd.DataFrame,
                         output_dir: Optional[str] = None,
                         file_prefix: str = 'figure_1_shared_axes'):
  """[R3-m1] Fig 1 with the same axes in each pair of panels.

  Calls publication_figures.generate_figure_1 unchanged, then sets the union
  of the y-limits for A/B (RHR women/men) and C/D (HRV women/men) and the
  union of x-limits across A-D, and re-saves. REVIEW: union limits, i.e.
  RHR 57-75 bpm and HRV 24-72 ms with the current hard-coded limits; the
  cleaner fix is to change ``y_lim`` in generate_figure_1 itself (lines with
  y_lim=(61, 75) / (57, 73) and (24, 66) / (24, 72)).
  """
  import publication_figures as pf  # pylint: disable=g-import-not-at-top
  fig, _, _ = pf.generate_figure_1(df_rhr, df_hrv, output_dir=None)
  ax = fig.axes
  for i, j in ((0, 1), (2, 3), (4, 5)):
    lo = min(ax[i].get_ylim()[0], ax[j].get_ylim()[0])
    hi = max(ax[i].get_ylim()[1], ax[j].get_ylim()[1])
    ax[i].set_ylim(lo, hi)
    ax[j].set_ylim(lo, hi)
  xlo = min(a.get_xlim()[0] for a in ax[:4])
  xhi = max(a.get_xlim()[1] for a in ax[:4])
  for a in ax[:4]:
    a.set_xlim(xlo, xhi)
  paths = []
  if output_dir:
    fio.make_dirs(output_dir)
    for ext in ('png', 'pdf', 'svg'):
      p = os.path.join(output_dir, f'{file_prefix}.{ext}')
      with fio.open_file(p, 'wb') as f:
        fig.savefig(f, format=ext, dpi=300)
      paths.append(p)
  return fig, paths


def save_fig(fig: plt.Figure, out_dir: str, name: str) -> str:
  fio.make_dirs(out_dir)
  path = os.path.join(out_dir, f'{name}.png')
  with fio.open_file(path, 'wb') as f:
    fig.savefig(f, format='png', dpi=200, bbox_inches='tight')
  return path
