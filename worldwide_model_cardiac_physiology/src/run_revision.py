# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs the heavy refits of 04_revision_analyses.ipynb from a terminal.

Same functions and output folders as the notebook, so results written here
are picked up by the notebook (and vice versa). Every section is resumable:
interrupt and rerun the same command to continue.

B3 (effect sizes, R1-m9) needs no cohort data at all, only two files of the
notebook-02 run: ``model_<metric>_coefficients_interpreted.csv`` and
``feature_stats.json``. ``--sections B3`` runs it in seconds without loading
the cohort; ``--run_dir`` can point at any folder holding those two files.

Example (from the repo folder, in tmux/screen or with nohup):

  python src/run_revision.py --metric rhr --sections A1,A2,A3,A6
  python src/run_revision.py --metric hrv --sections A1 \
      --holdout_countries GB,IN,CA,BR,ZA,AU
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import file_io as fio  # pylint: disable=g-import-not-at-top
import revision_lib as rl  # pylint: disable=g-import-not-at-top


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  p.add_argument('--metric', required=True, choices=('rhr', 'hrv'))
  p.add_argument('--sections', default='A1,A2,A3,A6',
                 help='comma list among A1 (OOS), A2 (within/between), '
                      'A3 (country), A5 (thresholds), A6 (extra models)')
  p.add_argument('--run_tag', default=None)
  p.add_argument('--run_dir', default=None,
                 help='folder with the run\'s coefficients CSV and '
                      'feature_stats.json (B3 only; default results/<run_tag>)')
  p.add_argument('--formula', default=None, help='override the winner')
  p.add_argument('--subsample_frac', type=float, default=1.0)
  p.add_argument('--random_state', type=int, default=42)
  p.add_argument('--holdout_countries', default='GB,IN,CA,BR,ZA,AU')
  p.add_argument('--min_users_per_fold', type=int, default=200)
  p.add_argument('--sparse_country_max_users', type=int, default=1000)
  p.add_argument('--valid_days_csv', default=None)
  p.add_argument('--force', action='store_true')
  a = p.parse_args()

  data_dir = fio.get_data_dir()
  t0 = time.time()
  sections = {s.strip() for s in a.sections.split(',') if s.strip()}
  if 'B3' in sections:
    run_effect_sizes(a, data_dir)
    sections.discard('B3')
    if not sections:
      return
  df, _ = rl.load_prepared(a.metric, data_dir, a.subsample_frac,
                           random_state=a.random_state)
  if a.valid_days_csv:
    rl.attach_valid_days(df, a.valid_days_csv)
  formula = a.formula or rl.winner_formula(a.metric, data_dir, a.run_tag)
  print(f'{len(df):,} rows / {df["user_id_n"].nunique():,} users '
        f'({time.time() - t0:.0f}s)\nformula: {formula}', flush=True)
  out = lambda *parts: rl.revision_dir(data_dir, a.metric, *parts)

  if 'A1' in sections:
    countries = [c for c in a.holdout_countries.split(',') if c]
    folds = rl.default_folds(df, countries, a.min_users_per_fold,
                             a.sparse_country_max_users)
    rl.run_oos_validation(df, formula, a.metric, out('A1_oos'), folds,
                          random_state=a.random_state, force=a.force)
  if 'A2' in sections:
    rl.run_within_between(df, formula, a.metric, out('A2_within_between'),
                          force=a.force)
  if 'A3' in sections:
    rl.run_country_robustness(df, formula, a.metric, out('A3_country'),
                              force=a.force)
  if 'A5' in sections and rl.VALID_DAYS_COL in df:
    rl.run_threshold_sensitivity(df, formula, a.metric, out('A5_thresholds'),
                                 force=a.force)
  if 'A6' in sections:
    rl.run_extra_models(df, formula, a.metric, out('A6_extra_models'),
                        force=a.force)
  print(f'done in {(time.time() - t0) / 60:.1f} min')


def run_effect_sizes(a, data_dir: str) -> None:
  """B3 from the saved coefficients and feature statistics only."""
  if a.run_dir:
    import pandas as pd  # pylint: disable=g-import-not-at-top
    import seasonality_lib as sl  # pylint: disable=g-import-not-at-top
    fe = rl.read_csv(sl.coefficients_csv_path(a.run_dir, a.metric),
                     index_col=0)['coef']
    fe = fe[~fe.index.astype(str).str.endswith(' Var')]
    st = sl.read_feature_stats(a.run_dir)
    if st is not None:
      st['scaling_factors'] = sl.scaling_factors_from_summary(
          st['standardisation'])
  else:
    fe = rl.winner_coefficients(a.metric, data_dir, a.run_tag)
    st = rl.run_stats_from_run(a.metric, data_dir, a.run_tag)
  if fe is None or st is None:
    raise FileNotFoundError(
        'B3 needs model_<metric>_coefficients_interpreted.csv and '
        'feature_stats.json from the notebook-02 run (use --run_dir).')
  d = rl.revision_dir(data_dir, a.metric, 'B3_effect_sizes')
  es = rl.effect_sizes(fe, st, a.metric)
  rl.write_csv(es, os.path.join(d, 'effect_sizes.csv'))
  grid = rl.effect_size_grid(fe, st, a.metric)
  rl.write_csv(grid, os.path.join(d, 'effect_size_grid.csv'))
  print(rl.save_fig(rl.plot_effect_grid(grid, a.metric), d,
                    'effect_size_grid'))
  print(es.round(2).to_string())


if __name__ == '__main__':
  main()
