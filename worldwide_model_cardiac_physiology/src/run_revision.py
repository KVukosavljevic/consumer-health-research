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
  p.add_argument('--formula', default=None, help='override the winner')
  p.add_argument('--subsample_frac', type=float, default=1.0)
  p.add_argument('--random_state', type=int, default=42)
  p.add_argument('--holdout_countries', default='GB,IN,CA,BR,ZA,AU')
  p.add_argument('--min_users_per_fold', type=int, default=200)
  p.add_argument('--sparse_country_max_users', type=int, default=1000)
  p.add_argument('--country_cap', type=int, default=300)
  p.add_argument('--valid_days_csv', default=None)
  p.add_argument('--force', action='store_true')
  a = p.parse_args()

  data_dir = fio.get_data_dir()
  t0 = time.time()
  df, _ = rl.load_prepared(a.metric, data_dir, a.subsample_frac,
                           random_state=a.random_state)
  if a.valid_days_csv:
    rl.attach_valid_days(df, a.valid_days_csv)
  formula = a.formula or rl.winner_formula(a.metric, data_dir, a.run_tag)
  print(f'{len(df):,} rows / {df["user_id_n"].nunique():,} users '
        f'({time.time() - t0:.0f}s)\nformula: {formula}', flush=True)
  out = lambda *parts: rl.revision_dir(data_dir, a.metric, *parts)
  sections = {s.strip() for s in a.sections.split(',') if s.strip()}

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
                              cap=a.country_cap, force=a.force)
  if 'A5' in sections and rl.VALID_DAYS_COL in df:
    rl.run_threshold_sensitivity(df, formula, a.metric, out('A5_thresholds'),
                                 force=a.force)
  if 'A6' in sections:
    rl.run_extra_models(df, formula, a.metric, out('A6_extra_models'),
                        force=a.force)
  print(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
  main()
