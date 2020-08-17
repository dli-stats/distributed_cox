"""Experiment result collection."""

import glob
import os
import json
import argparse
import pickle
import collections

import numpy as onp
import pandas as pd
import tqdm
from varderiv.experiments.run import compute_results_averaged
# pylint: disable=missing-docstring, unused-import


def iterate_experiments(runs_dir):
  for run_dir in glob.iglob(os.path.join(runs_dir, "*")):
    if run_dir.endswith("_sources"):
      continue
    try:
      with open(os.path.join(run_dir, "config.json"), "r") as config_file:
        config_json = json.load(config_file)
      with open(os.path.join(run_dir, "run.json"), "r") as run_file:
        run_json = json.load(run_file)
      yield run_dir, config_json, run_json
    except IOError:
      pass


def recursive_dict_subset(d1, d2):
  """Recursively compare two dicts.

  Assuming d1 and d2 has the same structure, with d2 having less keys than d1.
  This function compares if all key value pairs in d2 are exactly the same
  in d1.
  """
  if isinstance(d1, dict) and isinstance(d2, dict):
    for k in d2:
      if k not in d1:
        return False
      if not recursive_dict_subset(d1[k], d2[k]):
        return False
    return True
  return d1 == d2


def find_experiment(runs_dir, **kwargs):
  for run_dir, config_json, run_json in iterate_experiments(runs_dir):
    if recursive_dict_subset(config_json, kwargs):
      yield (run_dir, config_json, run_json)


expkey_names = ("eq K N T_star_factors X_DIM "
                "group_X group_labels_generator_kind").split()


def get_eq_name(experiment):
  _, config_json, _ = experiment
  eq = config_json['eq']
  if eq == "meta_analysis":
    if config_json.get("univariate", False):
      eq = "meta_analysis_univariate"
  return eq


def get_expkey(experiment):
  _, config_json, _ = experiment
  expkey = tuple(
      config_json["base"][n] if n != "eq" else get_eq_name(experiment)
      for n in expkey_names)
  return expkey


def merge_experiments_same_setting(runs_dir, **kwargs):
  experiments = list(iterate_experiments(runs_dir))
  same_experiments = collections.defaultdict(list)
  for exp in experiments:
    expkey = get_expkey(exp)
    same_experiments[expkey].append(exp)
  return same_experiments


def main(args):
  runs = [
      experiment for experiment in iterate_experiments(args.runs_dir)
      if experiment[-1]["status"] == "COMPLETED"
  ]

  paper_results = {}

  cov_names = set()

  for (run_dir, config_json, run_json) in tqdm.tqdm(runs):
    expkey = get_expkey((run_dir, config_json, run_json))
    with open(os.path.join(run_dir, "result"), "rb") as f:
      result = pickle.load(f)
    beta_hat, covs, n_kept = compute_results_averaged(result)
    n_converged = onp.sum(result.sol.converged)
    paper_results[expkey] = {
        'beta_hat': beta_hat,
        'n_converged': n_converged,
        'n_kept': n_kept,
        **covs
    }
    cov_names = cov_names.union(covs.keys())

  df = pd.DataFrame(columns=["beta_hat", "n_converged", "n_kept"] +
                    list(sorted(cov_names)),
                    index=pd.MultiIndex.from_tuples(paper_results.keys(),
                                                    names=expkey_names))
  for k1, r in paper_results.items():
    for k2, v in r.items():
      df[k2].loc[k1] = v

  df.to_csv(args.out_csv)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--runs_dir', type=str, default="runs/")
  parser.add_argument('--out_csv', type=str, default="paper_results.csv")
  main(parser.parse_args())
