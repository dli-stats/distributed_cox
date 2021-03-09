"""Experiment result collection."""

import glob
import os
import json
import argparse
import pickle
import collections

import numpy as onp
import pandas as pd
from sacred.experiment import Experiment
import tqdm
from distributed_cox.experiments.run import (compute_results_averaged,
                                             init_data_gen_fn, ExperimentResult)


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


expkey_names = ("eq data.K data.N data.T_star_factors data.X_DIM "
                "data.group_X data.group_labels_generator_kind "
                "distributed.taylor_order meta_analysis.univariate").split()


def get_expkey(experiment):
  _, config_json, _ = experiment
  expkey = []
  for n in expkey_names:
    k = config_json
    for part in n.split("."):
      k = k[part]
    expkey.append(k)
  return tuple(expkey)


def merge_experiments_same_setting(runs_dir, **kwargs):
  del kwargs
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

  same_data_setting_groups = collections.defaultdict(list)
  # Assumes data settings have the same beta_true if X_dim are the same
  same_X_dim_beta_true = {}
  for (run_dir, config_json, run_json) in tqdm.tqdm(runs):
    data_key = json.dumps(config_json["data"], sort_keys=True)
    same_data_setting_groups[data_key].append((run_dir, config_json, run_json))
    if config_json["data"]["X_DIM"] not in same_X_dim_beta_true:
      with open(os.path.join(run_dir, "result"), "rb") as f:
        result: ExperimentResult = pickle.load(f)
      _, data_gen = init_data_gen_fn(**config_json["data"])
      (_, _, _, _, beta_true, _) = data_gen(result.data_generation_key[0])
      same_X_dim_beta_true[config_json["data"]["X_DIM"]] = beta_true

  same_data_setting_kept_idxs = {}
  if args.intersect_kept:
    for (run_dir, config_json, run_json) in tqdm.tqdm(runs):
      with open(os.path.join(run_dir, "result"), "rb") as f:
        result: ExperimentResult = pickle.load(f)
      _, _, keep_idxs = compute_results_averaged(result,
                                                 std=args.std,
                                                 keep_idxs=None)
      data_key = json.dumps(config_json["data"], sort_keys=True)
      if data_key not in same_data_setting_kept_idxs:
        same_data_setting_kept_idxs[data_key] = keep_idxs
      else:
        same_data_setting_kept_idxs[data_key] &= keep_idxs

  for (run_dir, config_json, run_json) in tqdm.tqdm(runs):
    exp_key = get_expkey((run_dir, config_json, run_json))
    data_key = json.dumps(config_json["data"], sort_keys=True)
    with open(os.path.join(run_dir, "result"), "rb") as f:
      result = pickle.load(f)
    keep_idxs = same_data_setting_kept_idxs.get(data_key, None)
    beta_hat, covs, keep_idxs = compute_results_averaged(result,
                                                         std=args.std,
                                                         keep_idxs=keep_idxs)
    n_converged = onp.sum(result.sol.converged)

    beta_l1_norm = onp.mean(
        onp.linalg.norm(result.guess[keep_idxs] -
                        same_X_dim_beta_true[config_json["data"]["X_DIM"]],
                        ord=1,
                        axis=1),
        axis=0)

    paper_results[exp_key] = {
        'beta_hat': beta_hat,
        "beta_l1_norm": beta_l1_norm,
        'n_converged': n_converged,
        'n_kept': onp.sum(keep_idxs),
        **covs
    }
    cov_names = cov_names.union(covs.keys())

  df = pd.DataFrame(
      columns=["beta_hat", "n_converged", "n_kept", "beta_l1_norm"] +
      list(sorted(cov_names)),
      index=pd.MultiIndex.from_tuples(paper_results.keys(), names=expkey_names))
  for k1, r in paper_results.items():
    for k2, v in r.items():
      df[k2].loc[k1] = v

  df.to_csv(args.out_csv)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--runs_dir', type=str, default="runs/")
  parser.add_argument('--out_csv', type=str, default="paper_results.csv")
  parser.add_argument('--std', action="store_true", default=False)
  parser.add_argument('--intersect-kept', action="store_true", default=False)
  main(parser.parse_args())
