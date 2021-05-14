"""Experiment result collection."""

import glob
import os
import json
import argparse
import pickle
import collections
import functools

import numpy as onp
import pandas as pd
import scipy.stats
import jax.scipy.stats
import jax.numpy as jnp
from jax import vmap, jit

from sacred.experiment import Experiment  # pylint: disable=unused-import
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


expkey_names = (
    "method data.K data.N data.T_star_factors data.X_DIM data.npz_path "
    "data.group_X data.group_labels_generator_kind "
    "distributed.taylor_order meta_analysis.univariate").split()


def get_expkey(experiment):
  _, config_json, _ = experiment
  expkey = []
  for n in expkey_names:
    k = config_json
    try:
      for part in n.split("."):
        k = k[part]
    except KeyError:
      k = None
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


def compute_confidence_interval_overlap(beta,
                                        var,
                                        beta_true,
                                        var_true,
                                        lb=0.025,
                                        ub=1 - 0.025):
  std, std_true = jnp.sqrt(var), jnp.sqrt(var_true)
  f_orig_cdf = functools.partial(jax.scipy.stats.norm.cdf,
                                 loc=beta_true,
                                 scale=std_true)
  f_rel_cdf = functools.partial(jax.scipy.stats.norm.cdf, loc=beta, scale=std)
  L_rel = beta + scipy.stats.norm.ppf(lb) * std
  U_rel = beta + scipy.stats.norm.ppf(ub) * std
  L_orig = beta_true + scipy.stats.norm.ppf(lb) * std_true
  U_orig = beta_true + scipy.stats.norm.ppf(ub) * std_true

  I = ((f_orig_cdf(U_rel) - f_orig_cdf(L_rel)) +
       (f_rel_cdf(U_orig) - f_rel_cdf(L_orig))) / 2

  return I


def compute_confidence_interval_overlap_clip(beta,
                                             var,
                                             beta_true,
                                             lb=0.025,
                                             ub=1 - 0.025):
  std = jnp.sqrt(var)
  L_rel = beta + scipy.stats.norm.ppf(lb) * std
  U_rel = beta + scipy.stats.norm.ppf(ub) * std
  clip = jnp.logical_and(L_rel < beta_true, beta_true < U_rel)
  return clip


def _get_true_model_in_group(group):
  _, config_json, _ = group[0]
  assert config_json["data"]["T_star_factors"] is not None
  if config_json["data"]["T_star_factors"] == "None":
    method_true = "unstratified_pooled"
  else:
    method_true = "stratified_pooled"
  return next(setting for setting in group if setting[1]["eq"] == method_true)


def _eq_get_var(config_json, result):
  method = config_json["method"]
  if method == "unstratified_pooled":
    if config_json["data"]["T_star_factors"] == "None":
      kind = "ese"
    else:
      kind = "rse"
  elif method == "unstratified_distributed":
    kind = "rse"
  else:
    kind = "ese"

  mapping = {
      ("unstratified_pooled", "ese"):
          "cov:no_group_correction|no_sandwich|no_cox_correction|no_sum_first",
      ("unstratified_pooled", "rse"):
          "cov:no_group_correction|sandwich|cox_correction|no_sum_first",
      ("unstratified_distributed", "ese"):
          "cov:group_correction|no_sandwich|no_cox_correction|no_sum_first",
      ("unstratified_distributed", "rse"):
          "cov:group_correction|sandwich|cox_correction|no_sum_first",
      ("stratified_pooled", "ese"):
          "cov:no_group_correction|no_sandwich|no_cox_correction|no_sum_first",
      ("stratified_pooled", "rse"):
          None,
      ("stratified_distributed", "ese"):
          "cov:group_correction|no_sandwich|no_cox_correction|no_sum_first",
      ("stratified_distributed", "rse"):
          None,
      ("meta_analysis", "ese"):
          "cov:meta_analysis",
      ("meta_analysis", "rse"):
          None
  }

  cov_kind = mapping[method, kind]
  return onp.diagonal(result.covs[cov_kind], axis1=1, axis2=2)


def main(args):
  if args.compute_confidence_interval_overlap:
    args.intersect_kept = True

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

  compute_confidence_interval_overlap_jit = jit(
      vmap(compute_confidence_interval_overlap))
  compute_confidence_interval_overlap_clip_jit = jit(
      vmap(compute_confidence_interval_overlap_clip,
           in_axes=(0, 0, None),
           out_axes=0))

  for (run_dir, config_json, run_json) in tqdm.tqdm(runs):
    exp_key = get_expkey((run_dir, config_json, run_json))
    data_key = json.dumps(config_json["data"], sort_keys=True)
    with open(os.path.join(run_dir, "result"), "rb") as f:
      result: ExperimentResult = pickle.load(f)
    keep_idxs = same_data_setting_kept_idxs.get(data_key, None)
    beta_hat, covs, keep_idxs = compute_results_averaged(result,
                                                         std=args.std,
                                                         keep_idxs=keep_idxs)
    n_converged = onp.sum(result.sol.converged)

    beta_l1_norm = onp.mean(
        onp.abs(result.guess[keep_idxs] -
                same_X_dim_beta_true[config_json["data"]["X_DIM"]]),
        axis=0)

    # Get true model result
    if args.compute_confidence_interval_overlap:
      true_model_run_dir, true_model_config_json, _ = _get_true_model_in_group(
          same_data_setting_groups[data_key])
      with open(os.path.join(true_model_run_dir, "result"), "rb") as f:
        result_true_model: ExperimentResult = pickle.load(f)

      cio_stats = compute_confidence_interval_overlap_jit(
          result.guess,
          _eq_get_var(config_json, result),
          result_true_model.guess,
          _eq_get_var(true_model_config_json, result_true_model),
      )
      cio_stats = onp.mean(cio_stats[keep_idxs], axis=0)

      cr_stats = compute_confidence_interval_overlap_clip_jit(
          result.guess, _eq_get_var(config_json, result),
          same_X_dim_beta_true[config_json["data"]["X_DIM"]])
      cr_stats = cr_stats[keep_idxs]
      cr_stats1 = onp.mean(cr_stats, axis=0)
      cr_stats2 = onp.mean(onp.all(cr_stats, axis=1))

    else:
      cio_stats = None
      cr_stats1 = None
      cr_stats2 = None

    paper_results[exp_key] = {
        'beta_hat': beta_hat,
        "beta_l1_norm": beta_l1_norm,
        'n_converged': n_converged,
        'n_kept': onp.sum(keep_idxs),
        'cio_stats': cio_stats,
        'cr_stats_mean': cr_stats1,
        'cr_stats_all': cr_stats2,
        **covs
    }
    cov_names = cov_names.union(covs.keys())

  df = pd.DataFrame(columns=[
      "beta_hat",
      "n_converged",
      "n_kept",
      "beta_l1_norm",
      "cio_stats",
      "cr_stats_mean",
      "cr_stats_all",
  ] + list(sorted(cov_names)),
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
  parser.add_argument('--std', action="store_true", default=False)
  parser.add_argument('--intersect-kept', action="store_true", default=False)
  parser.add_argument('--compute_confidence_interval_overlap',
                      action="store_true",
                      default=False)
  main(parser.parse_args())
