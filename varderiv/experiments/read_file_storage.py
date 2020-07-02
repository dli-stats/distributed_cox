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

# pylint: disable=missing-docstring, unused-import

from varderiv.experiments.eq1 import Experiment1CovResult
from varderiv.experiments.eq2 import Experiment2SolResult, Experiment2CovResult
from varderiv.experiments.eq3 import Experiment3CovResult
from varderiv.experiments.eq4 import Experiment4SolResult, Experiment4CovResult


def iterate_experiements(runs_dir):
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


def get_paper_data(result):
  """
  Args:
    - result: the experiment result object.
    - A function given a result.sol, returns the beta
    - A dict from str to function. The string represents a key
      to plot the analytical cov. The function returns the cov
      matrices from result.cov.
  """
  if get_cov_mapping is None:
    get_cov_mapping = {"Analytical": lambda cov: cov}

  all_covs = collections.OrderedDict()

  results = result.results

  if "pt2" in results[0].sol._fields:
    sol_get_beta = lambda sol: sol.pt2.guess
  else:
    sol_get_beta = lambda sol: sol.guess

  beta = onp.stack([sol_get_beta(r.sol) for r in results])

  beta_empirical_nan_idxs = onp.any(onp.isnan(beta), axis=1)
  # print("Cov empirical has {} nans".format(np.sum(beta_empirical_nan_idxs)))

  beta_norm = onp.linalg.norm(onp.abs(beta), axis=1)
  out_of_range_beta_idxs = onp.any(beta < -2, axis=1) | onp.any(beta > 2,
                                                                axis=1)
  # beta_norm_median = onp.median(beta_norm, axis=0)
  outlier_betas_idxs = out_of_range_beta_idxs
  # print("Cov empirical has {} outliers".format(
  #     np.sum(outlier_betas_idxs)))

  beta = beta[~outlier_betas_idxs & ~beta_empirical_nan_idxs]
  beta_hat = onp.average(beta, axis=0)

  cov_empirical = onp.cov(beta, rowvar=False)
  if cov_empirical.shape == tuple():
    cov_empirical = cov_empirical.reshape((1, 1))
  all_covs["Empirical"] = cov_empirical

  for analytical_name in results[0].cov._fields:
    cov_analyticals = onp.array(
        [getattr(r.cov, analytical_name) for r in results])
    cov_analyticals = cov_analyticals[~outlier_betas_idxs &
                                      ~beta_empirical_nan_idxs]
    cov_analyticals_nan_idxs = onp.any(onp.isnan(
        cov_analyticals.reshape(-1, cov_analyticals.shape[1]**2)),
                                       axis=1)
    # print("Cov {} has {} nans".format(
    #     analytical_name,
    # np.sum(cov_analyticals_nan_idxs)))
    out_of_range_cov_analyticals_idxs = onp.any(
        cov_analyticals.reshape(-1, cov_analyticals.shape[1]**2) < -1000,
        axis=1)
    cov_analyticals = cov_analyticals[~cov_analyticals_nan_idxs &
                                      ~out_of_range_cov_analyticals_idxs]
    cov_analytical = onp.mean(cov_analyticals, axis=0)
    all_covs[analytical_name] = cov_analytical

  all_stds = {}
  for n in all_covs:
    all_stds[n] = onp.sqrt(onp.diagonal(all_covs[n]))

  return beta_hat, all_stds


def main(args):
  runs = [
      experiment for experiment in iterate_experiements(args.runs_dir)
      if experiment[-1]["status"] == "COMPLETED"
  ]

  paper_results = {}

  cov_names = set()

  expkey_names = ("K N T T_star_factors X_DIM "
                  "group_X_same group_labels_generator_kind").split()

  for (run_dir, config_json, run_json) in tqdm.tqdm(runs):
    eq = run_json["name"]
    expkey = tuple(config_json["base"][n] for n in expkey_names)
    with open(os.path.join(run_dir, "result"), "rb") as f:
      result = pickle.load(f)

    beta_hat, covs = get_paper_data(result)
    paper_results[(eq, *expkey)] = {'beta_hat': beta_hat, **covs}
    cov_names = cov_names.union(covs.keys())

  df = pd.DataFrame(columns=["beta_hat"] + list(sorted(cov_names)),
                    index=pd.MultiIndex.from_tuples(paper_results.keys(),
                                                    names=["eq"] +
                                                    expkey_names))
  for k1, r in paper_results.items():
    for k2, v in r.items():
      df[k2].loc[k1] = v

  return df


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--runs_dir', type=str, default="runs/")
  main(parser.parse_args())
