"""Variance Experiment Utilities."""

import math
import time
import collections
import itertools
import functools

from multiprocessing.dummy import Pool as ThreadPool

import tqdm

import numpy as onp

import jax.numpy as np
from jax import random as jrandom

from varderiv.data import key, data_generation_key


def process_params(**params):
  return params


def grouper(iterable, n, fillvalue=None):
  """grouper
    (3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"""
  return itertools.zip_longest(*[iter(iterable)] * n, fillvalue=fillvalue)


# Common experiment result namedtuples
ExperimentResultItem = collections.namedtuple("ExperimentResultItem", "sol cov")

ExperimentResult = collections.namedtuple("ExperimentResult",
                                          "data_generation_key results")


def expand_namedtuples(tup):
  tup_t = type(tup)
  tup_d = tup._asdict()
  num_items = len(tup_d[tup._fields[0]])
  ret = []
  for i in range(num_items):
    ret.append(tup_t(**{field: values[i] for field, values in tup_d.items()}))
  return ret


def run_cov_experiment(
    init_fn,
    experiment_fn,
    data_generation_key=data_generation_key,  #pylint: disable=redefined-outer-name
    num_experiments=1000,
    num_threads=8,
    batch_size=32,
    experiment_rand_key=key,
    **experiment_params):

  assert batch_size >= 1, "Invalid batch_size"

  subkeys = jrandom.split(experiment_rand_key, num_experiments)
  data_generation_subkeys = jrandom.split(data_generation_key, num_experiments)
  data_iterator = list(zip(subkeys, data_generation_subkeys))

  # We fill in some arbitrary key value for the residuals
  data_iterator = grouper(data_iterator,
                          batch_size,
                          fillvalue=(subkeys[0], subkeys[0]))

  init_fn(experiment_params)

  results = []
  with ThreadPool(num_threads) as pool:
    with tqdm.tqdm_notebook(desc="Experiments {}".format(
        experiment_fn.__name__),
                            total=num_experiments) as pbar:
      pbar.update(0)
      time.sleep(1)
      for sol in pool.imap_unordered(
          functools.partial(experiment_fn, **experiment_params), data_iterator):
        results += sol
        pbar.update(len(sol))

  # Trim results that are padded
  results = results[:num_experiments]

  return ExperimentResult(data_generation_key=data_generation_key,
                          results=results)


def plot_cov_experiment(result: ExperimentResult,
                        sol_get_beta,
                        get_cov_mapping=None):
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

  print("Data generated using master key {}".format(result.data_generation_key))

  all_covs = collections.OrderedDict()

  results = result.results

  beta = onp.stack([sol_get_beta(r.sol) for r in results])
  beta_empirical_nan_idxs = onp.any(onp.isnan(beta), axis=1)
  beta = beta[~beta_empirical_nan_idxs]
  print("Cov empirical has {} nans".format(np.sum(beta_empirical_nan_idxs)))

  beta_norm = onp.linalg.norm(beta, axis=1)
  beta_norm_median = onp.median(beta_norm, axis=0)
  outlier_betas_idxs = beta_norm > 100 * beta_norm_median
  beta = beta[~outlier_betas_idxs]
  print("Cov empirical has {} outliers".format(np.sum(outlier_betas_idxs)))

  cov_empirical = onp.cov(beta, rowvar=False)
  all_covs["Empirical"] = cov_empirical

  for analytical_name, get_cov_fn in get_cov_mapping.items():
    cov_analyticals = onp.array([get_cov_fn(r.cov) for r in results])
    cov_analyticals_nan_idxs = onp.any(
        onp.isnan(cov_analyticals.reshape(-1, cov_analyticals.shape[1]**2)),  # pylint:disable=unsubscriptable-object
        axis=1)
    print("Cov {} has {} nans".format(analytical_name,
                                      np.sum(cov_analyticals_nan_idxs)))
    cov_analyticals = cov_analyticals[~cov_analyticals_nan_idxs]
    cov_analytical = onp.mean(cov_analyticals, axis=0)
    all_covs[analytical_name] = cov_analytical

  for name, cov in all_covs.items():
    print(f"{name}:")
    print(cov)

  ncols = 2
  nrows = int(math.ceil(len(all_covs) / ncols))
  fig, axes = plt.subplots(nrows=nrows,
                           ncols=ncols,
                           figsize=np.array([ncols, nrows]) * 3.5)

  vmin = min(*[onp.min(cov) for cov in all_covs.values()])
  vmax = max(*[onp.max(cov) for cov in all_covs.values()])

  for i, (name, cov) in enumerate(all_covs.items()):
    ax = axes.flat[i]
    ax.set_title(name, pad=20)
    im = ax.matshow(cov, vmin=vmin, vmax=vmax)

  fig.subplots_adjust(right=0.8, hspace=0.5)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(im, cax=cbar_ax)

  plt.show()
