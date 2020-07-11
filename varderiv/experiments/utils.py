"""Variance Experiment Utilities."""

import sys
import time
import collections
import itertools

from multiprocessing.dummy import Pool as ThreadPool

import numpy as onp

import tqdm

from jax import random as jrandom
from jax import jit

from varderiv.data import (key, data_generation_key, group_sizes_generator,
                           data_generator)


def in_notebook():
  """Detects if we are on IPython or regular script.

  Ref. https://stackoverflow.com/questions/15411967/
  """
  if 'google.colab' in sys.modules:
    return True
  try:
    cfg = get_ipython().config
    if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
      return True
    else:
      return False
  except NameError:
    return False


def grouper(iterable, n, fillvalue=None):
  """grouper
    (3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"""
  return itertools.zip_longest(*[iter(iterable)] * n, fillvalue=fillvalue)


# Common experiment result namedtuples
CovExperimentResultItem = collections.namedtuple("CovExperimentResultItem",
                                                 "sol cov")

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


def init_data_gen_fn(params, **extra_data_gen_params):
  """Initializes data generation."""
  group_sizes = group_sizes_generator(
      params["N"],
      params["K"],
      group_labels_generator_kind=params["group_labels_generator_kind"],
      **params["group_labels_generator_kind_kwargs"])
  del params["group_labels_generator_kind"]
  del params["group_labels_generator_kind_kwargs"]

  X_generator = params.pop("X_generator", None)
  gen = jit(
      data_generator(params["N"],
                     params["X_DIM"],
                     group_sizes,
                     exp_scale=params.pop('exp_scale', 3.5),
                     T_star_factors=params.pop('T_star_factors', None),
                     X_generator=X_generator,
                     **extra_data_gen_params))
  params["gen"] = gen


def run_cov_experiment(
    init_fn,
    experiment_fn,
    check_fail_fn=lambda x: False,
    data_generation_key=data_generation_key,  # pylint: disable=redefined-outer-name
    num_experiments=1000,
    num_threads=8,
    batch_size=32,
    experiment_rand_key=key,
    **experiment_params):
  """Helper function to run experiment in parallel."""

  assert batch_size >= 1, "Invalid batch_size"

  subkeys = jrandom.split(experiment_rand_key, num_experiments)
  data_generation_subkeys = jrandom.split(data_generation_key, num_experiments)
  all_data = list(zip(range(num_experiments), subkeys, data_generation_subkeys))

  # We fill in some arbitrary key value for the residuals
  data_iterator = grouper(all_data,
                          batch_size,
                          fillvalue=(-1, subkeys[0], subkeys[0]))

  init_data_gen_fn(experiment_params)
  init_fn(experiment_params)

  if num_threads > 1:
    pool = ThreadPool(num_threads)
    parallel_map = pool.imap_unordered
  else:
    parallel_map = map

  def experiment_fn_wrapper(args):
    i = [i for i, *_ in args]
    keys = [keys for _, *keys in args]
    return i, experiment_fn(keys, **experiment_params)

  desc = "Experiment {}".format(experiment_fn.__name__)
  if in_notebook():
    pbar = tqdm.tqdm_notebook(desc=desc, total=num_experiments)
  else:
    pbar = tqdm.tqdm(desc=desc, total=num_experiments)
  pbar.update(0)
  time.sleep(1)

  num_processed = 0
  results, failed = [], []
  for batch_idxs, batch_sols in parallel_map(experiment_fn_wrapper,
                                             data_iterator):
    num_succeed = 0
    for idx, sol in zip(batch_idxs, batch_sols):
      if num_processed >= num_experiments:
        continue
      if check_fail_fn(sol):
        failed.append(idx)
      else:
        results.append(sol)
        num_succeed += 1
      num_processed += 1
    pbar.update(len(batch_sols))
  pbar.close()

  print("Failed {}".format(len(failed)))

  failed_data = [(fi, all_data[fi]) for fi in failed]
  if len(failed_data) > 0:
    pass  # TODO(camyang) need to further process failed data

  if num_threads > 1:
    pool.close()
    pool.join()

  return ExperimentResult(data_generation_key=data_generation_key,
                          results=results)


def check_value_converged(value, tol=1e-3):
  return onp.all(
      onp.isfinite(value)) and onp.linalg.norm(value, ord=onp.inf) > tol
