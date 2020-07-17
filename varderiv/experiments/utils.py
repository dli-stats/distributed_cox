"""Variance Experiment Utilities."""

from typing import Dict, BinaryIO

import sys
import time
import collections
import dataclasses
import itertools
import pickle

from multiprocessing.dummy import Pool as ThreadPool

import numpy as onp

import tqdm

import jax.tree_util as tu
from jax import random as jrandom
from jax import jit, vmap

import varderiv.data as data


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


def run_cov_experiment(init_fn,
                       experiment_fn,
                       data_generation_key,
                       experiment_rand_key,
                       check_ok_fn=lambda x: True,
                       num_experiments=1000,
                       num_threads=8,
                       batch_size=32,
                       save_interval=1,
                       result_file: BinaryIO = None,
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

  experiment_params = init_fn(**experiment_params)
  check_ok_fn = jit(vmap(check_ok_fn))

  def save(result):
    if result_file is not None:
      pickle.dump(result, result_file)

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

  result = None
  num_failed = 0
  for i, (_, batch_sols) in enumerate(
      parallel_map(experiment_fn_wrapper, data_iterator)):
    oks = check_ok_fn(batch_sols)
    num_failed += len(oks) - onp.sum(oks)
    pbar.set_description("{} (failed {})".format(desc, num_failed))
    if result is None:
      result = tu.tree_map(onp.array, batch_sols)
    else:
      result = tu.tree_multimap(lambda x, y: onp.concatenate([x, y]), result,
                                batch_sols)
    if i > 0 and i % save_interval == 0:
      save(result)
    pbar.update(batch_size)
  pbar.close()

  if i % save_interval != 0:
    # Save final result
    save(result)

  print("Total failed: {}".format(num_failed))

  if num_threads > 1:
    pool.close()
    pool.join()

  return result


def check_value_converged(value, tol=1e-3):
  return onp.all(
      onp.isfinite(value)) and onp.linalg.norm(value, ord=onp.inf) > tol


def check_solver_converged(sol):
  return sol.converged
