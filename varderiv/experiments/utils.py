"""Variance Experiment Utilities."""

import sys
import time
import collections
import itertools
import functools

from multiprocessing.dummy import Pool as ThreadPool

import tqdm

from jax import random as jrandom

from varderiv.data import key, data_generation_key


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


def run_cov_experiment(
    init_fn,
    experiment_fn,
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
  data_iterator = list(zip(subkeys, data_generation_subkeys))

  # We fill in some arbitrary key value for the residuals
  data_iterator = grouper(data_iterator,
                          batch_size,
                          fillvalue=(subkeys[0], subkeys[0]))

  init_fn(experiment_params)

  results = []

  if num_threads > 1:
    pool = ThreadPool(num_threads)
    parallel_map = pool.imap_unordered
  else:
    parallel_map = map

  desc = "Experiment {}".format(experiment_fn.__name__)
  if in_notebook():
    pbar = tqdm.tqdm_notebook(desc=desc, total=num_experiments)
  else:
    pbar = tqdm.tqdm(desc=desc, total=num_experiments)
  pbar.update(0)
  time.sleep(1)
  for sol in parallel_map(functools.partial(experiment_fn, **experiment_params),
                          data_iterator):
    results += sol
    pbar.update(len(sol))
  pbar.close()

  if num_threads > 1:
    pool.close()
    pool.join()

  # Trim results that are padded
  results = results[:num_experiments]

  return ExperimentResult(data_generation_key=data_generation_key,
                          results=results)
