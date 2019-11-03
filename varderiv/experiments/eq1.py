"""Eq1 single experiment."""

import tempfile
import functools
import pickle

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.data import data_generator
from varderiv.equations.eq1 import solve_eq1_ad, solve_eq1_manual
from varderiv.equations.eq1 import eq1_cov_ad, eq1_cov_manual

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem

from varderiv.experiments.common import ingredient as base_ingredient

# pylint: disable=missing-function-docstring


def cov_experiment_eq1_init(params):
  gen = jit(data_generator(params["N"], params["X_DIM"]))
  params["gen"] = gen
  del params["N"], params["X_DIM"]

  if params["solve_eq1_use_ad"]:
    solve_eq1_fn = jit(solve_eq1_ad)
  else:
    solve_eq1_fn = jit(solve_eq1_manual)
  params["solve_eq1_fn"] = solve_eq1_fn
  del params["solve_eq1_use_ad"]

  if params["eq1_cov_use_ad"]:
    eq1_cov_fn = jit(eq1_cov_ad)
  else:
    eq1_cov_fn = jit(eq1_cov_manual)
  params["eq1_cov_fn"] = eq1_cov_fn
  del params["eq1_cov_use_ad"]


def cov_experiment_eq1_core(args, gen=None, solve_eq1_fn=None, eq1_cov_fn=None):
  assert gen is not None
  assert solve_eq1_fn is not None
  assert eq1_cov_fn is not None

  key, data_generation_key = map(np.array, zip(*args))
  X, delta, beta = gen(data_generation_key)

  assert key.shape == data_generation_key.shape

  sol = solve_eq1_fn(key, X, delta, beta)
  beta_hat = sol.guess
  sol = expand_namedtuples(
      type(sol)(*map(onp.array, sol)))  # release from jax to numpy
  cov = onp.array(eq1_cov_fn(X, delta, beta_hat))
  ret = expand_namedtuples(CovExperimentResultItem(sol=sol, cov=cov))
  return ret


cov_experiment_eq1 = functools.partial(run_cov_experiment,
                                       cov_experiment_eq1_init,
                                       cov_experiment_eq1_core)

ex = Experiment("eq1", ingredients=[base_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  solve_eq1_use_ad = True
  eq1_cov_use_ad = True


@ex.main
def cov_experiment_eq1_main(base, solve_eq1_use_ad, eq1_cov_use_ad):
  # pylint: disable=missing-function-docstring
  base = dict(base)
  base.pop("seed")
  pickle.dump(
      cov_experiment_eq1(solve_eq1_use_ad=solve_eq1_use_ad,
                         eq1_cov_use_ad=eq1_cov_use_ad,
                         **base), result_file)
  ex.add_artifact(result_file.name, name="result")


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  result_file = tempfile.NamedTemporaryFile(mode="wb+")
  ex.run_commandline()
