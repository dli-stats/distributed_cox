"""Equation single experiment."""

import numpy as onp

import jax.numpy as np
from jax import jit
import jax.random as jrandom

from sacred import Experiment

from varderiv.data import data_generator
from varderiv.equations.eq1 import solve_eq1_ad, solve_eq1_manual
from varderiv.equations.eq1 import eq1_cov_ad, eq1_cov_manual

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem

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


ex = Experiment("eq1")


@ex.config
def config():
  # pylint: disable=unused-variable
  num_experiments = 100000
  num_threads = 1
  batch_size = 128
  N = 500
  X_DIM = 3
  solve_eq1_use_ad = True
  eq1_cov_use_ad = True

  seed = 0
  key = jrandom.PRNGKey(seed)
  experiment_rand_key, data_generation_key = jrandom.split(key, 2)
  del key


@ex.main
def cov_experiment_eq1(experiment_rand_key,
                       data_generation_key,
                       N=500,
                       X_DIM=3,
                       num_experiments=100000,
                       num_threads=1,
                       batch_size=128,
                       solve_eq1_use_ad=True,
                       eq1_cov_use_ad=False):
  return run_cov_experiment(cov_experiment_eq1_init,
                            cov_experiment_eq1_core,
                            data_generation_key=data_generation_key,
                            num_experiments=num_experiments,
                            num_threads=num_threads,
                            batch_size=batch_size,
                            experiment_rand_key=experiment_rand_key,
                            N=N,
                            X_DIM=X_DIM,
                            solve_eq1_use_ad=solve_eq1_use_ad,
                            eq1_cov_use_ad=eq1_cov_use_ad)


if __name__ == '__main__':
  ex.run_commandline()
