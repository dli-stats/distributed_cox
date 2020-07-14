"""Eq1 single experiment."""

import tempfile
import functools
import pickle
import collections

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.equations.eq1 import get_eq1_solver
from varderiv.equations.eq1 import (eq1_cov_ad, eq1_cov_manual,
                                    eq1_cov_robust_ad, eq1_cov_robust2_ad,
                                    eq1_cov_robust3_ad)

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem
from varderiv.experiments.utils import check_value_converged

from varderiv.experiments.common import ingredient as base_ingredient
from varderiv.experiments.common import process_params

# pylint: disable=missing-function-docstring

Experiment1CovResult = collections.namedtuple(
    "Experiment1CovResult", "cov_robust cov_robust2 cov_robust3 cov_H")


def cov_experiment_eq1_init(params):
  solver_max_steps = params.pop("solver_max_steps", 80)
  params["solve_eq1_fn"] = jit(
      get_eq1_solver(use_ad=params["solve_eq1_use_ad"],
                     solver_max_steps=solver_max_steps))
  del params["solve_eq1_use_ad"]

  if params["eq1_cov_use_ad"]:
    eq1_cov_fn = jit(eq1_cov_ad)
  else:
    eq1_cov_fn = jit(eq1_cov_manual)
  params["eq1_cov_fn"] = eq1_cov_fn

  params["eq1_cov_robust_fn"] = jit(eq1_cov_robust_ad)
  params["eq1_cov_robust2_fn"] = jit(eq1_cov_robust2_ad)
  params["eq1_cov_robust3_fn"] = jit(eq1_cov_robust3_ad)

  del params["eq1_cov_use_ad"]


def cov_experiment_eq1_core(rnd_keys,
                            N=1000,
                            X_DIM=4,
                            K=3,
                            gen=None,
                            solve_eq1_fn=None,
                            eq1_cov_fn=None,
                            eq1_cov_robust_fn=None,
                            eq1_cov_robust2_fn=None,
                            eq1_cov_robust3_fn=None):
  del N, X_DIM, K
  assert gen is not None
  assert solve_eq1_fn is not None
  assert eq1_cov_fn is not None
  assert eq1_cov_robust_fn is not None
  assert eq1_cov_robust2_fn is not None
  assert eq1_cov_robust3_fn is not None

  key, data_generation_key = map(np.array, zip(*rnd_keys))
  X, delta, beta, _ = gen(data_generation_key)

  assert key.shape == data_generation_key.shape

  sol = solve_eq1_fn(X, delta, beta)
  beta_hat = sol.guess
  sol = expand_namedtuples(
      type(sol)(*map(onp.array, sol)))  # release from jax to numpy

  cov = onp.array(eq1_cov_fn(X, delta, beta_hat))
  cov_robust = onp.array(eq1_cov_robust_fn(X, delta, beta_hat))
  cov_robust2 = onp.array(eq1_cov_robust2_fn(X, delta, beta_hat))
  cov_robust3 = onp.array(eq1_cov_robust3_fn(X, delta, beta_hat))

  ret = expand_namedtuples(
      CovExperimentResultItem(sol=sol,
                              cov=expand_namedtuples(
                                  Experiment1CovResult(cov_robust=cov_robust,
                                                       cov_robust2=cov_robust2,
                                                       cov_robust3=cov_robust3,
                                                       cov_H=cov))))
  return ret


cov_experiment_eq1 = functools.partial(
    run_cov_experiment,
    cov_experiment_eq1_init,
    cov_experiment_eq1_core,
    check_fail_fn=lambda r: check_value_converged(r.sol.value) > 1e-3)

ex = Experiment("eq1", ingredients=[base_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  solve_eq1_use_ad = True
  eq1_cov_use_ad = True


@ex.main
def cov_experiment_eq1_main(base, solve_eq1_use_ad, eq1_cov_use_ad):
  # pylint: disable=missing-function-docstring
  base = process_params(**base)
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
