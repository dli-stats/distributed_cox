"""Eq3 single experiment."""

import tempfile
import functools
import pickle
import collections

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.data import group_data_by_labels

from varderiv.equations.eq1 import eq1_log_likelihood_grad_ad
from varderiv.equations.eq1 import eq1_log_likelihood_grad_manual

from varderiv.equations.eq3 import (get_eq3_solver, get_eq3_cov_fn,
                                    eq3_cov_robust_ad)

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem
from varderiv.experiments.utils import check_value_converged

from varderiv.experiments.common import ingredient as base_ingredient
from varderiv.experiments.common import process_params

# pylint: disable=missing-function-docstring

Experiment3CovResult = collections.namedtuple("Experiment3CovResult",
                                              "cov_robust cov_H")


def cov_experiment_eq3_init(params):
  solver_max_steps = params.pop("solver_max_steps", 80)

  if params["eq1_ll_grad_use_ad"]:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_ad
  else:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_manual
  solve_eq3_fn = jit(
      get_eq3_solver(eq1_ll_grad_fn, solver_max_steps=solver_max_steps))
  eq3_cov_fn = jit(get_eq3_cov_fn(eq1_ll_grad_fn))

  params["solve_eq3_fn"] = solve_eq3_fn
  params["eq3_cov_fn"] = eq3_cov_fn
  params["eq3_cov_robust_fn"] = jit(eq3_cov_robust_ad)

  del params["eq1_ll_grad_use_ad"]


def cov_experiment_eq3_core(rnd_keys,
                            N=1000,
                            X_DIM=4,
                            K=3,
                            gen=None,
                            solve_eq3_fn=None,
                            eq3_cov_fn=None,
                            eq3_cov_robust_fn=None):
  del N, X_DIM
  assert gen is not None
  assert solve_eq3_fn is not None
  assert eq3_cov_fn is not None
  assert eq3_cov_robust_fn is not None

  key, data_generation_key = map(np.array, zip(*rnd_keys))
  assert key.shape == data_generation_key.shape

  X, delta, beta, group_labels = gen(data_generation_key)

  batch_size = len(X)

  X_groups, delta_groups = group_data_by_labels(batch_size, K, X, delta,
                                                group_labels)

  sol = solve_eq3_fn(X_groups, delta_groups, beta)
  beta_hat = sol.guess

  sol = expand_namedtuples(type(sol)(*map(onp.array, sol)))

  cov = onp.array(eq3_cov_fn(X_groups, delta_groups, beta_hat))
  cov_robust = onp.array(eq3_cov_robust_fn(X_groups, delta_groups, beta_hat))

  ret = expand_namedtuples(
      CovExperimentResultItem(sol=sol,
                              cov=expand_namedtuples(
                                  Experiment3CovResult(cov_robust=cov_robust,
                                                       cov_H=cov))))
  return ret


cov_experiment_eq3 = functools.partial(
    run_cov_experiment,
    cov_experiment_eq3_init,
    cov_experiment_eq3_core,
    check_fail_fn=lambda r: check_value_converged(r.sol.value))

ex = Experiment("eq3", ingredients=[base_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  eq1_ll_grad_use_ad = True


@ex.main
def cov_experiment_eq3_main(base, eq1_ll_grad_use_ad):
  # pylint: disable=missing-function-docstring
  base = process_params(**base)
  base.pop("seed")
  pickle.dump(cov_experiment_eq3(eq1_ll_grad_use_ad=eq1_ll_grad_use_ad, **base),
              result_file)
  ex.add_artifact(result_file.name, name="result")


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  result_file = tempfile.NamedTemporaryFile(mode="wb+")
  ex.run_commandline()
