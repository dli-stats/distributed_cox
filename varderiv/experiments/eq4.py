"""Eq4 single experiment."""

import tempfile
import collections
import functools
import pickle

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.data import group_data_by_labels

from varderiv.equations.eq1 import get_eq1_solver
from varderiv.equations.eq1 import eq1_compute_H_ad, eq1_compute_H_manual
from varderiv.equations.eq1 import (eq1_log_likelihood_grad_ad,
                                    eq1_log_likelihood_grad_manual)

from varderiv.equations.eq2 import solve_grouped_eq_batch

from varderiv.equations.eq4 import get_eq4_cov_beta_k_correction_fn
from varderiv.equations.eq4 import get_eq4_rest_solver

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem
from varderiv.experiments.utils import check_value_converged

from varderiv.experiments.common import ingredient as base_ingredient

Experiment4SolResult = collections.namedtuple("Experiment4SolResult", "pt1 pt2")
Experiment4CovResult = collections.namedtuple(
    "Experiment4CovResult",
    "cov_beta_k_correction cov_beta_k_correction_robust")


def cov_experiment_eq4_init(params):
  """Initializes parameters for Eq 4."""
  solve_eq1_use_ad = params.pop("solve_eq1_use_ad", True)
  solver_max_steps = params.pop("solver_max_steps", 80)

  if params["eq1_cov_use_ad"]:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_ad
    eq1_compute_H_fn = eq1_compute_H_ad
  else:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_manual
    eq1_compute_H_fn = eq1_compute_H_manual
  del params["eq1_cov_use_ad"]

  params["solve_eq4_fn"] = functools.partial(
      solve_grouped_eq_batch,
      solve_eq1_fn=jit(
          get_eq1_solver(use_ad=solve_eq1_use_ad,
                         solver_max_steps=solver_max_steps)),
      solve_rest_fn=jit(get_eq4_rest_solver(solver_max_steps=solver_max_steps)))

  params["cov_beta_k_correction_fn"] = jit(
      get_eq4_cov_beta_k_correction_fn(eq1_compute_H_fn=eq1_compute_H_fn))

  params["cov_beta_k_correction_robust_fn"] = jit(
      get_eq4_cov_beta_k_correction_fn(robust=True,
                                       eq1_ll_grad_fn=eq1_ll_grad_fn,
                                       eq1_compute_H_fn=eq1_compute_H_fn))


def cov_experiment_eq4_core(  # pylint: disable=dangerous-default-value
    rnd_keys,
    N=1000,
    X_DIM=4,
    K=3,
    gen=None,
    solve_eq4_fn=None,
    cov_beta_k_correction_fn=None,
    cov_beta_k_correction_robust_fn=None):
  """Equation 4 experiment core."""
  del N

  key, data_generation_key = map(np.array, zip(*rnd_keys))
  del key

  X, delta, beta, group_labels = gen(data_generation_key)

  batch_size = len(X)
  assert beta.shape == (batch_size, X_DIM)

  X_groups, delta_groups = group_data_by_labels(batch_size, K, X, delta,
                                                group_labels)

  initial_guess = beta
  pt1_sols, pt2_sols = solve_eq4_fn(X,
                                    delta,
                                    K,
                                    group_labels,
                                    X_groups=X_groups,
                                    delta_groups=delta_groups,
                                    initial_guess=initial_guess,
                                    log=False)

  beta_k_hat = pt1_sols.guess
  beta_hat = pt2_sols.guess

  cov_beta_k_correction = cov_beta_k_correction_fn(X, delta, X_groups,
                                                   delta_groups, group_labels,
                                                   beta_k_hat, beta_hat)
  cov_beta_k_correction = onp.array(cov_beta_k_correction)

  cov_beta_k_correction_robust = cov_beta_k_correction_robust_fn(
      X, delta, X_groups, delta_groups, group_labels, beta_k_hat, beta_hat)
  cov_beta_k_correction = onp.array(cov_beta_k_correction)

  beta_k_hat = onp.array(beta_k_hat)
  beta_hat = onp.array(beta_hat)

  pt1_sols = expand_namedtuples(type(pt1_sols)(*map(onp.array, pt1_sols)))
  pt2_sols = expand_namedtuples(type(pt2_sols)(*map(onp.array, pt2_sols)))

  ret = expand_namedtuples(
      CovExperimentResultItem(
          sol=expand_namedtuples(
              Experiment4SolResult(pt1=pt1_sols, pt2=pt2_sols)),
          cov=expand_namedtuples(
              Experiment4CovResult(
                  cov_beta_k_correction=cov_beta_k_correction,
                  cov_beta_k_correction_robust=cov_beta_k_correction_robust))))
  return ret


cov_experiment_eq4 = functools.partial(
    run_cov_experiment,
    cov_experiment_eq4_init,
    cov_experiment_eq4_core,
    check_fail_fn=lambda r: check_value_converged(r.sol.pt2.value))

ex = Experiment("eq4", ingredients=[base_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  solve_eq1_use_ad = True
  eq1_cov_use_ad = True


@ex.main
def cov_experiment_eq4_main(base, solve_eq1_use_ad, eq1_cov_use_ad):
  # pylint: disable=missing-function-docstring
  base = dict(base)
  base.pop("seed")
  pickle.dump(
      cov_experiment_eq4(solve_eq1_use_ad=solve_eq1_use_ad,
                         eq1_cov_use_ad=eq1_cov_use_ad,
                         **base), result_file)
  ex.add_artifact(result_file.name, name="result")


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  result_file = tempfile.NamedTemporaryFile(mode="wb+")
  ex.run_commandline()
