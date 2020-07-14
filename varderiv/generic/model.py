"""Higher order functions for creating models."""

import functools
import collections

import jax.numpy as np

from varderiv.generic.solver import solve_newton

FitResult = collections.namedtuple("FitResult",
                                   "guess covs loglik steps converged")


def fit_fn(model_loglik_fn, **kwargs):
  return functools.partial(solve_newton, model_loglik_fn, **kwargs)


def fit_and_cov(model_loglik_fn, **kwargs):

  def wrapped(initial_guess):
    solver_result = solve_newton(model_loglik_fn, initial_guess, **kwargs)
    cov_H = -np.linalg.inv(solver_result.hessian)
    return FitResult(guess=solver_result.guess,
                     covs={"cov_H": cov_H},
                     loglik=solver_result.loglik,
                     steps=solver_result.step,
                     converged=solver_result.converged)

  return wrapped


if __name__ == "__main__":
  import jax.random as jrandom
  from jax import jit
  import varderiv.equations.eq1 as eq1
  import varderiv.data as data

  N = 500
  K = 3
  X_DIM = 3
  k1, k2 = jrandom.split(jrandom.PRNGKey(0))
  group_sizes = data.group_sizes_generator(N, K, "same")
  T, X, delta, beta, group_labels = data.data_generator(N,
                                                        X_DIM,
                                                        group_sizes,
                                                        return_T=True)(k1)
  X_groups, delta_groups = data.group_data_by_labels(1, K, X, delta,
                                                     group_labels)

  eq1_fit_and_cov = jit(
      fit_and_cov(functools.partial(eq1.eq1_log_likelihood, X, delta),
                  max_num_steps=10))

  print(eq1_fit_and_cov(np.ones_like(beta)))