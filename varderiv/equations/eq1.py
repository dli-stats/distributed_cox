"""Equation 1."""

import functools

import jax.numpy as np
from jax.experimental.vectorize import vectorize
from jax import jacfwd, jacrev, hessian
from jax import random as jrandom

from varderiv.solver import solve_newton

#########################################################
# BEGIN eq1
#########################################################


@vectorize('(N,p),(N),(p)->(p)')
def eq1_log_likelihood(X, delta, beta):
  bx = np.dot(X, beta)
  ebx_cs = np.cumsum(np.exp(bx), 0)
  log_term = np.log(ebx_cs)
  log_likelhihood = np.sum((bx - log_term) * delta, axis=0)
  return log_likelhihood


@vectorize('(N,p),(N),(p)->(p)')
def eq1_log_likelihood_grad_manual(X, delta, beta):
  """Computes eq1.

  Args:
    - X: N by DIM_X matrix representing features for each sample.
        sorted by T.
    - delta: N vector representing a mask (corresponding to each sample in X)
        for highlighted samples.
    - beta: DIM_X vector

  Returns:
    Evaluation of LHS of Eq 1.
  """
  e_beta_X = np.exp(np.dot(X, beta)).reshape((-1, 1))
  X_e_beta_X = X * e_beta_X

  e_beta_X_cs = np.cumsum(e_beta_X, axis=0)
  X_e_beta_X_cs = np.cumsum(X_e_beta_X, axis=0)

  sum_inner = (X - X_e_beta_X_cs / e_beta_X_cs) * delta.reshape(-1, 1)

  ret = np.sum(sum_inner, axis=0)

  return ret


eq1_log_likelihood_grad_ad = vectorize('(N,p),(N),(p)->(p)')(jacrev(
    eq1_log_likelihood, 2))


@functools.lru_cache(maxsize=None)
def get_eq1_solver(use_ad=True, solver_max_steps=10):
  """Returns solver for specified arguments"""
  if use_ad:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_ad
  else:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_manual

  @vectorize("(k),(N,p),(N),(p)->(p)")
  def wrapped(X, delta, initial_guess):
    sol = solve_newton(
        functools.partial(eq1_ll_grad_fn, X, delta),
        initial_guess,
        max_num_steps=solver_max_steps,
        sym_pos=True  # since eq1 is optimizing loglikelihood,
        # its hessian is always symmetric positive definite
    )
    return sol

  return wrapped


solve_eq1_ad = get_eq1_solver(use_ad=True)
solve_eq1_manual = get_eq1_solver(use_ad=False)

#########################################################
# BEGIN COV
#########################################################

eq1_compute_H_ad = vectorize("(N,p),(N),(p)->(p,p)")(hessian(
    eq1_log_likelihood, 2))


@vectorize("(N,p),(N),(p)->(p,p)")
def eq1_compute_H_manual(X, delta, beta):
  """Eq1 Hessian manual."""
  e_beta_X = np.exp(np.dot(X, beta)).reshape((-1, 1))
  X_e_beta_X = X * e_beta_X
  e_beta_X_cs = np.cumsum(e_beta_X, axis=0)
  X_e_beta_X_cs = np.cumsum(X_e_beta_X, axis=0)
  frac_term_inner = X_e_beta_X_cs / e_beta_X_cs

  X_sub_frac_term = X - frac_term_inner
  X_sub_frac_sq = np.einsum("bi,bj->bij", X_sub_frac_term, X_sub_frac_term)

  e_beta_X_mul_X_sub_frac_sq = e_beta_X.reshape((-1, 1, 1)) * X_sub_frac_sq

  num_outer = np.cumsum(e_beta_X_mul_X_sub_frac_sq, axis=0)
  denom_outer = e_beta_X_cs.reshape((-1, 1, 1))

  frac_term_outer = (num_outer / denom_outer) * delta.reshape(-1, 1, 1)

  return -np.sum(frac_term_outer, axis=0)


def eq1_compute_H(X, delta, beta, use_ad=False):
  if use_ad:
    return eq1_compute_H_ad(X, delta, beta)
  else:
    return eq1_compute_H_manual(X, delta, beta)


def _eq1_cov(X, delta, beta, eq1_H_fn):
  return np.linalg.inv(-eq1_H_fn(X, delta, beta))


@vectorize("(N,p),(N),(p)->(p,p)")
def eq1_cov_manual(X, delta, beta):
  return _eq1_cov(X, delta, beta, eq1_compute_H_manual)


@vectorize("(N,p),(N),(p)->(p,p)")
def eq1_cov_ad(X, delta, beta):
  return _eq1_cov(X, delta, beta, eq1_compute_H_ad)


def eq1_cov(X, delta, beta, use_ad=False):
  if use_ad:  # pylint: disable=no-else-return
    return eq1_cov_ad(X, delta, beta)
  else:
    return eq1_cov_manual(X, delta, beta)
