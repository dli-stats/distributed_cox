"""Equation 1."""

import functools

import jax.numpy as np
# from jax import jacfwd
from jax import jacrev
from jax import hessian
# from jax import random as jrandom

from varderiv.solver import solve_newton

#########################################################
# BEGIN eq1
#########################################################


@functools.partial(np.vectorize, signature='(N,p),(N),(p)->()')
def eq1_log_likelihood(X, delta, beta):
  bx = np.dot(X, beta)
  ebx_cs = np.cumsum(np.exp(bx), 0)
  log_term = np.log(ebx_cs)
  log_likelhihood = np.sum((bx - log_term) * delta, axis=0)
  return log_likelhihood


@functools.partial(np.vectorize, signature='(N,p),(N),(p)->(N,p)')
def eq1_compute_W_manual(X, delta, beta):
  """Computes eq1 W."""
  del delta
  e_beta_X = np.exp(np.dot(X, beta)).reshape((-1, 1))
  X_e_beta_X = X * e_beta_X

  e_beta_X_cs = np.cumsum(e_beta_X, axis=0)
  X_e_beta_X_cs = np.cumsum(X_e_beta_X, axis=0)

  W = (X - X_e_beta_X_cs / e_beta_X_cs)
  return W


@functools.partial(np.vectorize, signature='(N,p),(N),(p)->(p)')
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
  W = eq1_compute_W_manual(X, delta, beta)
  sum_inner = W * delta.reshape(-1, 1)

  ret = np.sum(sum_inner, axis=0)

  return ret


eq1_log_likelihood_grad_ad = functools.partial(
    np.vectorize, signature='(N,p),(N),(p)->(p)')(jacrev(eq1_log_likelihood, 2))


@functools.lru_cache(maxsize=None)
def get_eq1_solver(use_ad=True, solver_max_steps=10, norm_stop_thres=1e-3):
  """Returns solver for specified arguments"""
  if use_ad:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_ad
  else:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_manual

  @functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p),(p),()")
  def wrapped(X, delta, initial_guess):
    sol = solve_newton(
        functools.partial(eq1_ll_grad_fn, X, delta),
        initial_guess,
        max_num_steps=solver_max_steps,
        sym_pos=True,  # since eq1 is optimizing loglikelihood,
        # its hessian is always symmetric positive definite
        norm_stop_thres=norm_stop_thres)
    return sol

  return wrapped


solve_eq1_ad = get_eq1_solver(use_ad=True)
solve_eq1_manual = get_eq1_solver(use_ad=False)

#########################################################
# BEGIN COV
#########################################################

eq1_compute_H_ad = functools.partial(np.vectorize,
                                     signature="(N,p),(N),(p)->(p,p)")(hessian(
                                         eq1_log_likelihood, 2))


@functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p,p)")
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


@functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p,p)")
def eq1_cov_manual(X, delta, beta):
  return _eq1_cov(X, delta, beta, eq1_compute_H_manual)


@functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p,p)")
def eq1_cov_ad(X, delta, beta):
  return _eq1_cov(X, delta, beta, eq1_compute_H_ad)


def eq1_cov(X, delta, beta, use_ad=False):
  if use_ad:  # pylint: disable=no-else-return
    return eq1_cov_ad(X, delta, beta)
  else:
    return eq1_cov_manual(X, delta, beta)


@functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p,p)")
def eq1_cov_robust_ad(X, delta, beta):
  H = eq1_compute_H_ad(X, delta, beta)
  H1 = np.linalg.inv(H)
  W = eq1_compute_W_manual(X, delta, beta) * delta.reshape((-1, 1))
  J = np.einsum("bi,bj->ij", W, W, optimize='optimal')
  return H1 @ J @ H1


@functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p,p)")
def eq1_cov_robust2_ad(X, delta, beta):
  H = eq1_compute_H_ad(X, delta, beta)
  H1 = np.linalg.inv(H)
  t = eq1_log_likelihood_grad_ad(X, delta, beta)
  J = np.outer(t, t)
  return H1 @ J @ H1


@functools.partial(np.vectorize, signature="(N,p),(N),(p)->(p,p)")
def eq1_cov_robust3_ad(X, delta, beta):
  N = X.shape[0]
  H = eq1_compute_H_ad(X, delta, beta)
  H1 = np.linalg.inv(H)
  W = eq1_compute_W_manual(X, delta, beta)

  # Compute correction term
  bx = np.dot(X, beta)
  ebx = np.exp(bx)
  ebx_cs = np.cumsum(ebx, 0)
  ebx_cs_1 = 1. / ebx_cs
  frac_sum_term = np.cumsum(ebx_cs_1[::-1])[::-1].flatten()
  W = (delta - frac_sum_term * ebx / N).reshape((N, 1)) * W

  J = np.einsum("bi,bj->ij", W, W, optimize='optimal')
  return H1 @ J @ H1
