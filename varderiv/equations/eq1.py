"""Equation 1."""

import functools

import jax.numpy as np
# from jax import jacfwd
from jax import jacrev
from jax import hessian
# from jax import random as jrandom

from varderiv.solver import solve_newton
from varderiv.generic.model_solve import sum_log_likelihood
#########################################################
# BEGIN eq1
#########################################################


@functools.partial(np.vectorize, signature='(N,p),(N),(p)->(N)')
def batch_eq1_log_likelihood(X, delta, beta):
  bx = np.dot(X, beta)
  ebx_cs = np.cumsum(np.exp(bx), 0)
  log_term = np.log(ebx_cs)
  return (bx - log_term) * delta


eq1_log_likelihood = np.vectorize(sum_log_likelihood(batch_eq1_log_likelihood),
                                  signature='(N,p),(N),(p)->()')


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
