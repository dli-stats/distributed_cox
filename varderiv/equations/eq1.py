"""Equation 1."""

import functools

from jax import jacrev, jacfwd
import jax.numpy as np

from varderiv.generic.modeling import sum_log_likelihood

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

batch_eq1_score = np.vectorize(jacfwd(batch_eq1_log_likelihood, -1),
                               signature='(N,p),(N),(p)->(N,p)')


def _right_cumsum(X, axis=0):
  return np.cumsum(X[::-1], axis=axis)[::-1]


@functools.partial(np.vectorize, signature='(N,p),(N),(p)->(N,p)')
def batch_eq1_robust_cox_correction_score(X, delta, beta):
  bx = np.dot(X, beta)
  ebx = np.exp(bx).reshape((-1, 1))
  xebx = X * ebx
  ebx_cs = np.cumsum(ebx, axis=0)
  xebx_cs = np.cumsum(xebx, axis=0)

  ebx_cs_1 = (1. / ebx_cs) * delta.reshape((-1, 1))
  term_1 = X * ebx * _right_cumsum(ebx_cs_1, axis=0)
  term_2 = ebx * _right_cumsum(xebx_cs * (ebx_cs_1**2), axis=0)
  score_correction_term = term_1 - term_2
  score = batch_eq1_score(X, delta, beta)
  return score - score_correction_term
