"""Equation 2."""

import functools

import jax.numpy as np

import varderiv.data as data
from varderiv.generic.model_solve import sum_log_likelihood

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2
#########################################################


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(N)")
def _batch_eq2_log_likelihood(X, delta, beta, group_labels, X_groups,
                              delta_groups, beta_k_hat):
  del X_groups, delta_groups
  beta_k_hat_grouped = np.take(beta_k_hat, group_labels, axis=0)
  ebkx = np.exp(np.einsum("ni,ni->n", X, beta_k_hat_grouped,
                          optimize='optimal'))
  logterm = np.log((1. - np.einsum(
      "ni,ni->n", X, beta - beta_k_hat_grouped, optimize="optimal")) * ebkx)
  bx = np.dot(X, beta)
  batch_loglik = (bx - logterm) * delta
  return batch_loglik


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S)")
def batch_eq2_log_likelihood(X, delta, beta, group_labels, X_groups,
                             delta_groups, beta_k_hat):
  K, group_size, _ = X_groups.shape
  batch_loglik = _batch_eq2_log_likelihood(X, delta, beta, group_labels,
                                           X_groups, delta_groups, beta_k_hat)
  return data.group_by_labels(group_labels,
                              batch_loglik,
                              K=K,
                              group_size=group_size)


eq2_log_likelihood = np.vectorize(
    sum_log_likelihood(_batch_eq2_log_likelihood),
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->()")
