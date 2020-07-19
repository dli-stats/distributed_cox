"""Equation 2."""

import functools

import jax.numpy as np

import varderiv.data as data
from varderiv.generic.model_solve import sum_score

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2
#########################################################


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(N,p)")
def ungroupped_batch_eq2_score(X, delta, beta, group_labels, X_groups,
                               delta_groups, beta_k_hat):
  del X_groups, delta_groups
  beta_k_hat_grouped = np.take(beta_k_hat, group_labels, axis=0)
  bmb = beta - beta_k_hat_grouped

  bkx = np.einsum("ni,ni->n", X, beta_k_hat_grouped, optimize='optimal')
  ebkx = np.exp(bkx)
  xbmb = np.einsum("np,np->n", X, bmb, optimize="optimal")  # X @ (b - bk)

  denom = (ebkx * (1. + xbmb)).reshape((-1, 1))
  numer = X * denom
  frac = np.cumsum(numer, axis=0) / np.cumsum(denom, axis=0)
  ret = (X - frac) * delta.reshape((-1, 1))
  return ret


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S,p)")
def batch_eq2_score(X, delta, beta, group_labels, X_groups, delta_groups,
                    beta_k_hat):
  K, group_size, _ = X_groups.shape
  batch_score = ungroupped_batch_eq2_score(X, delta, beta, group_labels,
                                           X_groups, delta_groups, beta_k_hat)
  return data.group_by_labels(group_labels,
                              batch_score,
                              K=K,
                              group_size=group_size)


eq2_score = np.vectorize(sum_score(ungroupped_batch_eq2_score),
                         signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(p)")
