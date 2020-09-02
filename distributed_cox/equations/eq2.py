"""Equation 2."""

import functools

import jax.numpy as np

import distributed_cox.utils as vutils

from distributed_cox.generic.modeling import sum_score

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2
#########################################################


def group_by(fun):
  """Simple wrapper for grouping."""

  def wrapped(X, delta, beta, group_labels, X_groups, delta_groups, beta_k_hat):
    K, group_size, _ = X_groups.shape
    batch_score = fun(X, delta, beta, group_labels, X_groups, delta_groups,
                      beta_k_hat)
    return vutils.group_by_labels(group_labels,
                                  batch_score,
                                  K=K,
                                  group_size=group_size)

  return wrapped


@functools.partial(
    np.vectorize,
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(N,p),(N,1)")
def _ungroupped_batch_eq2_score_frac_term(X, delta, beta, group_labels,
                                          X_groups, delta_groups, beta_k_hat):
  """Common function for computing the numerator and denomator used in eq2."""
  del delta, X_groups, delta_groups
  beta_k_hat_grouped = np.take(beta_k_hat, group_labels, axis=0)
  bmb = beta - beta_k_hat_grouped

  bkx = np.einsum("ni,ni->n", X, beta_k_hat_grouped, optimize='optimal')
  ebkx = np.exp(bkx)
  xbmb = np.einsum("np,np->n", X, bmb, optimize="optimal")  # X @ (b - bk)

  denom = (ebkx * (1. + xbmb)).reshape((-1, 1))
  numer = X * denom
  return numer, denom


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(N,p)")
def ungroupped_batch_eq2_score(X, delta, beta, group_labels, X_groups,
                               delta_groups, beta_k_hat):
  numer, denom = _ungroupped_batch_eq2_score_frac_term(X, delta, beta,
                                                       group_labels, X_groups,
                                                       delta_groups, beta_k_hat)
  frac = np.cumsum(numer, axis=0) / np.cumsum(denom, axis=0)
  ret = (X - frac) * delta.reshape((-1, 1))
  return ret


batch_eq2_score = np.vectorize(
    group_by(ungroupped_batch_eq2_score),
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S,p)")

eq2_score = np.vectorize(sum_score(ungroupped_batch_eq2_score),
                         signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(p)")


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(N,p)")
def ungroupped_batch_eq2_robust_cox_correction_score(X, delta, beta,
                                                     group_labels, X_groups,
                                                     delta_groups, beta_k_hat):
  """Cox model specific correction for eq2."""

  def _right_cumsum(X, axis=0):
    return np.cumsum(X[::-1], axis=axis)[::-1]

  numer, denom = _ungroupped_batch_eq2_score_frac_term(X, delta, beta,
                                                       group_labels, X_groups,
                                                       delta_groups, beta_k_hat)
  numer_cs = np.cumsum(numer, axis=0)
  denom_cs = np.cumsum(denom, axis=0)

  taylor_ebx_cs_1 = (1. / denom_cs) * delta.reshape((-1, 1))
  term_1 = X * denom * _right_cumsum(taylor_ebx_cs_1, axis=0)
  term_2 = denom * _right_cumsum(numer_cs * (taylor_ebx_cs_1**2), axis=0)
  score_correction_term = term_1 - term_2
  batch_score = (X - numer_cs / denom_cs) * delta.reshape((-1, 1))

  return batch_score - score_correction_term


batch_eq2_robust_cox_correction_score = np.vectorize(
    group_by(ungroupped_batch_eq2_robust_cox_correction_score),
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S,p)")


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(p,p)")
def hessian_taylor2(X, delta, beta, group_labels, X_groups, delta_groups,
                    beta_k_hat):
  """Taylor Approximation to Eq1's hessian of log likelihood function."""
  score_numer, score_denom = _ungroupped_batch_eq2_score_frac_term(
      X, delta, beta, group_labels, X_groups, delta_groups, beta_k_hat)

  score_numer_cs = np.cumsum(score_numer, axis=0)
  score_denom_cs = np.cumsum(score_denom, axis=0).reshape((-1, 1, 1))

  term_1 = (np.cumsum(
      np.einsum("Ni,Nj->Nij", X, score_numer, optimize="optimal"), axis=0) /
            score_denom_cs)

  term_2 = (np.einsum(
      "Ni,Nj->Nij", score_numer_cs, score_numer_cs, optimize="optimal") /
            score_denom_cs**2)

  return np.sum((term_2 - term_1) * delta.reshape((-1, 1, 1)), axis=0)
