"""Equation 4."""

import functools

from jax import vmap
import jax.numpy as np

from varderiv.generic.model_solve import sum_score

import varderiv.equations.eq2 as eq2

#########################################################
# BEGIN eq4
#########################################################


def _batch_eq4_score(ungroupped_batch_eq2_score_fun):

  @functools.partial(np.vectorize,
                     signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S,p)")
  def wrapped(X, delta, beta, group_labels, X_groups, delta_groups, beta_k_hat):
    del X, delta

    K = X_groups.shape[0]
    Nk = X_groups.shape[1]
    X_dim = X_groups.shape[2]

    group_labels = np.zeros((Nk,), dtype=group_labels.dtype)

    fun = vmap(ungroupped_batch_eq2_score_fun,
               in_axes=(0, 0, None, None, 0, 0, 0))
    # yapf: disable
    ret = fun(X_groups, delta_groups, beta, group_labels,
              X_groups.reshape((K, 1, Nk, X_dim)),
              delta_groups.reshape((K, 1, Nk)),
              beta_k_hat.reshape((K, 1, X_dim)))
    # yapf: enable
    return ret

  return wrapped


batch_eq4_score = _batch_eq4_score(eq2.ungroupped_batch_eq2_score)

batch_eq4_robust_cox_correction_score = _batch_eq4_score(
    eq2.ungroupped_batch_eq2_robust_cox_correction_score)

eq4_score = np.vectorize(sum_score(batch_eq4_score),
                         signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(p)")
