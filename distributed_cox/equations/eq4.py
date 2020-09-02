"""Equation 4."""

from jax import vmap
import jax.numpy as np

from distributed_cox.generic.modeling import sum_score, sum_hessian

import distributed_cox.equations.eq2 as eq2

#########################################################
# BEGIN eq4
#########################################################


def _batch_from_eq2(eq2_fun):
  """Computes a function in eq 4 using a function in eq2."""

  def wrapped(X, delta, beta, group_labels, X_groups, delta_groups, beta_k_hat):
    del X, delta

    K = X_groups.shape[0]
    Nk = X_groups.shape[1]
    X_dim = X_groups.shape[2]

    group_labels = np.zeros((Nk,), dtype=group_labels.dtype)

    fun = vmap(eq2_fun, in_axes=(0, 0, None, None, 0, 0, 0))
    # yapf: disable
    ret = fun(X_groups, delta_groups, beta, group_labels,
              X_groups.reshape((K, 1, Nk, X_dim)),
              delta_groups.reshape((K, 1, Nk)),
              beta_k_hat.reshape((K, 1, X_dim)))
    # yapf: enable
    return ret

  return wrapped


batch_eq4_score = np.vectorize(
    _batch_from_eq2(eq2.ungroupped_batch_eq2_score),
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S,p)")

batch_eq4_robust_cox_correction_score = np.vectorize(
    _batch_from_eq2(eq2.ungroupped_batch_eq2_robust_cox_correction_score),
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(K,S,p)")

eq4_score = np.vectorize(sum_score(batch_eq4_score),
                         signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(p)")

hessian_taylor2 = np.vectorize(
    sum_hessian(_batch_from_eq2(eq2.hessian_taylor2)),
    signature="(N,p),(N),(p),(N),(K,S,p),(K,S),(K,p)->(p,p)")
