"""Equation 4."""

#########################################################
# BEGIN eq4
#########################################################

import functools

import jax.numpy as np
from jax.experimental.vectorize import vectorize
from jax import jacfwd

from varderiv.solver import solve_newton

from varderiv.equations.eq1 import eq1_compute_H_ad
from varderiv.equations.eq2 import _get_cov_beta_k_correction_fn
from varderiv.equations.eq2 import _solve_grouped_eq_batch
from varderiv.equations.eq2 import eq2_jac_manual


@vectorize("(K,S,p),(K,S),(K,p),(p)->(p)")
def eq4(X_groups, delta_groups, beta_k_hat, beta):
  """Equation 4's ll_grad function."""
  K = X_groups.shape[0]
  S = X_groups.shape[1]
  X_dim = X_groups.shape[2]

  beta_k_hat_grouped = np.transpose(np.broadcast_to(beta_k_hat, (S, K, X_dim)),
                                    (1, 0, 2))

  # All groups are now separarete, then group_labels is 0 for each sample
  group_labels = np.zeros(delta_groups.shape, dtype=np.uint32)

  ret = np.sum(eq2_jac_manual(X_groups, delta_groups, group_labels,
                              beta_k_hat_grouped, beta),
               axis=0)

  return ret


def _eq4_solve_rest(key, X, delta, K, group_labels, X_groups, delta_groups,
                    beta_k_hat, beta_guess):
  """Function used by `_solve_grouped_eq_batch`, customized for Eq 4."""
  del K, X, delta, group_labels

  @vectorize(f"(k),(K,N,p),(K,N),(K,p),(p)->(p)")
  def _solve(key, X_groups, delta_groups, beta_k_hat, beta_guess):
    sol = solve_newton(
        functools.partial(eq4, X_groups, delta_groups, beta_k_hat), key,
        beta_guess)
    return sol

  return _solve(key, X_groups, delta_groups, beta_k_hat, beta_guess)


solve_eq4 = functools.partial(_solve_grouped_eq_batch,
                              solve_rest_fn=_eq4_solve_rest)

#########################################################
# BEGIN eq4 cov
#########################################################

# Take gradient with respect to beta and beta_k_hat
eq4_compute_I_row = jacfwd(eq4, (-2, -1))


def eq4_compute_I_row_wrapped(X, delta, X_groups, delta_groups, group_labels,
                              beta_k_hat, beta):
  del X, delta, group_labels
  return eq4_compute_I_row(X_groups, delta_groups, beta_k_hat, beta)


get_eq4_cov_beta_k_correction_fn = functools.partial(
    _get_cov_beta_k_correction_fn, eq4_compute_I_row_wrapped)

# Default
eq4_cov = get_eq4_cov_beta_k_correction_fn(eq1_compute_H_fn=eq1_compute_H_ad)
