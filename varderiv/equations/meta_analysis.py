"""Meta Analysis."""

import collections
import functools

import jax.numpy as np
from jax import random as jrandom

from varderiv.equations.eq1 import eq1_compute_H_ad
from varderiv.equations.eq2 import solve_grouped_eq_batch

from varderiv.data import group_data_by_labels

# pylint: disable=redefined-outer-name

# A dummy solver state to make API consistent
MetaAnalysisSolverState = collections.namedtuple("MetaAnalysisSolverState",
                                                 "guess value step")


@functools.lru_cache(maxsize=None)
def get_meta_analysis_rest_solver(eq1_compute_H_fn=eq1_compute_H_ad,
                                  slice_X_DIMs=None):
  """HOF for getting Meta analysis's solve rest function."""

  def _meta_analysis_solve_rest(X, delta, K, group_labels, X_groups,
                                delta_groups, beta_k_hat, beta_guess):
    """Function used by `solve_grouped_eq_batch`,
    customized for Meta Analysis."""
    del K, X, delta, group_labels, beta_guess

    if slice_X_DIMs is None:
      signature = f"(K,N,p),(K,N),(K,p)->(p),(p),()"
    else:
      signature = f"(K,N,p),(K,N),(K,p)->(l),(l),()"

    @functools.partial(np.vectorize, signature=signature)
    def _solve(X_groups, delta_groups, beta_k_hat):
      I_diag_wo_last = -eq1_compute_H_fn(X_groups, delta_groups, beta_k_hat)
      if slice_X_DIMs is not None:
        I_diag_wo_last = np.take(np.take(I_diag_wo_last, slice_X_DIMs, axis=1),
                                 slice_X_DIMs,
                                 axis=2)
        beta_k_hat = np.take(beta_k_hat, slice_X_DIMs, axis=-1)
      beta_hat = np.linalg.solve(
          np.sum(I_diag_wo_last, axis=0),
          np.einsum("Kab,Kb->a", I_diag_wo_last, beta_k_hat,
                    optimize='optimal'))
      return MetaAnalysisSolverState(guess=beta_hat,
                                     value=np.zeros_like(beta_hat),
                                     step=0)

    return _solve(X_groups, delta_groups, beta_k_hat)

  return _meta_analysis_solve_rest


meta_analysis_solve_rest = get_meta_analysis_rest_solver()
solve_meta_analysis = functools.partial(solve_grouped_eq_batch,
                                        solve_rest_fn=meta_analysis_solve_rest)

#########################################################
# BEGIN Meta Analysis cov
#########################################################


@functools.lru_cache(maxsize=None)
def get_cov_meta_analysis_fn(eq1_compute_H_fn=eq1_compute_H_ad,
                             slice_X_DIMs=None):
  """HOF for covariance computation for Meta Analysis."""

  if slice_X_DIMs is None:
    signature = "(N,p),(N),(k,s,p),(k,s),(N),(k,p),(p)->(p,p)"
  else:
    signature = "(N,p),(N),(k,s,p),(k,s),(N),(k,p),(p)->(l,l)"

  @functools.partial(np.vectorize, signature=signature)
  def wrapped(X, delta, X_groups, delta_groups, group_labels, beta_k_hat, beta):
    del X, delta, group_labels, beta
    I_diag_wo_last = -eq1_compute_H_fn(X_groups, delta_groups, beta_k_hat)
    if slice_X_DIMs is not None:
      I_diag_wo_last = np.take(np.take(I_diag_wo_last, slice_X_DIMs, axis=1),
                               slice_X_DIMs,
                               axis=2)
    return np.linalg.inv(np.sum(I_diag_wo_last, axis=0))

  return wrapped


# Default
meta_analysis_cov = get_cov_meta_analysis_fn(eq1_compute_H_fn=eq1_compute_H_ad)

#########################################################
# END
#########################################################

if __name__ == '__main__':
  N = 1000
  K = 3
  X_DIM = 4
  from varderiv.data import data_generator, group_sizes_generator
  k1, k2 = jrandom.split(jrandom.PRNGKey(0))
  group_sizes = group_sizes_generator(N, K, "same")
  T, X, delta, beta, group_labels = data_generator(N,
                                                   X_DIM,
                                                   group_sizes,
                                                   return_T=True)(k1)
  X_groups, delta_groups = group_data_by_labels(1, K, X, delta, group_labels)
  sol_pt1, sol_pt2 = solve_meta_analysis(X,
                                         delta,
                                         K,
                                         group_labels,
                                         initial_guess=beta,
                                         log=True)
  beta_k_hat = sol_pt1.guess
  beta_hat = sol_pt2.guess
  cov = meta_analysis_cov(X, delta, X_groups, delta_groups, group_labels,
                          beta_k_hat, beta_hat)
