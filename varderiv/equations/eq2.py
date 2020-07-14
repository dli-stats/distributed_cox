"""Equation 2."""

import functools

import jax.numpy as np
from jax import jacfwd
from jax import random as jrandom

from varderiv.solver import solve_newton

from varderiv.equations.eq1 import (solve_eq1_manual,
                                    eq1_log_likelihood_grad_ad)
from varderiv.equations.eq1 import eq1_compute_H_ad
from varderiv.equations.eq1 import eq1_compute_W_manual
from varderiv.data import group_data_by_labels, group_by_labels

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2
#########################################################


def eq2_log_likelihood(X, delta, beta):
  pass


precomputed_signature = "(N,c),(N,p),(N,p,p),(N,c),(N,p),(N,p)"


@functools.partial(np.vectorize,
                   signature=f"(N,p),(N),(k,p)->{precomputed_signature}")
def _precompute_eq2_terms(X, group_labels, beta_k_hat):
  """Precomputes some tensors for equation 2."""
  beta_k_hat_grouped = np.take(beta_k_hat, group_labels, axis=0)

  beta_k_hat_X = np.einsum("bi,bi->b",
                           X,
                           beta_k_hat_grouped,
                           optimize='optimal')
  e_beta_k_hat_X = np.exp(beta_k_hat_X).reshape((-1, 1))
  X_e_beta_k_hat_X = X * e_beta_k_hat_X

  XX_e_beta_k_hat_X = np.einsum('bi,bj,bk->bij',
                                X,
                                X,
                                e_beta_k_hat_X,
                                optimize='optimal')

  e_beta_k_hat_X_cs = np.cumsum(e_beta_k_hat_X, 0)
  X_e_beta_k_hat_X_cs = np.cumsum(X_e_beta_k_hat_X, 0)
  return e_beta_k_hat_X, X_e_beta_k_hat_X, XX_e_beta_k_hat_X, \
    e_beta_k_hat_X_cs, X_e_beta_k_hat_X_cs, beta_k_hat_grouped


@functools.partial(np.vectorize,
                   signature=f"(N,p),(N),{precomputed_signature},(p)->(N,p)")
def eq2_compute_W(X, delta, e_beta_k_hat_X, X_e_beta_k_hat_X, XX_e_beta_k_hat_X,
                  e_beta_k_hat_X_cs, X_e_beta_k_hat_X_cs, beta_k_hat_grouped,
                  beta):
  """Computes W matrix.

  W is a N by X_DIM matrix that is same as left side of eq2_ll_grad, without
  the final summation over all the deltas.
  """
  # TODO(camyang) better docstring, explain W
  del delta, e_beta_k_hat_X
  beta_sub_beta_k_hat = beta - beta_k_hat_grouped
  xxebxbmb = np.einsum("bij,bj->bi",
                       XX_e_beta_k_hat_X,
                       beta_sub_beta_k_hat,
                       optimize='optimal')

  xebxbmb = np.einsum("bi,bi->b",
                      X_e_beta_k_hat_X,
                      beta_sub_beta_k_hat,
                      optimize='optimal')
  xxebxbmb_cs = np.cumsum(xxebxbmb, 0)
  xebxbmb_cs = np.cumsum(xebxbmb, 0).reshape((-1, 1))

  W = X - ((X_e_beta_k_hat_X_cs + xxebxbmb_cs) /
           (e_beta_k_hat_X_cs + xebxbmb_cs))
  return W


@functools.partial(np.vectorize,
                   signature=f"(N,p),(N),{precomputed_signature},(p)->(N,p)")
def eq2_compute_eq1_W(X, delta, e_beta_k_hat_X, X_e_beta_k_hat_X,
                      XX_e_beta_k_hat_X, e_beta_k_hat_X_cs, X_e_beta_k_hat_X_cs,
                      beta_k_hat_grouped, beta):
  """Computes the each group's W by eq1.
  """
  del (delta, e_beta_k_hat_X, X_e_beta_k_hat_X, XX_e_beta_k_hat_X,
       beta_k_hat_grouped, beta)
  W = X - X_e_beta_k_hat_X_cs / e_beta_k_hat_X_cs
  return W


@functools.partial(np.vectorize,
                   signature=f"(N,p),(N),{precomputed_signature},(p)->(p)")
def eq2_rest(X, delta, e_beta_k_hat_X, X_e_beta_k_hat_X, XX_e_beta_k_hat_X,
             e_beta_k_hat_X_cs, X_e_beta_k_hat_X_cs, beta_k_hat_grouped, beta):
  W = eq2_compute_W(X, delta, e_beta_k_hat_X, X_e_beta_k_hat_X,
                    XX_e_beta_k_hat_X, e_beta_k_hat_X_cs, X_e_beta_k_hat_X_cs,
                    beta_k_hat_grouped, beta)
  return np.sum(W * delta.reshape((-1, 1)), axis=0)


@functools.partial(np.vectorize, signature="(N,p),(N),(N),(k,p),(p)->(p)")
def eq2_jac_manual(X, delta, group_labels, beta_k_hat, beta):
  precomputed = _precompute_eq2_terms(X, group_labels, beta_k_hat)
  return eq2_rest(X, delta, *precomputed, beta)


@functools.lru_cache(maxsize=None)
def get_eq2_rest_solver(solver_max_steps=10):
  """HOF for getting Eq 2's solve rest function."""

  def eq2_solve_rest(X, delta, K, group_labels, X_groups, delta_groups,
                     beta_k_hat, beta_guess):
    """Function used by `solve_grouped_eq_batch`, customized for Eq 2."""

    del K, X_groups, delta_groups

    precomputed = _precompute_eq2_terms(X, group_labels, beta_k_hat)

    @functools.partial(
        np.vectorize,
        signature=f"(N,p),(N),{precomputed_signature},(p)->(p),(p),()")
    def _solve(X, delta, e_beta_k_hat_X, X_e_beta_k_hat_X, XX_e_beta_k_hat_X,
               e_beta_k_hat_X_cs, X_e_beta_k_hat_X_cs, beta_k_hat_grouped,
               beta_guess):
      return solve_newton(functools.partial(eq2_rest, X, delta, e_beta_k_hat_X,
                                            X_e_beta_k_hat_X, XX_e_beta_k_hat_X,
                                            e_beta_k_hat_X_cs,
                                            X_e_beta_k_hat_X_cs,
                                            beta_k_hat_grouped),
                          beta_guess,
                          max_num_steps=solver_max_steps)

    return _solve(X, delta, *precomputed, beta_guess)

  return eq2_solve_rest


eq2_solve_rest = get_eq2_rest_solver()


def solve_grouped_eq_batch(  # pylint: disable=too-many-arguments
    X,
    delta,
    K,
    group_labels,
    X_groups=None,
    delta_groups=None,
    initial_guess=None,
    solve_eq1_fn=solve_eq1_manual,
    solve_rest_fn=eq2_solve_rest,
    log=False,
    log_solve_rest_name="Eq2"):
  """Common function used by Equation 2 and 4.

  This function is done in few stages:
    1. Arguments are tested to see if we are in batch mode, if not, necessary
      arguments are turned into batch of size 1.
    2. A single group_size is decided over all batch and all groups. All groups
      are padded and converted; then a single solve_eq_fn is invoked to solve
      for all groups at the same time.
    3. solve_rest is invoked across batch and result is returned.

  Note that this function DOES NOT perform any jitting. For one, groupping data
  cannot be jitt'ed currently in jax. User is responsible for passing in jitt'ed
  versions of `solve_eq1_fn` and `solve_rest_fn` if desirable.

  Returns:
    beta_k_hat: solved K groups of beta_k_hat,
    beta: final solution beta
  """
  if len(X.shape) == 2:
    # We are not in batch mode
    assert len(delta.shape) == 1
    assert len(group_labels.shape) == 1
    X = X.reshape((1,) + X.shape)
    delta = delta.reshape((1,) + delta.shape)
    group_labels = group_labels.reshape((1,) + group_labels.shape)
    initial_guess = initial_guess.reshape((1,) + initial_guess.shape)
  assert len(X.shape) == 3

  batch_size = X.shape[0]
  X_dim = X.shape[-1]

  if initial_guess is None:
    initial_guess = np.zeros((batch_size, X_dim))

  assert initial_guess.shape == (batch_size, X_dim)

  step_1_initial_guess = np.broadcast_to(initial_guess, (K, batch_size, X_dim))
  step_1_initial_guess = np.transpose(step_1_initial_guess, axes=[1, 0, 2])
  assert step_1_initial_guess.shape == (batch_size, K, X_dim)

  if X_groups is None or delta_groups is None:
    X_groups, delta_groups = group_data_by_labels(batch_size, K, X, delta,
                                                  group_labels)

  group_size = delta_groups.shape[-1]
  assert X_groups.shape == (batch_size, K, group_size, X_dim)
  assert delta_groups.shape == (batch_size, K, group_size)

  eq1_sols = solve_eq1_fn(X_groups, delta_groups, step_1_initial_guess)
  if log:
    for i, sol_single_batch in enumerate(
        zip(eq1_sols.guess, eq1_sols.value, eq1_sols.step)):
      for k, (beta, value, step) in enumerate(zip(*sol_single_batch)):
        print("batch {} solved Eq1 for group {} "
              "beta={} value={} in {} steps".format(i, k, beta, value, step))

  beta_k_hat = eq1_sols.guess
  rest_sol = solve_rest_fn(X, delta, K, group_labels, X_groups, delta_groups,
                           beta_k_hat, initial_guess)

  if log:
    print("Solved {} beta={} value={} in {} steps".format(
        log_solve_rest_name, *rest_sol))

  return eq1_sols, rest_sol


solve_eq2 = functools.partial(solve_grouped_eq_batch,
                              solve_rest_fn=eq2_solve_rest)

#########################################################
# BEGIN eq2 cov
#########################################################

# Take gradient with respect to beta and beta_k_hat
eq2_compute_I_row = jacfwd(eq2_jac_manual, (-2, -1))


def eq2_compute_I_row_wrapped(X, delta, X_groups, delta_groups, group_labels,
                              beta_k_hat, beta):
  del X_groups, delta_groups
  return eq2_compute_I_row(X, delta, group_labels, beta_k_hat, beta)


@functools.partial(np.vectorize, signature="(N,p),(N),(N),(k,p),(p)->(N,p)")
def eq2_compute_pt2_W(X, delta, group_labels, beta_k_hat, beta):
  precomputed = _precompute_eq2_terms(X, group_labels, beta_k_hat)
  pt2_W = eq2_compute_W(X, delta, *precomputed, beta)
  return pt2_W


def eq2_compute_B(X, delta, X_groups, delta_groups, group_labels, beta_k_hat,
                  beta):
  N = X.shape[0]
  K = X_groups.shape[0]
  Nk = X_groups.shape[1]
  X_dim = X_groups.shape[2]  # pylint: disable=unused-variable
  pt1_W = eq1_compute_W_manual(X_groups, delta_groups,
                               beta_k_hat)  # K x Nk x X_DIM
  pt2_W = eq2_compute_pt2_W(X, delta, group_labels, beta_k_hat,
                            beta)  # N x X_DIM
  pt1_W = pt1_W * delta_groups.reshape((K, Nk, 1))
  pt2_W = pt2_W * delta.reshape((N, 1))
  B_diag_wo_last = np.einsum("kbi,kbj->kij", pt1_W, pt1_W, optimize="optimal")
  B_diag_last = np.einsum("ki,kj->ij", pt2_W, pt2_W, optimize="optimal")
  pt2_W_grouped = group_by_labels(K, Nk, pt2_W, group_labels)
  B_row_wo_last = np.einsum("kbi,kbj->kij",
                            pt2_W_grouped,
                            pt1_W,
                            optimize="optimal")
  return B_diag_wo_last, B_diag_last, B_row_wo_last


@functools.lru_cache(maxsize=None)
def get_cov_beta_k_correction_fn(compute_I_row_wrapped_fn,
                                 compute_B_fn,
                                 robust=False,
                                 eq1_ll_grad_fn=eq1_log_likelihood_grad_ad,
                                 eq1_compute_H_fn=eq1_compute_H_ad):
  """HOF for covariance computation with beta_k correction."""

  if not robust:
    del eq1_ll_grad_fn

  @functools.partial(np.vectorize,
                     signature="(N,p),(N),(k,s,p),(k,s),(N),(k,p),(p)->(p,p)")
  def wrapped(X, delta, X_groups, delta_groups, group_labels, beta_k_hat, beta):
    """Computes Eq 2 cov with beta_k correction.

    Computes hessian using AD, then adjust with beta_k variances.
    This function is not vectorized itself.

    Args:
      - compute_I_row_wrapped_fn: a *vectorized* function that computes the last
        row in I matrix.
    """

    I_diag_wo_last = -eq1_compute_H_fn(X_groups, delta_groups, beta_k_hat)  # pylint:disable=invalid-unary-operand-type

    I_row, I_diag_last = compute_I_row_wrapped_fn(X, delta, X_groups,
                                                  delta_groups, group_labels,
                                                  beta_k_hat, beta)

    I_row, I_diag_last = -I_row, -I_diag_last

    if not robust:
      cov = cov_pure_analytical_from_I(I_diag_wo_last, I_diag_last, I_row)
    else:
      B_diag_wo_last, B_diag_last, B_row_wo_last = compute_B_fn(
          X, delta, X_groups, delta_groups, group_labels, beta_k_hat, beta)

      cov = cov_pure_analytical_from_I_robust(I_diag_wo_last, I_diag_last,
                                              I_row, B_diag_wo_last,
                                              B_diag_last, B_row_wo_last)
    return cov

  return wrapped


@functools.partial(np.vectorize, signature="(k,p,p),(p,p),(p,k,p)->(p,p)")
def cov_pure_analytical_from_I(I_diag_wo_last, I_diag_last, I_row):
  """
  Args:
    - I_diag_wo_last: array of shape (K, P, P)
    - I_diag_last: array of shape (P, P)
    - I_row: array of shape (P, K, P)
  """
  I_diag_inv_last = np.linalg.inv(I_diag_last)
  I_diag_inv_wo_last = np.linalg.inv(I_diag_wo_last)

  S = np.einsum("ab,bBc->Bac", I_diag_inv_last, I_row, optimize="optimal")
  cov = np.einsum(
      "Bab,Bbc,Bdc->ad", S, I_diag_inv_wo_last, S,
      optimize='optimal') + I_diag_inv_last

  return cov


@functools.partial(
    np.vectorize,
    signature="(k,p,p),(p,p),(p,k,p),(k,p,p),(p,p),(k,p,p)->(p,p)")
def cov_pure_analytical_from_I_robust(I_diag_wo_last, I_diag_last, I_row,
                                      B_diag_wo_last, B_diag_last,
                                      B_row_wo_last):
  """Computes I^-1 B I"""
  I_diag_inv_last = np.linalg.inv(I_diag_last)
  I_diag_inv_wo_last = np.linalg.inv(I_diag_wo_last)

  S = np.einsum("ab,bBc,Bcd->Bad",
                I_diag_inv_last,
                I_row,
                I_diag_inv_wo_last,
                optimize="optimal")

  sas = np.einsum("Bab,Bbc,Bdc->ad", S, B_diag_wo_last, S, optimize="optimal")
  sb1s = np.einsum("ab,Bbc,Bdc->ad",
                   I_diag_inv_last,
                   B_row_wo_last,
                   S,
                   optimize="optimal")
  sb2s = np.einsum("Bab,Bbc,dc->ad",
                   S,
                   B_row_wo_last,
                   I_diag_inv_last,
                   optimize="optimal")
  scs = np.einsum('ab,bc,dc->ad',
                  I_diag_inv_last,
                  B_diag_last,
                  I_diag_inv_last,
                  optimize='optimal')
  cov = sas - sb1s - sb2s + scs
  return cov


get_eq2_cov_beta_k_correction_fn = functools.partial(
    get_cov_beta_k_correction_fn, eq2_compute_I_row_wrapped, eq2_compute_B)

eq2_compute_H = jacfwd(eq2_jac_manual, -1)


@functools.partial(np.vectorize,
                   signature="(N,p),(N),(N),(k,p),(p)->(p,p),(p,p)")
def eq2_cov_robust_ad_impl(X, delta, group_labels, beta_k_hat, beta):
  """Computes covariance for eq2 with AD jacobian.

  Not really robust! Name from `Robust Estimate`.
  Uses AD to computes Hessian of equation 2, then return
                `H^-1 J H^-1`
  """
  # TODO(camyang) this is not optimized

  precomputed = _precompute_eq2_terms(X, group_labels, beta_k_hat)

  H = jacfwd(eq2_rest, -1)(X, delta, *precomputed, beta)

  # compute J
  W = eq2_compute_W(X, delta, *precomputed, beta)
  W2 = np.einsum("bi,bj->bij", W, W, optimize='optimal')
  J = np.sum(W2 * delta.reshape((-1, 1, 1)), axis=0)

  H_inv = np.linalg.inv(H)
  ret = H_inv @ J @ H_inv

  return -H_inv, ret


def eq2_cov_robust_ad(X, delta, X_groups, delta_groups, group_labels,
                      beta_k_hat, beta):

  del X_groups, delta_groups

  ret = eq2_cov_robust_ad_impl(X, delta, group_labels, beta_k_hat, beta)

  return ret[1]


# Default
eq2_cov_beta_k_correction = get_eq2_cov_beta_k_correction_fn(
    robust=False, eq1_compute_H_fn=eq1_compute_H_ad)
eq2_cov_beta_k_correction_robust = get_eq2_cov_beta_k_correction_fn(
    robust=True,
    eq1_ll_grad_fn=eq1_log_likelihood_grad_ad,
    eq1_compute_H_fn=eq1_compute_H_ad)
eq2_cov = eq2_cov_beta_k_correction

#########################################################
# END
#########################################################

if __name__ == '__main__':
  N = 500
  K = 3
  X_DIM = 3
  from varderiv.data import data_generator, group_sizes_generator
  k1, k2 = jrandom.split(jrandom.PRNGKey(0))
  group_sizes = group_sizes_generator(N, K, "same")
  T, X, delta, beta, group_labels = data_generator(N,
                                                   X_DIM,
                                                   group_sizes,
                                                   return_T=True)(k1)
  X_groups, delta_groups = group_data_by_labels(1, K, X, delta, group_labels)
  sol_pt1, sol_pt2 = solve_eq2(X,
                               delta,
                               K,
                               group_labels,
                               initial_guess=beta,
                               log=True)
  beta_k_hat = sol_pt1.guess
  beta_hat = sol_pt2.guess
  cov_beta_k_correction = eq2_cov_beta_k_correction(X, delta, X_groups,
                                                    delta_groups, group_labels,
                                                    beta_k_hat, beta_hat)
