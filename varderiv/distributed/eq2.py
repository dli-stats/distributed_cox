"""Distributed implementation of Equation 2."""

import functools

import numpy as onp

import jax.numpy as np
from jax import jacfwd
from jax import random as jrandom

from varderiv.solver import solve_newton

from varderiv.equations.eq1 import solve_eq1_ad
from varderiv.equations.eq1 import eq1_compute_H_ad
from varderiv.equations.eq2 import cov_pure_analytical_from_I
# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2 distributed ver.
#########################################################


def distributed_compute_eq2_local(key,
                                  T_group,
                                  X_group,
                                  delta_group,
                                  T_delta,
                                  initial_guess=None,
                                  solve_eq1_fn=solve_eq1_ad,
                                  eq1_compute_H_fn=eq1_compute_H_ad):
  """Compute local values.

  Args:
    - T_group: array of shape (N, )
    - X_group: array of shape (N, X_DIM)
    - delta_group: array of shape (N, )
    - T_delta: array of shape (D, )
  """
  assert T_group.shape[0] == X_group.shape[0] == delta_group.shape[0]

  X_dim = X_group.shape[-1]

  if initial_guess is None:
    initial_guess = np.abs(jrandom.normal(key, shape=(X_dim,)))

  eq1_sol = solve_eq1_fn(X_group, delta_group, initial_guess)
  beta_k_hat = eq1_sol.guess
  eq1_H = eq1_compute_H_fn(X_group, delta_group, beta_k_hat)

  ebkx = np.exp(np.dot(X_group, beta_k_hat))
  xebkx = ebkx.reshape(ebkx.shape + (-1,)) * X_group
  xxebkx = np.einsum("bi,bj->bij", xebkx, X_group)
  xxxebkx = np.einsum("bij,bk->bijk", xxebkx, X_group)

  ebkx_cs = np.cumsum(ebkx, 0)
  xebkx_cs = np.cumsum(xebkx, 0)
  xxebkx_cs = np.cumsum(xxebkx, 0)
  xxxebkx_cs = np.cumsum(xxxebkx, 0)

  idxs = onp.searchsorted(onp.array(-T_group),
                          onp.array(-T_delta),
                          side='right')
  ebkx_cs_d = np.take(ebkx_cs, idxs, axis=0)
  xebkx_cs_d = np.take(xebkx_cs, idxs, axis=0)
  xxebkx_cs_d = np.take(xxebkx_cs, idxs, axis=0)
  xxxebkx_cs_d = np.take(xxxebkx_cs, idxs, axis=0)

  X_delta_sum = np.sum(X_group * delta_group.reshape((-1, 1)), axis=0)

  return (eq1_H, X_delta_sum, ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d, xxxebkx_cs_d,
          beta_k_hat)


def distributed_eq2_precompute(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d, bmb):
  ebkx_cs_ds = np.sum(ebkx_cs_d, axis=0)
  xebkxbmb_cs_ds = np.einsum("kdp,kp->d", xebkx_cs_d, bmb)
  denom = ebkx_cs_ds + xebkxbmb_cs_ds
  xebkx_cs_ds = np.sum(xebkx_cs_d, axis=0)
  xxebkxbmb_cs_ds = np.einsum("kdij,kj->di", xxebkx_cs_d, bmb)
  num = xebkx_cs_ds + xxebkxbmb_cs_ds
  return num, denom


def distributed_eq2_jac_master(X_delta_sum, ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                               xxxebkx_cs_d, beta_k_hat, beta):
  del xxxebkx_cs_d
  bmb = beta - beta_k_hat
  num, denom = distributed_eq2_precompute(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                                          bmb)
  denom = denom.reshape((-1, 1))
  return np.sum(X_delta_sum, axis=0) - np.sum(num / denom, axis=0)


distributed_eq2_hess_master = jacfwd(distributed_eq2_jac_master, -1)


def distributed_eq2_grad_beta_k_master(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                                       xxxebkx_cs_d, beta_k_hat, beta):
  """Compute gradient of jac relative to beta_k."""
  bmb = beta - beta_k_hat
  xxebkxbmb_cs_d = np.einsum("kdij,kj->kdi", xxebkx_cs_d, bmb)
  xxxebkxbmb_cs_d = np.einsum("kdnmp,kp->kdnm", xxxebkx_cs_d, bmb)

  num, denom = distributed_eq2_precompute(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                                          bmb)
  denom = denom.reshape((1, len(denom), 1, 1))

  return np.sum((np.einsum("dm,kdn->kdmn", num, xxebkxbmb_cs_d) / denom**2 -
                 xxxebkxbmb_cs_d / denom),
                axis=1)


def distributed_compute_eq2_master(eq1_H, X_delta_sum, ebkx_cs_d, xebkx_cs_d,
                                   xxebkx_cs_d, xxxebkx_cs_d, beta_k_hat):
  """Compute master values from local results."""
  beta_guess = np.mean(beta_k_hat, axis=0)
  sol = solve_newton(functools.partial(distributed_eq2_jac_master, X_delta_sum,
                                       ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                                       xxxebkx_cs_d, beta_k_hat),
                     beta_guess,
                     max_num_steps=30)
  beta_hat = sol.guess

  I_diag_wo_last = -eq1_H
  I_row = -distributed_eq2_grad_beta_k_master(
      ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d, xxxebkx_cs_d, beta_k_hat, beta_hat)
  I_row = np.swapaxes(I_row, 0, 1)

  I_diag_last = -distributed_eq2_hess_master(
      X_delta_sum, ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d, xxxebkx_cs_d, beta_k_hat,
      beta_hat)

  beta_corrected_cov = cov_pure_analytical_from_I(I_diag_wo_last, I_diag_last,
                                                  I_row)
  return beta_hat, beta_corrected_cov


#########################################################
# END
#########################################################

if __name__ == '__main__':
  N = 1000
  K = 3
  X_DIM = 4
  from varderiv.data import data_generator, key, \
    data_generation_key, group_labels_generator
  key, subkey = jrandom.split(key)
  T, X, delta, beta = data_generator(N, X_DIM,
                                     return_T=True)(data_generation_key)

  group_labels = group_labels_generator(N, K, "same")(data_generation_key)

  T_delta = T[delta == 1]

  local_data = []
  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    local_data.append(
        distributed_compute_eq2_local(key, T_group, X_group, delta_group,
                                      T_delta))
  local_data = tuple(
      onp.stack([ld[d]
                 for ld in local_data])
      for d, _ in enumerate(local_data[0]))

  beta_k_hat = local_data[-1]
  # print(beta_k_hat)
  from varderiv.equations.eq1 import eq1_log_likelihood_grad_ad
  print(eq1_log_likelihood_grad_ad(X, delta, beta_k_hat))
  print(distributed_eq2_jac_master(*local_data[1:], beta_k_hat))

  beta_hat, beta_corrected_cov = distributed_compute_eq2_master(*local_data)
