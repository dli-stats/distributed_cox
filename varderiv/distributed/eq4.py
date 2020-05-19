"""Distributed implementation of Equation 4."""

import functools

import numpy as onp

import jax.numpy as np
from jax import jacfwd
from jax import random as jrandom

from varderiv.solver import solve_newton

from varderiv.equations.eq1 import solve_eq1_ad
from varderiv.equations.eq1 import eq1_compute_H_ad
from varderiv.equations.eq2 import cov_pure_analytical_from_I
from varderiv.distributed.eq2 import distributed_compute_eq2_local

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2 distributed ver.
#########################################################


def distributed_compute_eq4_local(T_group,
                                  X_group,
                                  delta_group,
                                  initial_guess=None,
                                  solve_eq1_fn=solve_eq1_ad,
                                  eq1_compute_H_fn=eq1_compute_H_ad):
  """Compute local values.

  Args:
    - T_group: array of shape (N, )
    - X_group: array of shape (N, X_DIM)
    - delta_group: array of shape (N, )

  Returns:
    array of shape (D, ), where D is #(delta_group == 1)
  """
  T_delta = T_group[delta_group == 1]
  return distributed_compute_eq2_local(T_group,
                                       X_group,
                                       delta_group,
                                       T_delta,
                                       initial_guess=initial_guess,
                                       solve_eq1_fn=solve_eq1_fn,
                                       eq1_compute_H_fn=eq1_compute_H_fn)


def distributed_eq4_precompute_1_group(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d, bmb):
  xebkxbmb_cs_d = np.einsum("dp,p->d", xebkx_cs_d, bmb)
  denom = ebkx_cs_d + xebkxbmb_cs_d
  xxebkxbmb_cs_d = np.einsum("dij,j->di", xxebkx_cs_d, bmb)
  num = xebkx_cs_d + xxebkxbmb_cs_d
  return num, denom


def distributed_eq4_jac_master(X_delta_sum, ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                               xxxebkx_cs_d, beta_k_hat, beta):
  del xxxebkx_cs_d
  K = X_delta_sum.shape[0]
  nums, denoms = [], []
  for k in range(K):
    bmb = beta - beta_k_hat[k]
    num, denom = distributed_eq4_precompute_1_group(ebkx_cs_d[k], xebkx_cs_d[k],
                                                    xxebkx_cs_d[k], bmb)
    denom = denom.reshape((-1, 1))
    nums.append(num)
    denoms.append(denom)

  num = np.concatenate(nums)
  denom = np.concatenate(denoms)
  return np.sum(X_delta_sum, axis=0) - np.sum(num / denom, axis=0)


distributed_eq4_hess_master = jacfwd(distributed_eq4_jac_master, -1)


def distributed_eq4_grad_beta_k_master_1_group(ebkx_cs_d, xebkx_cs_d,
                                               xxebkx_cs_d, xxxebkx_cs_d,
                                               beta_k_hat, beta):
  """Compute gradient of jac relative to beta_k."""
  bmb = beta - beta_k_hat
  xxebkxbmb_cs_d = np.einsum("dij,j->di", xxebkx_cs_d, bmb)
  xxxebkxbmb_cs_d = np.einsum("dnmp,p->dnm", xxxebkx_cs_d, bmb)

  num, denom = distributed_eq4_precompute_1_group(ebkx_cs_d, xebkx_cs_d,
                                                  xxebkx_cs_d, bmb)

  denom = denom.reshape((len(denom), 1, 1))

  return np.sum((np.einsum("dm,dn->dmn", num, xxebkxbmb_cs_d) / denom**2 -
                 xxxebkxbmb_cs_d / denom),
                axis=0)


def distributed_eq4_grad_beta_k_master(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                                       xxxebkx_cs_d, beta_k_hat, beta):
  K = beta_k_hat.shape[0]
  ret = []
  for k in range(K):
    ret.append(
        distributed_eq4_grad_beta_k_master_1_group(ebkx_cs_d[k], xebkx_cs_d[k],
                                                   xxebkx_cs_d[k],
                                                   xxxebkx_cs_d[k],
                                                   beta_k_hat[k], beta))
  return np.stack(ret)


def distributed_compute_eq4_master(eq1_H, X_delta_sum, ebkx_cs_d, xebkx_cs_d,
                                   xxebkx_cs_d, xxxebkx_cs_d, beta_k_hat):
  """Compute master values from local results."""
  beta_guess = np.mean(beta_k_hat, axis=0)
  sol = solve_newton(functools.partial(distributed_eq4_jac_master, X_delta_sum,
                                       ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
                                       xxxebkx_cs_d, beta_k_hat),
                     beta_guess,
                     max_num_steps=30)
  beta_hat = sol.guess

  I_diag_wo_last = -eq1_H
  # pylint: disable=invalid-unary-operand-type
  I_row = -distributed_eq4_grad_beta_k_master(
      ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d, xxxebkx_cs_d, beta_k_hat, beta_hat)
  I_row = np.swapaxes(I_row, 0, 1)

  I_diag_last = -distributed_eq4_hess_master(
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
  from varderiv.data import (data_generator, key, data_generation_key,
                             group_sizes_generator)
  key, subkey = jrandom.split(key)
  group_sizes = group_sizes_generator(N, K, "same")
  T, X, delta, beta, group_labels = data_generator(
      N, X_DIM, group_sizes, return_T=True)(data_generation_key)

  local_data = []
  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    local_data.append(
        distributed_compute_eq4_local(key, T_group, X_group, delta_group))

  def list_or_array(data):
    shape = data[0].shape
    for d in data:
      if not d.shape == shape:
        return list(data)
    return onp.stack(data)

  local_data = tuple(
      list_or_array([ld[d]
                     for ld in local_data])
      for d, _ in enumerate(local_data[0]))

  beta_k_hat = local_data[-1]
  # print(beta_k_hat)
  from varderiv.equations.eq1 import eq1_log_likelihood_grad_ad
  print(eq1_log_likelihood_grad_ad(X, delta, beta_k_hat))
  print(distributed_eq4_jac_master(*local_data[1:], np.mean(beta_k_hat,
                                                            axis=0)))

  beta_hat, beta_corrected_cov = distributed_compute_eq4_master(*local_data)
