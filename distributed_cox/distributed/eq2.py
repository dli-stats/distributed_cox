"""Distributed implementation of Equation 2."""

from typing import Sequence

import functools
import collections

import numpy as onp
import opt_einsum as oe

import jax.numpy as np
from jax import jacfwd, vmap
from jax import random as jrandom
import jax.tree_util as tu

from distributed_cox.generic.modeling import solve_single, model_temporaries
import distributed_cox.generic.modeling as modeling

from distributed_cox.cox import eq1_loglik

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2 distributed ver.
#########################################################

DistributedEq2LocalResult = collections.namedtuple(
    "DistributedEq2LocalResult", "eq1_H X_delta_sum nxebkx_cs_ds beta_k_hat")


def distributed_compute_eq2_local(T_group,
                                  X_group,
                                  delta_group,
                                  T_delta,
                                  initial_guess=None,
                                  taylor_order=1,
                                  solver_kwargs={}
                                 ) -> DistributedEq2LocalResult:
  """Compute local values.

  Args:
    - T_group: array of shape (N, )
    - X_group: array of shape (N, X_DIM)
    - delta_group: array of shape (N, )
    - T_delta: array of shape (D, )
  """
  assert T_group.shape[0] == X_group.shape[0] == delta_group.shape[0]
  assert taylor_order >= 1

  if initial_guess is None:
    initial_guess = np.zeros((X_group.shape[1],))

  solve_eq1_fn = modeling.solve_single(eq1_loglik,
                                       use_likelihood=True,
                                       **solver_kwargs)
  eq1_sol = solve_eq1_fn(X_group, delta_group, initial_guess)
  beta_k_hat = eq1_sol.guess
  eq1_H = eq1_sol.hessian

  ebkx = np.exp(np.dot(X_group, beta_k_hat))
  nxebkxs = [ebkx]
  cur_ebkx = ebkx

  batch_dim_name = oe.get_symbol(0)
  x_dim_name = oe.get_symbol(1)
  o_dim_names = ""
  for i in range(1, taylor_order + 2):
    cur_ebkx = np.einsum("{B}{o},{B}{x}->{B}{o}{x}".format(B=batch_dim_name,
                                                           x=x_dim_name,
                                                           o=o_dim_names),
                         cur_ebkx,
                         X_group,
                         optimize='optimal')
    o_dim_names += oe.get_symbol(i + 1)
    nxebkxs.append(cur_ebkx)

  nxebkx_css = [np.cumsum(nxebkx, 0) for nxebkx in nxebkxs]

  idxs = np.searchsorted(-T_group, -T_delta, side='right')

  nxebkx_cs_ds = [np.take(nxebkx_cs, idxs, axis=0) for nxebkx_cs in nxebkx_css]
  X_delta_sum = np.sum(X_group * delta_group.reshape((-1, 1)), axis=0)

  return DistributedEq2LocalResult(eq1_H, X_delta_sum, nxebkx_cs_ds, beta_k_hat)


mark, collect = modeling.model_temporaries("distributed_eq2")


def distributed_eq2_model(X_delta_sum, nxebkx_cs_ds, beta_k_hat, beta):
  K, D, X_dim = nxebkx_cs_ds[1].shape

  bmb = mark(beta - beta_k_hat, "bmb")
  numer = np.zeros((D, X_dim), dtype=beta.dtype)
  denom = np.zeros(D, dtype=beta.dtype)
  dbeta_pow = np.ones((K, 1), dtype=beta.dtype)
  taylor_order = len(nxebkx_cs_ds) - 2
  fact = 1.
  for i in range(taylor_order + 1):
    denom += np.einsum("kdp,kp->d", nxebkx_cs_ds[i].reshape(
        (K, D, -1)), dbeta_pow) / fact
    numer += np.einsum("kdip,kp->di", nxebkx_cs_ds[i + 1].reshape(
        (K, D, X_dim, -1)), dbeta_pow) / fact
    dbeta_pow = np.einsum("kp,ki->kpi", dbeta_pow, bmb).reshape((K, -1))
    fact *= (i + 1)

  mark(dbeta_pow, "dbeta_pow_order")  # (b - b0)^order
  numer = mark(numer, "numer")
  denom = mark(denom, "denom")

  return np.sum(X_delta_sum, axis=0) - np.sum(numer / denom.reshape((-1, 1)),
                                              axis=0)


# def distributed_eq2_jac_master(X_delta_sum, ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
#                                xxxebkx_cs_d, beta_k_hat, beta):
#   del xxxebkx_cs_d
#   bmb = beta - beta_k_hat
#   num, denom = distributed_eq2_precompute(ebkx_cs_d, xebkx_cs_d, xxebkx_cs_d,
#                                           bmb)
#   denom = denom.reshape((-1, 1))
#   return np.sum(X_delta_sum, axis=0) - np.sum(num / denom, axis=0)

distributed_eq2_hess_master = jacfwd(distributed_eq2_model, -1)


def distributed_eq2_grad_beta_k_master(X_delta_sum, nxebkx_cs_ds, beta_k_hat,
                                       beta):
  K, D, X_dim = nxebkx_cs_ds[1].shape
  order = len(nxebkx_cs_ds) - 2

  t1xebkx_cs_ds = nxebkx_cs_ds[order + 1]  # order + 1 number of Xs times ebx
  t2xebkx_cs_ds = nxebkx_cs_ds[order + 2]  # order + 2 number of Xs times ebx

  bmb, numer, denom, dbeta_pow_t = collect(
      distributed_eq2_model,
      ["bmb", "number", "denum", "dbeta_pow_order"])(X_delta_sum, nxebkx_cs_ds,
                                                     beta_k_hat, beta)
  t1ebkxbmb_cs_d = np.einsum("kdip,kp->kdp",
                             t1xebkx_cs_ds.reshape((K, D, X_dim, -1)),
                             dbeta_pow_t)
  t2ebkxbmb_cs_d = np.einsum("kdijp,kp->kdij",
                             t2xebkx_cs_ds.reshape((K, D, X_dim, X_dim, -1)),
                             dbeta_pow_t)

  denom = denom.reshape((1, len(denom), 1, 1))

  return np.sum((np.einsum("dm,kdn->kdmn", numer, t1ebkxbmb_cs_d) / denom**2 -
                 t2ebkxbmb_cs_d / denom),
                axis=1)


def distributed_compute_eq2_master(
    local_data: Sequence[DistributedEq2LocalResult]):
  """Compute master values from local results."""

  eq1_H, X_delta_sum, nxebkx_cs_ds, beta_k_hat = tu.tree_multimap(
      lambda *args: np.array(args), *local_data)

  beta_guess = np.mean(beta_k_hat, axis=0)

  sol = modeling.solve_single(functools.partial(distributed_eq2_model,
                                                X_delta_sum, nxebkx_cs_ds,
                                                beta_k_hat),
                              use_likelihood=False)(beta_guess)
  beta_hat = sol.guess

  cov_H = modeling.cov_H()(sol)
  return beta_hat, cov_H


#########################################################
# END
#########################################################

if __name__ == '__main__':
  N = 1000
  K = 3
  X_DIM = 3
  import distributed_cox.data as vdata
  key, subkey = jrandom.split(vdata.key)
  group_sizes = vdata.group_sizes_generator(N, K, "same")
  T, X, delta, beta, group_labels = vdata.data_generator(
      N, X_DIM, group_sizes, return_T=True)(vdata.data_generation_key)

  T_delta = T[delta == 1]

  local_data: Sequence[DistributedEq2LocalResult] = []
  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    local_data.append(
        distributed_compute_eq2_local(T_group, X_group, delta_group, T_delta))

  beta_hat, cov_H = distributed_compute_eq2_master(local_data)
