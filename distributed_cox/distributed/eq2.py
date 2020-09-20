"""Distributed implementation of Equation 2."""

from typing import Optional, List

import functools

import opt_einsum as oe

import jax.numpy as np
from jax import jacfwd
import jax.lax as lax

import distributed_cox.generic.modeling as modeling

from distributed_cox.cox import eq1_loglik, eq1_batch_score

from distributed_cox.distributed.common import (VarianceSetting, ClientState,
                                                Message)

# pylint: disable=redefined-outer-name

#########################################################
# BEGIN eq2 distributed ver.
#########################################################


def eq2_local_send_T(local_state: ClientState) -> Message:
  T_group = local_state["T_group"]
  delta_group = local_state["delta_group"]
  return {"T_delta_per_group": T_group[delta_group]}


def eq2_master_send_T(master_state: ClientState) -> Message:
  # Pop T_delta_oper_group since it will not be numpy serializable
  T_deltas = list(master_state.pop("T_delta_per_group"))
  num_groups = len(T_deltas)

  msg = {}

  T_delta = np.concatenate(T_deltas)
  idxs = np.argsort(-T_delta)
  T_delta = T_delta[idxs]
  master_state["T_delta"] = T_delta
  T_delta_group_labels = np.concatenate(
      [np.repeat(k, len(T_deltas[k])) for k in range(num_groups)])
  T_delta_group_labels = T_delta_group_labels[idxs]
  master_state["T_delta_group_labels"] = T_delta_group_labels
  master_state["num_groups"] = num_groups
  msg["T_delta"] = T_delta

  if master_state.config.require_cox_correction:
    master_state["delta_group_labels"] = T_delta_group_labels
    msg["T_delta_group_mask"] = [
        (T_delta_group_labels == k) for k in range(num_groups)
    ]
  return msg


def eq2_local(local_state: ClientState,
              initial_guess: Optional[List[float]] = None,
              loglik_eps: float = 1e-6,
              max_num_steps: int = 10) -> Message:
  """Compute local values."""

  T_group, X_group, delta_group, T_delta = local_state.get_vars(
      "T_group",
      "X_group",
      "delta_group",
      "T_delta",
  )

  assert T_group.shape[0] == X_group.shape[0] == delta_group.shape[0]
  assert local_state.config.taylor_order >= 1

  if initial_guess is None:
    initial_guess = np.zeros((X_group.shape[1],))

  # e.g. order=1 ==> need X * X * exp(beta * X)
  required_orders = local_state.config.taylor_order + 1
  # e.g. group_correction further requires X * X * X * exp(beta * X)
  if local_state.config.require_group_correction:
    required_orders += 1

  solve_eq1_fn = modeling.solve_single(eq1_loglik,
                                       use_likelihood=True,
                                       loglik_eps=loglik_eps,
                                       max_num_steps=max_num_steps)
  eq1_sol = solve_eq1_fn(X_group, delta_group, initial_guess)
  beta_k_hat = eq1_sol.guess
  eq1_H = eq1_sol.hessian

  ebkx = np.exp(np.dot(X_group, beta_k_hat))
  nxebkxs = [ebkx]
  cur_ebkx = ebkx

  batch_dim_name = oe.get_symbol(0)
  x_dim_name = oe.get_symbol(1)
  o_dim_names = ""
  for i in range(1, required_orders + 1):
    cur_ebkx = np.einsum(
        "{B}{o},{B}{x}->{B}{o}{x}".format(B=batch_dim_name,
                                          x=x_dim_name,
                                          o=o_dim_names), cur_ebkx, X_group)
    o_dim_names += oe.get_symbol(i + 1)
    nxebkxs.append(cur_ebkx)

  if local_state.config.require_cox_correction:
    for i, x in enumerate(nxebkxs):
      local_state["nxebkxs{}".format(i)] = x

  nxebkx_css = [np.cumsum(nxebkx, 0) for nxebkx in nxebkxs]

  idxs = np.searchsorted(-T_group, -T_delta, side='right')

  nxebkx_cs_ds = {
      "nxebkx_cs_ds{}".format(i): np.take(nxebkx_cs, idxs, axis=0)
      for i, nxebkx_cs in enumerate(nxebkx_css)
  }

  X_delta_sum = np.sum(X_group * delta_group.reshape((-1, 1)), axis=0)

  if (local_state.config.require_robust or
      local_state.config.require_group_correction):
    local_state["beta_k_hat"] = beta_k_hat

  return dict(eq1_H=eq1_H,
              X_delta_sum=X_delta_sum,
              beta_k_hat=beta_k_hat,
              **nxebkx_cs_ds)


mark, collect = modeling.model_temporaries("distributed_eq2")


def _eq2_model(X_delta_sum, nxebkx_cs_ds, beta_k_hat, beta, taylor_order=1):
  K, D, X_dim = nxebkx_cs_ds[1].shape

  bmb = mark(beta - beta_k_hat, "bmb")
  numer = np.zeros((D, X_dim), dtype=beta.dtype)
  denom = np.zeros(D, dtype=beta.dtype)
  dbeta_pow = np.ones((K, 1), dtype=beta.dtype)
  fact = 1.
  for i in range(taylor_order + 1):
    denom += np.einsum("kdp,kp->d", nxebkx_cs_ds[i].reshape(
        (K, D, -1)), dbeta_pow) / fact
    numer += np.einsum("kdip,kp->di", nxebkx_cs_ds[i + 1].reshape(
        (K, D, X_dim, -1)), dbeta_pow) / fact
    if i != taylor_order:
      dbeta_pow = np.einsum("kp,ki->kpi", dbeta_pow, bmb).reshape((K, -1))
    fact *= (i + 1)

  mark(dbeta_pow, "dbeta_pow_order")  # (b - b0)^|order - 1|
  numer = mark(numer, "numer")
  denom = mark(denom, "denom")

  return np.sum(X_delta_sum, axis=0) - np.sum(numer / denom.reshape((-1, 1)),
                                              axis=0)


def _eq2_grad_beta_k_master(X_delta_sum, nxebkx_cs_ds, beta_k_hat, beta,
                            taylor_order):
  K, D, X_dim = nxebkx_cs_ds[1].shape

  t1xebkx_cs_ds = nxebkx_cs_ds[taylor_order + 1]  # order + 1 Xs times ebx
  t2xebkx_cs_ds = nxebkx_cs_ds[taylor_order + 2]  # order + 2  Xs times ebx

  numer, denom, dbeta_pow_t = collect(_eq2_model,
                                      ["numer", "denom", "dbeta_pow_order"])(
                                          X_delta_sum,
                                          nxebkx_cs_ds,
                                          beta_k_hat,
                                          beta,
                                          taylor_order=taylor_order)
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


def eq2_master(master_state: ClientState) -> Message:
  """Compute master values from local results."""

  X_delta_sum, nxebkx_cs_ds, beta_k_hat = master_state.get_vars(
      "X_delta_sum",
      "nxebkx_cs_ds*",
      "beta_k_hat",
  )

  beta_guess = np.mean(beta_k_hat, axis=0)

  fn_to_solve = functools.partial(_eq2_model,
                                  X_delta_sum,
                                  nxebkx_cs_ds,
                                  beta_k_hat,
                                  taylor_order=master_state.config.taylor_order)
  sol = modeling.solve_single(fn_to_solve, use_likelihood=False)(beta_guess)
  beta_hat = sol.guess

  cov_H = modeling.cov_H()(sol)
  master_state["beta_hat"] = beta_hat
  master_state[str(VarianceSetting(False, False, False))] = cov_H

  msg = {}

  if master_state.config.require_robust:
    numer, denom = collect(_eq2_model, ["numer", "denom"])(
        X_delta_sum,
        nxebkx_cs_ds,
        beta_k_hat,
        beta_hat,
        taylor_order=master_state.config.taylor_order)
    if not master_state.config.require_cox_correction:
      delta_group_labels = master_state["delta_group_labels"]
      num_groups = master_state["num_groups"]
      numer = [numer[delta_group_labels == k] for k in range(num_groups)]
      denom = [denom[delta_group_labels == k] for k in range(num_groups)]
      msg.update({"numer_only_group": numer, "denom_only_group": denom})
    else:
      msg.update({"numer_all_delta": numer, "denom_all_delta": denom})

  if master_state.config.require_cox_correction:
    msg["beta_hat"] = beta_hat

  return msg


def _right_cumsum(X, axis=0):
  return np.cumsum(X[::-1], axis=axis)[::-1]


def _batch_score_cox_correction(local_state: ClientState, batch_score,
                                numer_all_delta, denom_all_delta):
  beta_k_hat, beta_hat, T_group, T_delta, delta_group = local_state.get_vars(
      "beta_k_hat", "beta_hat", "T_group", "T_delta", "delta_group")
  # Obtain the denom and numers for the current group
  denom_all_group, numer_all_group = collect(_eq2_model, ["denom", "numer"])(
      np.zeros_like(beta_k_hat).reshape((1, -1)),  # argument not used
      [x.reshape((1,) + x.shape) for x in local_state.get_var("nxebkxs*")],
      beta_k_hat.reshape((1, -1)),
      beta_hat.reshape((1, -1)),
      taylor_order=local_state.config.taylor_order,
  )
  denom_all_group = denom_all_group.reshape((-1, 1))
  idxs = np.searchsorted(-T_delta, -T_group, side='right')
  denom_all_delta_1 = 1. / denom_all_delta
  term1 = numer_all_group * _right_cumsum(denom_all_delta_1, axis=0)[idxs]
  term2 = denom_all_group * _right_cumsum(
      numer_all_delta * (denom_all_delta_1**2), axis=0)[idxs]

  # Pad batch_score to entire group's size where non-marked indices are set to 0
  batch_score = np.array(batch_score)

  def insert_at(carry, delta):
    return carry + delta, batch_score[carry] * delta

  _, batch_score = lax.scan(insert_at, 0, delta_group)
  return batch_score - (term1 - term2)


def _compute_B(local_state,
               pt2_batch_score,
               cox_correction=False,
               group_correction=False,
               robust=False):
  ret = {}
  ret['B_diag_last'] = np.einsum("ni,nj->ij",
                                 pt2_batch_score,
                                 pt2_batch_score,
                                 optimize="optimal")
  if group_correction and robust:
    X_group, delta_group, beta_k_hat = local_state.get_vars(
        "X_group", "delta_group", "beta_k_hat")
    pt1_batch_score = eq1_batch_score(X_group, delta_group, beta_k_hat)
    ret['B_diag_wo_last'] = np.einsum("ni,nj->ij", pt1_batch_score,
                                      pt1_batch_score)
    if not cox_correction:
      pt1_batch_score = pt1_batch_score[delta_group]
    ret['B_row_wo_last'] = np.einsum("ni,nj->ij", pt1_batch_score,
                                     pt2_batch_score)

  if cox_correction:
    ret = {(k + "_cox_correction"): v for k, v in ret.items()}

  return ret


def eq2_local_variance(local_state: ClientState) -> Message:

  msg = {}

  X_group, delta_group = local_state.get_vars("X_group", "delta_group")
  X_delta_group = X_group[delta_group]

  if local_state.config.require_cox_correction:
    numer_all_delta, denom_all_delta, T_delta_group_mask = local_state.get_vars(
        "numer_all_delta", "denom_all_delta", "T_delta_group_mask")
    denom_all_delta = denom_all_delta.reshape((-1, 1))
    numer_only_group_delta, denom_only_group_delta = (
        numer_all_delta[T_delta_group_mask],
        denom_all_delta[T_delta_group_mask])
  else:
    numer_only_group_delta, denom_only_group_delta = local_state.get_vars(
        "numer_only_group", "denom_only_group")

  denom_only_group_delta = denom_only_group_delta.reshape((-1, 1))

  pt2_batch_score = X_delta_group - (numer_only_group_delta /
                                     denom_only_group_delta)

  msg.update(
      _compute_B(local_state, pt2_batch_score, False,
                 local_state.config.require_group_correction,
                 local_state.config.require_robust))

  if local_state.config.require_cox_correction:
    pt2_batch_cox_correction_score = _batch_score_cox_correction(
        local_state, pt2_batch_score, numer_all_delta, denom_all_delta)
    msg.update(
        _compute_B(local_state, pt2_batch_cox_correction_score, True,
                   local_state.config.require_group_correction,
                   local_state.config.require_robust))
  return msg


def _get_B(state, *B_names, cox_correction=False):
  return state.get_vars(*(B_names if not cox_correction else [(
      n + "_cox_correction") for n in B_names]))


def _eq2_master_variance(master_state: ClientState,
                         variance_setting: VarianceSetting) -> Message:

  (eq1_H, X_delta_sum, nxebkx_cs_ds, beta_k_hat,
   beta_hat) = master_state.get_vars(
       "eq1_H",
       "X_delta_sum",
       "nxebkx_cs_ds*",
       "beta_k_hat",
       "beta_hat",
   )

  eq2_hess_master = jacfwd(
      functools.partial(_eq2_model,
                        taylor_order=master_state.config.taylor_order), -1)
  I_diag_last = -eq2_hess_master(X_delta_sum, nxebkx_cs_ds, beta_k_hat,
                                 beta_hat)
  I_diag_inv_last = np.linalg.inv(I_diag_last)

  if variance_setting.group_correction:
    I_diag_wo_last = -eq1_H
    I_row_wo_last = -_eq2_grad_beta_k_master(X_delta_sum, nxebkx_cs_ds,
                                             beta_k_hat, beta_hat,
                                             master_state.config.taylor_order)
    I_diag_inv_wo_last = np.linalg.inv(I_diag_wo_last)
    if variance_setting.robust:
      B_diag_wo_last, B_diag_last, B_row_wo_last = _get_B(
          master_state,
          "B_diag_wo_last",
          "B_diag_last",
          "B_row_wo_last",
          cox_correction=variance_setting.cox_correction)
      S = np.einsum("ab,bBc,Bcd->Bad", I_diag_inv_last, I_row_wo_last,
                    I_diag_inv_wo_last)

      sas = np.einsum("Bab,Bbc,Bdc->ad", S, B_diag_wo_last, S)
      sb1s = np.einsum("ab,Bbc,Bdc->ad", I_diag_inv_last, B_row_wo_last, S)
      sb2s = sb1s.T  # pylint: disable=no-member
      scs = np.einsum('ab,kbc,dc->ad', I_diag_inv_last, B_diag_last,
                      I_diag_inv_last)
      cov = sas - sb1s - sb2s + scs
    else:
      S = np.einsum("ab,bBc->Bac", I_diag_inv_last, I_row_wo_last)

      cov = np.einsum(
          "Bab,Bbc,Bdc->ad", S, I_diag_inv_wo_last, S,
          optimize='optimal') + I_diag_inv_last
  elif variance_setting.robust:
    B_diag_last, = _get_B(master_state,
                          "B_diag_last",
                          cox_correction=variance_setting.cox_correction)
    B_diag_last = np.sum(B_diag_last, axis=0)
    cov = I_diag_inv_last @ B_diag_last @ I_diag_inv_last.T

  else:
    return

  master_state[str(variance_setting)] = cov


def eq2_master_variance(master_state: ClientState):
  for setting in master_state.config.variance_settings:
    _eq2_master_variance(master_state, setting)


#########################################################
# END
#########################################################

# if __name__ == '__main__':
#   N = 1000
#   K = 3
#   X_DIM = 3
#   import distributed_cox.data as vdata
#   key, subkey = jrandom.split(vdata.key)
#   group_sizes = vdata.group_sizes_generator(N, K, "same")
#   T, X, delta, beta, group_labels = vdata.data_generator(
#       N, X_DIM, group_sizes, return_T=True)(vdata.data_generation_key)

#   T_delta = T[delta == 1]

#   local_data: Sequence[Eq2LocalResult] = []
#   for k in range(K):
#     X_group = X[group_labels == k]
#     T_group = T[group_labels == k]
#     delta_group = delta[group_labels == k]
#     local_data.append(eq2_local(T_group, X_group, delta_group, T_delta))

#   beta_hat, cov_H = eq2_master(local_data)
