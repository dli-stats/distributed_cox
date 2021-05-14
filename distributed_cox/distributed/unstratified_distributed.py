"""Distributed implementation of Equation 2."""

from typing import Optional, List

import functools

import opt_einsum as oe

import jax.numpy as jnp
import jax.lax as lax

import distributed_cox.generic.modeling as modeling

from distributed_cox.cox import (unstratified_pooled_loglik,
                                 unstratified_pooled_batch_score)

from distributed_cox.distributed.common import (VarianceSetting, ClientState,
                                                Message)


def unstratified_distributed_local_send_T(local_state: ClientState) -> Message:
  T_group = local_state["T_group"]
  delta_group = local_state["delta_group"]
  return {"T_delta_per_group": T_group[delta_group]}


def unstratified_distributed_master_send_T(master_state: ClientState
                                          ) -> Message:
  # Pop T_delta_oper_group since it will not be numpy serializable
  T_deltas = list(master_state.pop("T_delta_per_group"))
  num_groups = len(T_deltas)

  msg = {}

  T_delta = jnp.concatenate(T_deltas)
  idxs = jnp.argsort(-T_delta)
  T_delta = T_delta[idxs]
  master_state["T_delta"] = T_delta
  T_delta_group_labels = jnp.concatenate(
      [jnp.repeat(k, len(T_deltas[k])) for k in range(num_groups)])
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


def solve_distributed_pt1(local_state: ClientState,
                          initial_guess: Optional[List[float]] = None,
                          loglik_eps: float = 1e-6,
                          max_num_steps: int = 10):
  """Solves unstratified_pooled for a single group."""

  X_group, delta_group = local_state.get_vars(
      "X_group",
      "delta_group",
  )
  if initial_guess is None:
    initial_guess = jnp.zeros((X_group.shape[1],))

  solve_unstratified_pooled_fn = modeling.solve_single(
      unstratified_pooled_loglik,
      use_likelihood=True,
      loglik_eps=loglik_eps,
      max_num_steps=max_num_steps)
  return solve_unstratified_pooled_fn(X_group, delta_group, initial_guess)


def compute_nxebkxs(X_group, beta_k_hat, required_orders: int):
  """Computes all of nxebkx for n = required_orders."""
  ebkx = jnp.exp(jnp.dot(X_group, beta_k_hat))
  nxebkxs = [ebkx]
  cur_ebkx = ebkx

  batch_dim_name = oe.get_symbol(0)
  x_dim_name = oe.get_symbol(1)
  o_dim_names = ""
  for i in range(1, required_orders + 1):
    cur_ebkx = jnp.einsum(
        "{B}{o},{B}{x}->{B}{o}{x}".format(B=batch_dim_name,
                                          x=x_dim_name,
                                          o=o_dim_names), cur_ebkx, X_group)
    o_dim_names += oe.get_symbol(i + 1)
    nxebkxs.append(cur_ebkx)
  return nxebkxs


def unstratified_distributed_local(local_state: ClientState,
                                   initial_guess: Optional[List[float]] = None,
                                   loglik_eps: float = 1e-6,
                                   max_num_steps: int = 30) -> Message:
  """Compute local values."""

  T_group, X_group, delta_group, T_delta = local_state.get_vars(
      "T_group",
      "X_group",
      "delta_group",
      "T_delta",
  )

  assert T_group.shape[0] == X_group.shape[0] == delta_group.shape[0]
  assert local_state.config.taylor_order >= 1

  unstratified_pooled_sol = solve_distributed_pt1(local_state, initial_guess,
                                                  loglik_eps, max_num_steps)
  if not unstratified_pooled_sol.converged:
    print(f"Did not converge in {max_num_steps}!")

  beta_k_hat = unstratified_pooled_sol.guess
  unstratified_pooled_H = unstratified_pooled_sol.hessian

  # e.g. order=1 ==> need X * X * exp(beta * X)
  required_orders = local_state.config.taylor_order + 1
  # e.g. group_correction further requires X * X * X * exp(beta * X)
  if local_state.config.require_group_correction:
    required_orders += 1

  nxebkxs = compute_nxebkxs(X_group,
                            beta_k_hat,
                            required_orders=required_orders)
  nxebkx_css = [jnp.cumsum(nxebkx, 0) for nxebkx in nxebkxs]

  idxs = jnp.searchsorted(-T_group, -T_delta, side='right')

  nxebkx_cs_ds = {
      "nxebkx_cs_ds{}".format(i): jnp.take(nxebkx_cs, idxs, axis=0)
      for i, nxebkx_cs in enumerate(nxebkx_css)
  }

  X_delta_sum = jnp.sum(X_group * delta_group.reshape((-1, 1)), axis=0)

  local_state["beta_k_hat"] = beta_k_hat

  return dict(unstratified_pooled_H=unstratified_pooled_H,
              X_delta_sum=X_delta_sum,
              beta_k_hat=beta_k_hat,
              **nxebkx_cs_ds)


mark, collect = modeling.model_temporaries(
    "distributed_unstratified_distributed")


def _unstratified_distributed_model(X_delta_sum,
                                    nxebkx_cs_ds,
                                    beta_k_hat,
                                    beta,
                                    taylor_order=1):
  K, D, X_dim = nxebkx_cs_ds[1].shape

  bmb = beta - beta_k_hat
  numer = jnp.zeros((D, X_dim), dtype=beta.dtype)
  denom = jnp.zeros(D, dtype=beta.dtype)
  dbeta_pow = jnp.ones((K, 1), dtype=beta.dtype)
  fact = 1.
  for i in range(taylor_order + 1):
    denom += jnp.einsum("kdp,kp->d", nxebkx_cs_ds[i].reshape(
        (K, D, -1)), dbeta_pow) / fact
    numer += jnp.einsum("kdip,kp->di", nxebkx_cs_ds[i + 1].reshape(
        (K, D, X_dim, -1)), dbeta_pow) / fact
    if i != taylor_order:
      dbeta_pow = jnp.einsum("kp,ki->kpi", dbeta_pow, bmb).reshape((K, -1))
    fact *= (i + 1)

  mark(dbeta_pow, "dbeta_pow_order")  # (b - b0)^|order - 1|
  numer = mark(numer, "numer")
  denom = mark(denom, "denom")

  return jnp.sum(X_delta_sum, axis=0) - jnp.sum(numer / denom.reshape((-1, 1)),
                                                axis=0)


def _unstratified_distributed_grad_beta_k_master(X_delta_sum, nxebkx_cs_ds,
                                                 beta_k_hat, beta,
                                                 taylor_order):
  K, D, X_dim = nxebkx_cs_ds[1].shape

  t1xebkx_cs_ds = nxebkx_cs_ds[taylor_order + 1]  # order + 1 Xs times ebx
  t2xebkx_cs_ds = nxebkx_cs_ds[taylor_order + 2]  # order + 2 Xs times ebx

  numer, denom, dbeta_pow_t = collect(_unstratified_distributed_model,
                                      ["numer", "denom", "dbeta_pow_order"])(
                                          X_delta_sum,
                                          nxebkx_cs_ds,
                                          beta_k_hat,
                                          beta,
                                          taylor_order=taylor_order)
  t1ebkxbmb_cs_d = jnp.einsum("kdip,kp->kdi",
                              t1xebkx_cs_ds.reshape((K, D, X_dim, -1)),
                              dbeta_pow_t)
  t2ebkxbmb_cs_d = jnp.einsum("kdijp,kp->kdij",
                              t2xebkx_cs_ds.reshape((K, D, X_dim, X_dim, -1)),
                              dbeta_pow_t)

  denom = denom.reshape((1, len(denom), 1, 1))
  return jnp.sum((jnp.einsum("dm,kdn->kdmn", numer, t1ebkxbmb_cs_d) / denom**2 -
                  t2ebkxbmb_cs_d / denom),
                 axis=1)


def solve_unstratified_distributed_model(master_state: ClientState,
                                         score_norm_eps: float = 1e-3,
                                         max_num_steps: int = 30):
  """Solves unstratified_distributed_model."""
  X_delta_sum, nxebkx_cs_ds, beta_k_hat = master_state.get_vars(
      "X_delta_sum",
      "nxebkx_cs_ds*",
      "beta_k_hat",
  )
  beta_guess = jnp.mean(beta_k_hat, axis=0)

  fn_to_solve = functools.partial(_unstratified_distributed_model,
                                  X_delta_sum,
                                  nxebkx_cs_ds,
                                  beta_k_hat,
                                  taylor_order=master_state.config.taylor_order)
  return modeling.solve_single(fn_to_solve,
                               use_likelihood=False,
                               score_norm_eps=score_norm_eps,
                               max_num_steps=max_num_steps)(beta_guess)


def unstratified_distributed_master(master_state: ClientState,
                                    score_norm_eps: float = 1e-3,
                                    max_num_steps: int = 30) -> Message:
  """Compute master values from local results."""

  X_delta_sum, nxebkx_cs_ds, beta_k_hat = master_state.get_vars(
      "X_delta_sum",
      "nxebkx_cs_ds*",
      "beta_k_hat",
  )

  sol = solve_unstratified_distributed_model(master_state,
                                             score_norm_eps=score_norm_eps,
                                             max_num_steps=max_num_steps)
  beta_hat = sol.guess

  cov_H = modeling.cov_H()(sol)
  master_state["beta_hat"] = beta_hat
  master_state[str(VarianceSetting(False, False, False))] = cov_H

  msg = {}

  if not master_state.config.require_robust:
    unstratified_distributed_master_all_variances(master_state)
    return msg

  if master_state.config.require_robust:
    numer, denom = collect(_unstratified_distributed_model, ["numer", "denom"])(
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
  return jnp.cumsum(X[::-1], axis=axis)[::-1]


def batch_score_cox_correction(local_state: ClientState, batch_score,
                               numer_all_delta, denom_all_delta):
  (beta_k_hat, beta_hat, X_group, T_group, T_delta,
   delta_group) = local_state.get_vars("beta_k_hat", "beta_hat", "X_group",
                                       "T_group", "T_delta", "delta_group")
  # Obtain the denom and numers for the current group
  nxebkxs = compute_nxebkxs(X_group,
                            beta_k_hat,
                            required_orders=local_state.config.taylor_order + 1)
  denom_all_group, numer_all_group = collect(
      _unstratified_distributed_model, ["denom", "numer"])(
          jnp.zeros_like(beta_k_hat).reshape((1, -1)),  # argument not used
          [x.reshape((1,) + x.shape) for x in nxebkxs],
          beta_k_hat.reshape((1, -1)),
          beta_hat.reshape((1, -1)),
          taylor_order=local_state.config.taylor_order,
      )
  denom_all_group = denom_all_group.reshape((-1, 1))
  idxs = jnp.searchsorted(-T_delta, -T_group, side='right')
  denom_all_delta_1 = 1. / denom_all_delta
  term1 = numer_all_group * _right_cumsum(denom_all_delta_1, axis=0)[idxs]
  term2 = denom_all_group * _right_cumsum(
      numer_all_delta * (denom_all_delta_1**2), axis=0)[idxs]

  # Pad batch_score to entire group's size where non-marked indices are set to 0
  batch_score = jnp.array(batch_score)

  def insert_at(carry, delta):
    return carry + delta, batch_score[carry] * delta

  _, batch_score = lax.scan(insert_at, 0, delta_group)
  return batch_score - (term1 - term2)


def compute_B(local_state,
              pt2_batch_score,
              cox_correction=False,
              group_correction=False,
              robust=False):
  ret = {}
  ret['B_diag_last'] = jnp.einsum("ni,nj->ij",
                                  pt2_batch_score,
                                  pt2_batch_score,
                                  optimize="optimal")
  if group_correction and robust:
    X_group, delta_group, beta_k_hat = local_state.get_vars(
        "X_group", "delta_group", "beta_k_hat")
    pt1_batch_score = unstratified_pooled_batch_score(X_group, delta_group,
                                                      beta_k_hat)
    ret['B_diag_wo_last'] = jnp.einsum("ni,nj->ij", pt1_batch_score,
                                       pt1_batch_score)
    if not cox_correction:
      pt1_batch_score = pt1_batch_score[delta_group]
    ret['B_row_wo_last'] = jnp.einsum("ni,nj->ij", pt1_batch_score,
                                      pt2_batch_score)

  if cox_correction:
    ret = {(k + "_cox_correction"): v for k, v in ret.items()}

  return ret


def unstratified_distributed_local_variance(local_state: ClientState
                                           ) -> Message:

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
      compute_B(local_state, pt2_batch_score, False,
                local_state.config.require_group_correction,
                local_state.config.require_robust))

  if local_state.config.require_cox_correction:
    pt2_batch_cox_correction_score = batch_score_cox_correction(
        local_state, pt2_batch_score, numer_all_delta, denom_all_delta)
    msg.update(
        compute_B(local_state, pt2_batch_cox_correction_score, True,
                  local_state.config.require_group_correction,
                  local_state.config.require_robust))
  return msg


def _get_B(state, *B_names, cox_correction=False):
  return state.get_vars(*(B_names if not cox_correction else [(
      n + "_cox_correction") for n in B_names]))


def _unstratified_distributed_master_variance(master_state: ClientState,
                                              variance_setting: VarianceSetting
                                             ) -> Message:

  (unstratified_pooled_H, X_delta_sum, nxebkx_cs_ds, beta_k_hat,
   beta_hat) = master_state.get_vars(
       "unstratified_pooled_H",
       "X_delta_sum",
       "nxebkx_cs_ds*",
       "beta_k_hat",
       "beta_hat",
   )

  I_diag_inv_last = master_state[str(VarianceSetting(False, False, False))]

  if variance_setting.group_correction:
    I_diag_wo_last = -unstratified_pooled_H
    I_row_wo_last = -_unstratified_distributed_grad_beta_k_master(
        X_delta_sum, nxebkx_cs_ds, beta_k_hat, beta_hat,
        master_state.config.taylor_order)
    I_diag_inv_wo_last = jnp.linalg.inv(I_diag_wo_last)
    if variance_setting.robust:
      B_diag_wo_last, B_diag_last, B_row_wo_last = _get_B(
          master_state,
          "B_diag_wo_last",
          "B_diag_last",
          "B_row_wo_last",
          cox_correction=variance_setting.cox_correction)
      S = jnp.einsum("ab,Bbc,Bcd->Bad", I_diag_inv_last, I_row_wo_last,
                     I_diag_inv_wo_last)

      sas = jnp.einsum("Bab,Bbc,Bdc->ad", S, B_diag_wo_last, S)
      sb1s = jnp.einsum("ab,Bbc,Bdc->ad", I_diag_inv_last, B_row_wo_last, S)
      sb2s = sb1s.T  # pylint: disable=no-member
      scs = jnp.einsum('ab,kbc,dc->ad', I_diag_inv_last, B_diag_last,
                       I_diag_inv_last)
      cov = sas - sb1s - sb2s + scs
    else:
      S = jnp.einsum("ab,Bbc->Bac", I_diag_inv_last, I_row_wo_last)

      cov = jnp.einsum(
          "Bab,Bbc,Bdc->ad", S, I_diag_inv_wo_last, S,
          optimize='optimal') + I_diag_inv_last
  elif variance_setting.robust:
    B_diag_last, = _get_B(master_state,
                          "B_diag_last",
                          cox_correction=variance_setting.cox_correction)
    B_diag_last = jnp.sum(B_diag_last, axis=0)
    cov = I_diag_inv_last @ B_diag_last @ I_diag_inv_last.T

  else:
    return

  master_state[str(variance_setting)] = cov


def unstratified_distributed_master_all_variances(master_state: ClientState):
  for setting in master_state.config.variance_settings:
    _unstratified_distributed_master_variance(master_state, setting)
