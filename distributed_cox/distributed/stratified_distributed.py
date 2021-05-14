"""Distributed implementation of Equation 4."""

from typing import Optional, List

import functools

import jax.numpy as jnp

from distributed_cox.distributed.common import (ClientState, Message,
                                                VarianceSetting)
import distributed_cox.generic.modeling as modeling
from distributed_cox.distributed import unstratified_distributed


def stratified_distributed_local(local_state: ClientState,
                                 initial_guess: Optional[List[float]] = None,
                                 loglik_eps: float = 1e-6,
                                 max_num_steps: int = 200) -> Message:
  T_group, delta_group = local_state.get_vars("T_group", "delta_group")
  local_state["T_delta"] = T_group[delta_group]
  msg = unstratified_distributed.unstratified_distributed_local(
      local_state,
      initial_guess=initial_guess,
      loglik_eps=loglik_eps,
      max_num_steps=max_num_steps)
  if local_state.config.require_cox_correction:
    # Under cox correction there's one more round of communication
    # So that all the variance estimations are done locally
    del msg["nxebkx_cs_ds{}".format(local_state.config.taylor_order + 2)]

  return msg


mark, collect = modeling.model_temporaries("distributed_stratified_distributed")


def _stratified_distributed_model(X_delta_sum,
                                  nxebkx_cs_ds,
                                  beta_k_hat,
                                  beta,
                                  taylor_order=1):
  D, X_dim = nxebkx_cs_ds[1].shape
  bmb = beta - beta_k_hat
  numer = jnp.zeros((D, X_dim), dtype=beta.dtype)
  denom = jnp.zeros(D, dtype=beta.dtype)
  dbeta_pow = jnp.ones((D, 1), dtype=beta.dtype)
  fact = 1.
  for i in range(taylor_order + 1):
    denom += jnp.einsum("dp,dp->d", nxebkx_cs_ds[i].reshape(
        (D, -1)), dbeta_pow) / fact
    numer += jnp.einsum("dip,dp->di", nxebkx_cs_ds[i + 1].reshape(
        (D, X_dim, -1)), dbeta_pow) / fact
    if i != taylor_order:
      dbeta_pow = jnp.einsum("dp,di->dpi", dbeta_pow, bmb).reshape((D, -1))
    fact *= (i + 1)

  mark(dbeta_pow, "dbeta_pow_order")  # (b - b0)^|order - 1|
  numer = mark(numer, "numer")
  denom = mark(denom, "denom")

  return jnp.sum(X_delta_sum, axis=0) - jnp.sum(numer / denom.reshape((-1, 1)),
                                                axis=0)


def solve_stratified_distributed_model(master_state: ClientState,
                                       score_norm_eps: float = 1e-3,
                                       max_num_steps: int = 100):
  """Solves stratified_distributed_model."""
  X_delta_sum, nxebkx_cs_ds, beta_k_hat, group_sizes = master_state.get_vars(
      "X_delta_sum", "nxebkx_cs_ds*", "beta_k_hat", "group_sizes")
  beta_guess = jnp.mean(beta_k_hat, axis=0)
  beta_k_hat = jnp.repeat(beta_k_hat, group_sizes, axis=0)
  fn_to_solve = functools.partial(_stratified_distributed_model,
                                  X_delta_sum,
                                  nxebkx_cs_ds,
                                  beta_k_hat,
                                  taylor_order=master_state.config.taylor_order)
  return modeling.solve_single(fn_to_solve,
                               use_likelihood=False,
                               score_norm_eps=score_norm_eps,
                               max_num_steps=max_num_steps)(beta_guess)


def stratified_distributed_master(master_state: ClientState,
                                  score_norm_eps=1e-3,
                                  max_num_steps=1000) -> Message:
  nxebkx_cs_ds, = master_state.get_vars("nxebkx_cs_ds*")
  master_state["group_sizes"] = jnp.array(list(map(len, nxebkx_cs_ds[0])))
  nxebkx_cs_ds_concat = tuple(jnp.concatenate(x, axis=0) for x in nxebkx_cs_ds)
  master_state["nxebkx_cs_ds*"] = nxebkx_cs_ds_concat

  sol = solve_stratified_distributed_model(master_state,
                                           score_norm_eps=score_norm_eps,
                                           max_num_steps=max_num_steps)

  beta_hat = sol.guess
  if not sol.converged:
    print(f"Did not converge in {max_num_steps} steps!")

  master_state["converged"] = sol.converged
  master_state["beta_hat"] = beta_hat
  master_state[str(VarianceSetting(False, False,
                                   False))] = modeling.cov_H()(sol)

  if master_state.config.require_cox_correction:
    # One more round
    return {"beta_hat": beta_hat}
  # Compute all variances now
  master_state["nxebkx_cs_ds*"] = nxebkx_cs_ds
  stratified_distributed_master_all_variances(master_state)
  del master_state["nxebkx_cs_ds*"]
  return {}


def stratified_distributed_local_variance(local_state: ClientState) -> Message:
  (beta_k_hat, beta_hat, X_group, delta_group, T_group,
   T_delta) = local_state.get_vars("beta_k_hat", "beta_hat", "X_group",
                                   "delta_group", "T_group", "T_delta")
  taylor_order = local_state.config.taylor_order
  # Obtain the denom and numers for the current group
  nxebkxs = unstratified_distributed.compute_nxebkxs(
      X_group, beta_k_hat, required_orders=taylor_order + 2)
  nxebkx_css = [jnp.cumsum(nxebkx, 0) for nxebkx in nxebkxs]

  denom, numer = collect(_stratified_distributed_model, ["denom", "numer"])(
      jnp.zeros_like(beta_k_hat).reshape((1, -1)),  # argument not used
      nxebkx_css,
      jnp.tile(beta_k_hat, (X_group.shape[0], 1)),
      beta_hat,
      taylor_order=taylor_order,
  )
  denom_only_group_delta, numer_only_group_delta = (denom[delta_group],
                                                    numer[delta_group])
  denom_only_group_delta = denom_only_group_delta.reshape((-1, 1))

  msg = {}

  pt2_batch_score = X_group[delta_group] - (numer_only_group_delta /
                                            denom_only_group_delta)

  msg.update(
      unstratified_distributed.compute_B(
          local_state, pt2_batch_score, False,
          local_state.config.require_group_correction,
          local_state.config.require_robust))

  if local_state.config.require_cox_correction:
    pt2_batch_cox_correction_score = unstratified_distributed.batch_score_cox_correction(
        local_state, pt2_batch_score, numer_only_group_delta,
        denom_only_group_delta)
    msg.update(
        unstratified_distributed.compute_B(
            local_state, pt2_batch_cox_correction_score, True,
            local_state.config.require_group_correction,
            local_state.config.require_robust))

  idxs = jnp.searchsorted(-T_group, -T_delta, side='right')
  nxebkx_cs_ds = tuple(
      jnp.take(nxebkx_cs, idxs, axis=0) for nxebkx_cs in nxebkx_css)

  grad_beta_k = _compute_stratified_distributed_grad_beta_k(
      nxebkx_cs_ds, beta_k_hat, beta_hat, taylor_order)
  msg['grad_beta_k'] = grad_beta_k
  return msg


def _compute_stratified_distributed_grad_beta_k(nxebkx_cs_ds, beta_k_hat, beta,
                                                taylor_order):
  D, X_dim = nxebkx_cs_ds[1].shape

  t1xebkx_cs_ds = nxebkx_cs_ds[taylor_order + 1]  # order + 1 Xs times ebx
  t2xebkx_cs_ds = nxebkx_cs_ds[taylor_order + 2]  # order + 2 Xs times ebx

  numer, denom, dbeta_pow_t = collect(_stratified_distributed_model,
                                      ["numer", "denom", "dbeta_pow_order"])(
                                          jnp.zeros(X_dim),  # not used
                                          nxebkx_cs_ds,
                                          jnp.tile(beta_k_hat, (D, 1)),
                                          beta,
                                          taylor_order=taylor_order)
  dbeta_pow_t = dbeta_pow_t[0]
  t1ebkxbmb_cs_d = jnp.einsum("dip,p->di", t1xebkx_cs_ds.reshape(
      (D, X_dim, -1)), dbeta_pow_t)
  t2ebkxbmb_cs_d = jnp.einsum("dijp,p->dij",
                              t2xebkx_cs_ds.reshape((D, X_dim, X_dim, -1)),
                              dbeta_pow_t)

  denom = denom.reshape((len(denom), 1, 1))

  return jnp.sum((jnp.einsum("dm,dn->dmn", numer, t1ebkxbmb_cs_d) / denom**2 -
                  t2ebkxbmb_cs_d / denom),
                 axis=0)


def _get_or_compute_unstratified_distributed_grad_beta_k(
    master_state: ClientState):
  if "grad_beta_k" not in master_state:
    nxebkx_cs_ds, beta_k_hat, beta_hat, group_sizes = master_state.get_vars(
        "nxebkx_cs_ds*", "beta_k_hat", "beta_hat", "group_sizes")
    idx = 0
    grad_beta_k = []
    for i, group_size in enumerate(group_sizes):
      nxebkx_cs_ds_group = [n[idx:idx + group_size] for n in nxebkx_cs_ds]
      idx += group_size
      grad_beta_k_group = _compute_stratified_distributed_grad_beta_k(
          nxebkx_cs_ds_group, beta_k_hat[i], beta_hat,
          master_state.config.taylor_order)
      grad_beta_k.append(grad_beta_k_group)
    return jnp.stack(grad_beta_k)
  return master_state.get_var("grad_beta_k")


def _get_B(state, *B_names, cox_correction=False):
  return state.get_vars(*(B_names if not cox_correction else [(
      n + "_cox_correction") for n in B_names]))


def _stratified_distributed_master_variance(master_state: ClientState,
                                            variance_setting: VarianceSetting):
  unstratified_pooled_H = master_state.get_var("unstratified_pooled_H")

  I_diag_inv_last = master_state[str(VarianceSetting(False, False, False))]

  if variance_setting.group_correction:
    I_diag_wo_last = -unstratified_pooled_H
    I_row_wo_last = -_get_or_compute_unstratified_distributed_grad_beta_k(
        master_state)
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


def stratified_distributed_master_all_variances(master_state: ClientState):
  for setting in master_state.config.variance_settings:
    _stratified_distributed_master_variance(master_state, setting)
