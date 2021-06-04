"""Convenience functions for solving Cox models."""

from typing import Callable, Dict, Mapping, Optional, Any, Sequence

import collections

import itertools
import functools

import jax.numpy as jnp
from jax import jacfwd, jacrev

import distributed_cox.cox as cox
import distributed_cox.generic.modeling as modeling
import distributed_cox.data as vdata

CoxSolveResult = collections.namedtuple("CoxSolveResult", "pt1 pt2 covs")

_NUM_SINGLE_ARGS = 3


def get_cox_solve_fn(method: str,
                     K: Optional[int] = None,
                     distributed: Optional[Mapping[str, Any]] = None,
                     solver: Optional[Mapping[str, Any]] = None,
                     meta_analysis: Optional[Mapping[str, Any]] = None):
  """Constructs a single function that solve Cox model in one shot.

  Args:
    method: the analysis method.
    K: number of groups in the data.
    distributed: a Mapping configuration for the distributed setting.
    solver: a Mapping configuration for the solver.
    meta_analysis: a Mapping configuration for the meta_analysis, if applicable.

  Returns:
    a function that takes in the Cox inputs, and outputs the solved beta
    estimate.
  """
  distributed = distributed or {}
  solver = solver or {}
  meta_analysis = meta_analysis or {}

  get_cox_fun = functools.partial(cox.get_cox_fun,
                                  order=distributed.get('taylor_order', -1))

  if method == "meta_analysis":
    solve_fn = functools.partial(modeling.solve_meta_analysis,
                                 get_cox_fun("unstratified_pooled",
                                             "loglik",
                                             batch=False),
                                 use_likelihood=True,
                                 **meta_analysis)
  else:
    if method in ("unstratified_pooled", "stratified_pooled"):
      log_likelihood_fn = get_cox_fun(method, "loglik", batch=False)
      use_likelihood = True
      solve_fn = functools.partial(modeling.solve_single,
                                   log_likelihood_fn,
                                   use_likelihood=use_likelihood)
    elif method in ("unstratified_distributed", "stratified_distributed"):
      loglik_or_score_fn = get_cox_fun(method, "score", batch=False)
      use_likelihood = False
      if distributed.get("hessian_use_taylor", True):
        hessian_fn = get_cox_fun(method, "hessian", batch=False)
      else:
        hessian_fn = jacfwd(loglik_or_score_fn, _NUM_SINGLE_ARGS - 1)
      solve_fn = functools.partial(
          modeling.solve_distributed,
          get_cox_fun("unstratified_pooled", "loglik", batch=False),
          distributed_hessian_fn=hessian_fn,
          num_single_args=_NUM_SINGLE_ARGS,
          K=K,
          pt2_use_average_guess=distributed.get("pt2_use_average_guess", True),
          single_use_likelihood=True,
      )
      solve_fn = functools.partial(solve_fn,
                                   loglik_or_score_fn,
                                   distributed_use_likelihood=use_likelihood)
    else:
      raise ValueError("invalid method")

  return solve_fn(**solver)


def get_cov_fn(  # pylint: disable=too-many-return-statements
    method: str,
    group_correction: bool,
    sandwich_robust: bool,
    sandwich_robust_sum_group_first: bool,
    cox_correction: bool,
    distributed: Optional[Mapping[str, Any]] = None,
    meta_analysis: Optional[Mapping[str, Any]] = None,
):
  """Returns a function that computes a single convariance setting."""
  distributed = distributed or {}
  meta_analysis = meta_analysis or {}

  # Reject some non-sensical situations
  if method == "meta_analysis":
    if (group_correction or sandwich_robust or cox_correction or
        sandwich_robust_sum_group_first):
      return None
    return modeling.cov_meta_analysis(**meta_analysis)

  if not sandwich_robust and cox_correction:
    return None
  if not sandwich_robust and sandwich_robust_sum_group_first:
    return None
  if group_correction and method in ("unstratified_pooled",
                                     "stratified_pooled"):
    return None
  if sandwich_robust_sum_group_first and method in ("unstratified_pooled",
                                                    "unstratified_distributed"):
    return None
  if sandwich_robust_sum_group_first:  # disabled because it's too bad
    return None

  get_cox_fun = functools.partial(cox.get_cox_fun,
                                  order=distributed.get('taylor_order', -1))

  if method in ("unstratified_pooled", "stratified_pooled"):
    batch_loglik_or_score_fn = get_cox_fun(method, "loglik", batch=True)
    use_likelihood = True
  elif method in ("unstratified_distributed", "stratified_distributed"):
    use_likelihood = False
    batch_loglik_or_score_fn = get_cox_fun(method, "score", batch=True)

  batch_robust_cox_correction_score = get_cox_fun(
      method, "robust_cox_correction_score", True)
  if group_correction:
    batch_score = get_cox_fun(method, "score", batch=True)
    cov_fn = modeling.cov_group_correction(
        (get_cox_fun("unstratified_pooled", "robust_cox_correction_score",
                     True) if cox_correction else get_cox_fun(
                         "unstratified_pooled", "loglik", True)),
        (batch_robust_cox_correction_score if cox_correction else batch_score),
        distributed_cross_hessian_fn=jacrev(
            get_cox_fun(method, "score", batch=False), -1),
        batch_single_use_likelihood=not cox_correction,
        batch_distributed_use_likelihood=False,
        num_single_args=_NUM_SINGLE_ARGS,
        robust=sandwich_robust,
        robust_sum_group_first=sandwich_robust_sum_group_first)
  elif sandwich_robust:
    cov_batch_log_likelihood_or_score_fn = (batch_robust_cox_correction_score
                                            if cox_correction else
                                            batch_loglik_or_score_fn)
    cov_fn = modeling.cov_robust(
        batch_log_likelihood_or_score_fn=cov_batch_log_likelihood_or_score_fn,
        use_likelihood=(use_likelihood and not cox_correction),
        num_single_args=_NUM_SINGLE_ARGS,
        sum_group_first=sandwich_robust_sum_group_first)
  else:
    cov_fn = modeling.cov_H()

  return cov_fn


def get_cov_fns(
    method: str,
    distributed: Optional[Mapping[str, Any]] = None,
    meta_analysis: Optional[Mapping[str, Any]] = None) -> Dict[str, Callable]:
  """Constructs function to compute all covariances for an analysis."""
  distributed = distributed or {}
  meta_analysis = meta_analysis or {}

  cov_fns = {}
  for cov_setting in itertools.product(*[(True, False)] * 4):
    (group_correction, sandwich_robust, cox_correction,
     sandwich_robust_sum_group_first) = cov_setting
    cov_fn = get_cov_fn(
        method,
        group_correction=group_correction,
        sandwich_robust=sandwich_robust,
        cox_correction=cox_correction,
        sandwich_robust_sum_group_first=sandwich_robust_sum_group_first,
        distributed=distributed,
        meta_analysis=meta_analysis,
    )
    if cov_fn is None:
      continue
    if method == "meta_analysis":
      cov_name = "cov:meta_analysis"
    else:
      cov_name = ("cov:{}group_correction|{}sandwich"
                  "|{}cox_correction|{}sum_first").format(
                      *["" if v else "no_" for v in cov_setting])
    cov_fns[cov_name] = cov_fn
  return cov_fns


def get_cox_solve_and_cov_fn(method: str,
                             group_sizes: Sequence[int] = None,
                             distributed: Optional[Mapping[str, Any]] = None,
                             solver: Optional[Mapping[str, Any]] = None,
                             meta_analysis: Optional[Mapping[str, Any]] = None):
  """Constructs a single function that solve Cox model and covs in one shot.

  Args:
    method: the analysis method.
    group_sizes: a sequence of group sizes. If single site of size N, pass in
      a single element sequence e.g. (N,).
    distributed: see `get_cox_solve_fn`.
    solver: see `get_cox_solve_fn`.
    meta_analysis: see `get_cox_solve_fn`.

  Returns:
    a function that takes in the Cox inputs, and outputs the solved beta
    estimate as well as all the covariances.
  """
  distributed = distributed or {}
  solver = solver or {}
  meta_analysis = meta_analysis or {}

  K = len(group_sizes)

  solve_fn = get_cox_solve_fn(method,
                              K,
                              distributed=distributed,
                              solver=solver,
                              meta_analysis=meta_analysis)

  cov_fns = get_cov_fns(method,
                        distributed=distributed,
                        meta_analysis=meta_analysis)

  max_group_size = max(group_sizes)

  def solve_and_cov(X, delta, beta, group_labels) -> CoxSolveResult:
    initial_beta_hat = beta
    if method != "unstratified_pooled":
      X_groups, delta_groups = vdata.group_data_by_labels(
          group_labels, X, delta, K=K, group_size=max_group_size)
      initial_beta_k_hat = jnp.broadcast_to(beta, (K,) + beta.shape)

    if method == "unstratified_pooled":
      pt1_sol = None
      pt2_sol = sol = solve_fn(X, delta, initial_beta_hat)
      model_args = (X, delta, sol.guess)
    elif method == "stratified_pooled":
      pt1_sol = None
      pt2_sol = sol = solve_fn(X_groups, delta_groups, initial_beta_hat)
      model_args = (X_groups, delta_groups, sol.guess)
    elif method in ("unstratified_distributed", "stratified_distributed"):
      pt1_sol, pt2_sol = sol = solve_fn(X, delta, initial_beta_hat,
                                        group_labels, X_groups, delta_groups,
                                        initial_beta_k_hat)
      model_args = (X, delta, pt2_sol.guess, group_labels, X_groups,
                    delta_groups, pt1_sol.guess)
    elif method == "meta_analysis":
      sol = solve_fn(X_groups, delta_groups, initial_beta_k_hat)
      pt1_sol, pt2_sol = sol.pt1, sol.pt2
      model_args = tuple()

    cov_results = {}
    for cov_name, cov_fn in cov_fns.items():
      cov_results[cov_name] = cov_fn(sol, *model_args)

    return CoxSolveResult(pt1=pt1_sol, pt2=pt2_sol, covs=cov_results)

  return solve_and_cov
