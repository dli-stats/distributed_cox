"""Cox model(s) with generic distributed modeling."""

import functools

from jax import vmap
import jax.numpy as jnp

import oryx

import distributed_cox.utils as vutils

import distributed_cox.generic.modeling as modeling
from distributed_cox.generic.taylor import taylor_approx
from distributed_cox.generic.distribute import (taylor_distribute, sum, cumsum)  # pylint: disable=redefined-builtin

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant


def _right_cumsum(X, axis=0):
  return jnp.cumsum(X[::-1], axis=axis)[::-1]


def _group_by(fun):
  """Simple wrapper for grouping."""

  def wrapped(X, delta, beta, group_labels, X_groups, delta_groups, beta_k_hat):
    K, group_size, _ = X_groups.shape
    batch_score = fun(X, delta, beta, group_labels, X_groups, delta_groups,
                      beta_k_hat)
    return vutils.group_by_labels(group_labels,
                                  batch_score,
                                  K=K,
                                  group_size=group_size)

  return wrapped


####################
# Ordinary Cox model
####################

mark, collect = modeling.model_temporaries("cox")


def unstratified_pooled_loglik(X, delta, beta):
  bx = jnp.dot(X, beta)
  ebx_cs = jnp.cumsum(jnp.exp(bx), 0)
  log_term = jnp.log(ebx_cs)
  batch_loglik = mark((bx - log_term) * delta, "batch_loglik")
  return jnp.sum(batch_loglik, axis=0)


def unstratified_pooled_score(X, delta, beta):
  bx = jnp.dot(X, beta)
  ebx = mark(jnp.exp(bx).reshape((-1, 1)), "ebx")
  ebx = taylor_approx(ebx, name="ebx")
  xebx = mark(X * ebx, "xebx")
  ebx_cs = mark(cumsum(ebx, name="ebx_cs"), "ebx_cs")
  xebx_cs = mark(cumsum(xebx, name="xebx_cs"), "xebx_cs")
  frac_term = xebx_cs / ebx_cs
  batch_score = mark((X - frac_term) * delta.reshape((-1, 1)), "batch_score")
  return sum(batch_score, name="sum_score")


def unstratified_pooled_hessian(X, delta, beta):
  """unstratified_pooled's hessian."""
  score_numer, score_denom = collect(unstratified_pooled_score,
                                     ["xebx", "ebx"])(X, delta, beta)

  score_numer_cs = cumsum(score_numer, name="xebx_cs")
  score_denom_cs = cumsum(score_denom, name="ebx_cs").reshape((-1, 1, 1))

  term_1 = (cumsum(jnp.einsum("Ni,Nj->Nij", X, score_numer, optimize="optimal"),
                   name="xxebx_cs") / score_denom_cs)

  term_2 = (jnp.einsum(
      "Ni,Nj->Nij", score_numer_cs, score_numer_cs, optimize="optimal") /
            score_denom_cs**2)

  batch_hessian = mark((term_2 - term_1) * delta.reshape((-1, 1, 1)),
                       "batch_hessian")
  return sum(batch_hessian, name="sum_hessian")


def unstratified_pooled_batch_robust_cox_correction_score(X, delta, beta):
  ebx, xebx, ebx_cs, xebx_cs, batch_score = collect(
      unstratified_pooled_score,
      ["ebx", "xebx", "ebx_cs", "xebx_cs", "batch_score"])(X, delta, beta)

  ebx_cs_1 = (1. / ebx_cs) * delta.reshape((-1, 1))
  term_1 = xebx * _right_cumsum(ebx_cs_1, axis=0)
  term_2 = ebx * _right_cumsum(xebx_cs * (ebx_cs_1**2), axis=0)
  score_correction_term = term_1 - term_2
  return batch_score - score_correction_term


####################
# END
####################

# convenience used by unstratified_distributed and stratified_distributed
_distribute = functools.partial(taylor_distribute, argnums=2)


@functools.lru_cache(maxsize=None)
def get_cox_fun(method: str, kind: str, batch: bool = False, order: int = 1):
  """Convenience routine to get a desired cox model analysis function."""
  # pylint: disable=too-many-branches
  if method in ('unstratified_pooled', 'stratified_pooled'):
    order = -1
  if method in ('unstratified_distributed',
                'stratified_distributed') and kind == "loglik":
    raise TypeError("Does not have loglik option")
  if kind == "robust_cox_correction_score" and not batch:
    raise TypeError("robust_cox_correction_score"
                    " does not have un-batched option")

  is_robust = kind == "robust_cox_correction_score"
  unstratified_pooled_fn = globals()["unstratified_pooled_{}".format(
      "batch_robust_cox_correction_score" if is_robust else kind)]

  if method == "unstratified_pooled":
    fn = unstratified_pooled_fn
    if batch and not is_robust:
      fn = collect(fn, "batch_{}".format(kind))
  elif method == "unstratified_distributed":
    fn = _distribute(unstratified_pooled_fn,
                     reduction_kind="cumsum",
                     orders={'ebx': order})
    if is_robust:
      fn = _group_by(fn)
    elif batch:
      fn = _group_by(collect(fn, "batch_{}".format(kind)))
  elif method == "stratified_pooled":
    fn = vmap(unstratified_pooled_fn, in_axes=(0, 0, None))
    if not batch:
      fn = getattr(modeling, "sum_{}".format(kind))(fn)
  elif method == "stratified_distributed":
    if batch and not is_robust:
      fn = collect(unstratified_pooled_fn, "batch_{}".format(kind))
    else:
      fn = unstratified_pooled_fn
    fn = _distribute(fn,
                     reduction_kind=None if batch else "sum",
                     orders={'ebx': order})
  else:
    raise TypeError("Invalid method")

  return fn


## Some pre-built cox functions

# Unstratified Pooled (together with root modeling)
unstratified_pooled_batch_loglik = collect(unstratified_pooled_loglik,
                                           "batch_loglik")
unstratified_pooled_batch_score = collect(unstratified_pooled_score,
                                          "batch_score")

# Unstratified Distributed
unstratified_distributed_score = get_cox_fun("unstratified_distributed",
                                             "score", False, 1)
unstratified_distributed_batch_score = get_cox_fun("unstratified_distributed",
                                                   "score", True, 1)
unstratified_distributed_batch_robust_cox_correction_score = get_cox_fun(
    "unstratified_distributed", "robust_cox_correction_score", True, 1)
unstratified_distributed_hessian_taylor = get_cox_fun(
    "unstratified_distributed", "hessian", False, 1)

# Stratified Pooled
stratified_pooled_batch_loglik = get_cox_fun("stratified_pooled", "loglik",
                                             True)
stratified_pooled_loglik = get_cox_fun("stratified_pooled", "loglik", False)
stratified_pooled_batch_robust_cox_correction_score = get_cox_fun(
    "stratified_pooled", "robust_cox_correction_score", True)

# Stratified Distributed
stratified_distributed_score = get_cox_fun("stratified_distributed", "score",
                                           False, 1)
stratified_distributed_batch_score = get_cox_fun("stratified_distributed",
                                                 "score", True, 1)
stratified_distributed_batch_robust_cox_correction_score = get_cox_fun(
    "stratified_distributed", "robust_cox_correction_score", True, 1)
stratified_distributed_hessian_taylor = get_cox_fun("stratified_distributed",
                                                    "hessian", False, 1)
