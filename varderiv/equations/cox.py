"""Cox model(s) with generic distributed modeling."""

import functools

from jax import vmap
import jax.numpy as np

import oryx

import varderiv.utils as vutils

import varderiv.generic.modeling as modeling
from varderiv.generic.taylor import taylor_approx
from varderiv.generic.distribute import (taylor_distribute, sum, cumsum)  # pylint: disable=redefined-builtin

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant


def _right_cumsum(X, axis=0):
  return np.cumsum(X[::-1], axis=axis)[::-1]


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


mark, collect = modeling.model_temporaries("cox")


def eq1_loglik(X, delta, beta):
  bx = np.dot(X, beta)
  ebx_cs = np.cumsum(np.exp(bx), 0)
  log_term = np.log(ebx_cs)
  batch_loglik = mark((bx - log_term) * delta, "batch_loglik")
  return np.sum(batch_loglik, axis=0)


def eq1_score(X, delta, beta):
  bx = np.dot(X, beta)
  ebx = mark(np.exp(bx).reshape((-1, 1)), "ebx")
  ebx = taylor_approx(ebx, name="ebx")
  xebx = mark(X * ebx, "xebx")
  ebx_cs = mark(cumsum(ebx, name="ebx_cs"), "ebx_cs")
  xebx_cs = mark(cumsum(xebx, name="xebx_cs"), "xebx_cs")
  frac_term = xebx_cs / ebx_cs
  batch_score = mark((X - frac_term) * delta.reshape((-1, 1)), "batch_score")
  return sum(batch_score, name="sum_score")


def eq1_hessian(X, delta, beta):
  """Eq1's hessian."""
  score_numer, score_denom = collect(eq1_score, ["xebx", "ebx"])(X, delta, beta)

  score_numer_cs = cumsum(score_numer, name="numer_cs")
  score_denom_cs = cumsum(score_denom, name="denom_cs").reshape((-1, 1, 1))

  term_1 = (np.cumsum(
      np.einsum("Ni,Nj->Nij", X, score_numer, optimize="optimal"), axis=0) /
            score_denom_cs)

  term_2 = (np.einsum(
      "Ni,Nj->Nij", score_numer_cs, score_numer_cs, optimize="optimal") /
            score_denom_cs**2)

  batch_hessian = mark((term_2 - term_1) * delta.reshape((-1, 1, 1)),
                       "batch_hessian")
  return np.sum(batch_hessian, axis=0)


def eq1_batch_robust_cox_correction_score(X, delta, beta):
  ebx, ebx_cs, xebx_cs, batch_score = collect(
      eq1_score, ["ebx", "ebx_cs", "xebx_cs", "batch_score"])(X, delta, beta)

  ebx_cs_1 = (1. / ebx_cs) * delta.reshape((-1, 1))
  term_1 = X * ebx * _right_cumsum(ebx_cs_1, axis=0)
  term_2 = ebx * _right_cumsum(xebx_cs * (ebx_cs_1**2), axis=0)
  score_correction_term = term_1 - term_2
  return batch_score - score_correction_term


# convenience used by eq2 and eq4
_distribute = functools.partial(taylor_distribute, argnums=2)


@functools.lru_cache(maxsize=None)
def get_cox_fun(eq: str, kind: str, batch: bool = False, order: int = 1):
  """Convenience routine to get a desired cox model equation."""
  # pylint: disable=too-many-branches
  if eq in ('eq1', 'eq3'):
    order = -1
  if eq in ('eq2', 'eq4') and kind == "loglik":
    raise TypeError("Does not have loglik option")
  if kind == "robust_cox_correction_score" and not batch:
    raise TypeError("robust_cox_correction_score"
                    " does not have un-batched option")

  is_robust = kind == "robust_cox_correction_score"
  eq1_fn = globals()["eq1_{}".format(
      "batch_robust_cox_correction_score" if is_robust else kind)]

  if eq == "eq1":
    fn = eq1_fn
    if batch and not is_robust:
      fn = collect(fn, "batch_{}".format(kind))
  elif eq == "eq2":
    fn = _distribute(eq1_fn, reduction_kind="cumsum", orders={'ebx': order})
    if is_robust:
      fn = _group_by(fn)
    elif batch:
      fn = _group_by(collect(fn, "batch_{}".format(kind)))
  elif eq == "eq3":
    fn = vmap(eq1_fn, in_axes=(0, 0, None))
    if not batch:
      fn = getattr(modeling, "sum_{}".format(kind))(fn)
  elif eq == "eq4":
    if batch and not is_robust:
      fn = collect(eq1_fn, "batch_{}".format(kind))
    else:
      fn = eq1_fn
    fn = _distribute(fn,
                     reduction_kind=None if batch else "sum",
                     orders={'ebx': order})
  else:
    raise TypeError("Invalid eq")

  return fn


# eq1 (together with root modeling)
eq1_batch_loglik = collect(eq1_loglik, "batch_loglik")
eq1_batch_score = collect(eq1_score, "batch_score")

# Eq 2
eq2_score = get_cox_fun("eq2", "score", False, 1)
eq2_batch_score = get_cox_fun("eq2", "score", True, 1)
eq2_batch_robust_cox_correction_score = get_cox_fun(
    "eq2", "robust_cox_correction_score", True, 1)
eq2_hessian_taylor = get_cox_fun("eq2", "hessian", False, 1)

# Eq 3
eq3_batch_loglik = vmap(eq1_batch_loglik, in_axes=(0, 0, None))
eq3_loglik = modeling.sum_loglik(eq3_batch_loglik)
eq3_batch_robust_cox_correction_score = vmap(
    eq1_batch_robust_cox_correction_score, in_axes=(0, 0, None))

# Eq 4
eq4_score = _distribute(eq1_score, reduction_kind="sum", orders={'ebx': 1})
eq4_batch_score = _distribute(eq1_batch_score,
                              reduction_kind=None,
                              orders={'ebx': 1})
eq4_batch_robust_cox_correction_score = _distribute(
    eq1_batch_robust_cox_correction_score,
    reduction_kind=None,
    orders={'ebx': 1})
eq4_hessian_taylor = _distribute(eq1_hessian,
                                 reduction_kind="sum",
                                 orders={'ebx': 1})
