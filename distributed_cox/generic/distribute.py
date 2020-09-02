"""Distributed modeling."""
from typing import Dict

import functools

import jax.numpy as np
from jax import vmap
import jax.lax as lax
import jax.ops

import oryx

import distributed_cox.generic.taylor as taylor

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant
nest = oryx.core.nest


def cumsum(vals, *, name: str):
  """Custom cumsum for modeling."""
  return sow(np.cumsum(vals, axis=0, dtype=None),
             tag="cumsum",
             name=name,
             mode="clobber")


def sum(vals, *, name: str):  # pylint: disable=redefined-builtin
  """Custom sum for modeling."""
  return sow(np.sum(vals, axis=0, dtype=None),
             tag="sum",
             name=name,
             mode="clobber")


def reap_fun(fun, *, tag, name):

  def wrapped(*args, **kwargs):
    intermediate_vals = reap(fun, tag=tag, allowlist=[name])(*args, **kwargs)
    return intermediate_vals[name]

  return wrapped


def distribute(fun, reduction_kind="sum"):
  """Partitions a function into distributed version."""

  pt1_fun = reap(fun, tag=reduction_kind)

  def pt2_fun(intermediates, group_labels, *args, **kwargs):
    intermediates = dict(intermediates)
    for name in intermediates:
      intermediate = intermediates[name]
      K, *_ = intermediate.shape

      if reduction_kind == "cumsum":

        def groupped_cumsum(intermediate, group_cnts, group_label):
          K, *_ = intermediate.shape
          group_cnts = jax.ops.index_add(group_cnts, group_label, 1)
          cur_sum = np.sum(
              (group_cnts >= 0).reshape((K,) + (1,) *
                                        (len(intermediate.shape) - 2)) *
              intermediate[np.arange(K), group_cnts],
              axis=0)
          return group_cnts, cur_sum

        _, intermediate_reduced = lax.scan(
            functools.partial(groupped_cumsum, intermediate),
            np.zeros(K, dtype=np.int32) - 1, group_labels)
      elif reduction_kind == "sum":
        intermediate_reduced = np.sum(intermediate, axis=0)
      else:
        raise TypeError("Invalid reduction kind")

      intermediates[name] = intermediate_reduced

    return plant(fun, tag=reduction_kind)(intermediates, *args, **kwargs)

  return pt1_fun, pt2_fun


def taylor_distribute(fun, *, reduction_kind, orders: Dict[str, int],
                      argnums=0):
  """Taylor distributes function."""

  if isinstance(argnums, int):
    argnums = (argnums,)

  def wrapped(*args, **kwargs):
    approx_fn = functools.partial(fun, **kwargs)
    for name, order in orders.items():
      approx_fn = taylor.taylor_approx_expand(approx_fn,
                                              argnums=argnums,
                                              name=name,
                                              order=order)
    if reduction_kind is not None:
      pt1_fn, pt2_fn = distribute(approx_fn, reduction_kind=reduction_kind)
    else:
      pt1_fn, pt2_fn = approx_fn, None

    n_single_args = len(args) // 2
    single_args = args[:n_single_args]
    group_labels = args[n_single_args]
    dist_args = args[n_single_args + 1:]

    in_axes = tuple(None if i in argnums else 0
                    for i in range(len(single_args) + len(argnums)))
    pt1_fn = vmap(pt1_fn, in_axes=in_axes)

    diff_args = [arg for i, arg in enumerate(dist_args) if i in argnums]
    pt1_args = [
        single_args[i] if i in argnums else arg
        for i, arg in enumerate(dist_args)
    ] + diff_args
    intermediates = pt1_fn(*pt1_args)

    if reduction_kind is None:
      return intermediates
    else:
      return pt2_fn(intermediates, group_labels, *single_args,
                    *[arg for i, arg in enumerate(single_args) if i in argnums])

  return wrapped
