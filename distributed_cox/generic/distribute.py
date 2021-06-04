"""Distributed modeling."""
from typing import Dict, Tuple, Union

import functools

import jax.numpy as jnp
from jax import vmap
import jax.lax as lax
import jax.ops
# import numpy as np
import oryx

import distributed_cox.generic.taylor as taylor

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant
nest = oryx.core.nest


# Reduction primitives
# ------------------------------------------------------------
def cumsum(vals, *, name: str):
  """Custom cumsum for modeling."""
  vals = sow(vals, tag="pre_cumsum", name=name, mode="clobber")
  return sow(
      jnp.cumsum(vals, axis=0, dtype=None),
      tag="cumsum",
      name=name,
      mode="clobber",
  )


def sum(vals, *, name: str):  # pylint: disable=redefined-builtin
  """Custom sum for modeling."""
  vals = sow(vals, tag="pre_sum", name=name, mode="clobber")
  return sow(jnp.sum(vals, axis=0, dtype=None),
             tag="sum",
             name=name,
             mode="clobber")


# end primitives
# ------------------------------------------------------------


def distribute(fun, reduction_kind: str = "sum"):
  """Partitions a function into distributed version.

  Assuming `fun` contains invocations of the collective primitives, this
  function partitions `fun` into a composition of two functions
  `fun_partt1` and `fun_partt2`.
  The arguments to `fun_part1` are the same as `fun`; the outputs of `fun_part1`
  are the intermediate values right before the collective primitives in `fun`.

  The arguments to `fun_part2` contains two additional arguments compared to
  `fun`: `intermediates` and `group_labels`.
  `intermediates` has an additional group dimension compared to the
  `intermediates` output in `fun_part1`.
  `group_labels` is a global array containing individual group labels.

  Calling `fun_part1` on multiple sub-divisions of the original inputs,
  followed by `fun_part2` which collects all the result together, will return
  the same result as simply calling `fun`.
  """

  pt1_fun = reap(fun, tag="pre_" + reduction_kind)

  def pt2_fun(intermediates, group_labels, *args, **kwargs):
    intermediates = dict(intermediates)
    for name in intermediates:
      intermediate = intermediates[name]
      K, *_ = intermediate.shape

      if reduction_kind == "cumsum":

        def groupped_cumsum(intermediate, carry, group_label):
          group_cnts, curr_sum = carry
          group_cnt_before = group_cnts[group_label]
          val = intermediate[group_label, group_cnt_before]
          curr_sum = curr_sum + val
          group_cnts = jax.ops.index_add(group_cnts, group_label, 1)
          return (group_cnts, curr_sum), curr_sum

        _, intermediate_reduced = lax.scan(
            functools.partial(groupped_cumsum, intermediate),
            init=(
                jnp.zeros(K, dtype=jnp.int32),
                jnp.zeros(intermediate.shape[2:], dtype=intermediate.dtype),
            ),
            xs=group_labels,
        )
      elif reduction_kind == "sum":
        intermediate_reduced = jnp.sum(intermediate, axis=(0, 1))
      else:
        raise TypeError("Invalid reduction kind")

      intermediates[name] = intermediate_reduced

    return plant(fun, tag=reduction_kind)(intermediates, *args, **kwargs)

  return pt1_fun, pt2_fun


def taylor_distribute(fun,
                      *,
                      reduction_kind: str,
                      orders: Dict[str, int],
                      argnums: Union[int, Tuple[int]] = 0):
  """Taylor distributes function.

  First performs taylor expansion on ``fun``. Then, the function is broken into
  two parts based on a reduction. The reduction is defined by, for example,
  invoking :py:func:`cumsum` in ``fun``. The first part of the function is
  mapped by an additional batch axis using :py:func:`jax.vmap`, which allows
  simultanenous computation of the first part of the function across distributed
  sites. Then, the second part of the function reduces the result from those
  distributed sites, and returns the aggregated output.

  Args:
    fun: the function to be taylor expand then distributed.
    reduction_kind: the kind of the reduction. This assumes that the
      ``reduction_kind`` is used in ``fun``.
    orders: a mapping from string names of the :py:func:`taylor_expand` invoked
    in ``fun`` to their taylor expansion orders.
    argnums: the arguments with which the taylor expansion should be performed.

  Returns:
    the taylor expanded then distributed version of ``fun``.
  """

  if isinstance(argnums, int):
    argnums = (argnums,)

  @functools.wraps(fun)
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
    return pt2_fn(intermediates, group_labels, *single_args,
                  *[arg for i, arg in enumerate(single_args) if i in argnums])

  return wrapped
