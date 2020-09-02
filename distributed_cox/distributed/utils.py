"""Distributed utilities."""

import jax.numpy as np
import jax.experimental.loops as loops
import jax.ops as ops

# pylint: disable=redefined-outer-name


def searchsorted2(a, v):
  """Assumes inputs are sorted, returns the insertion indices.

  Args:
    - a: array of shape (A, )
    - v: array of shape (V, )

  Returns:
    idxs, integer array of shape (V, )
  """
  with loops.Scope() as scope:
    scope.cur_v_idx = 0
    scope.ret = np.zeros_like(v, dtype=np.int32)
    for i in scope.range(len(a)):
      for _ in scope.cond_range(a[i] >= v[scope.cur_v_idx]):
        scope.ret = ops.index_update(scope.ret, scope.cur_v_idx, i)
        scope.cur_v_idx += 1
    return scope.ret
