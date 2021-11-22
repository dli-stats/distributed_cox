"""Patch to compute value, jacobian and hessian in one shot."""

from typing import Callable, Tuple, Any, Union, Sequence
import functools

import jax.linear_util as lu
from jax import vmap
from jax.tree_util import (tree_map, tree_structure, tree_transpose)
from jax._src.api import (_check_input_dtype_jacrev, _check_output_dtype_jacrev,
                          _vjp, _unravel_array_into_pytree, _std_basis,
                          _check_callable, tree_flatten, safe_zip, _dtype,
                          flatten_fun_nokwargs, flatten_fun_nokwargs2,
                          tree_unflatten, _check_input_dtype_jacfwd,
                          _check_output_dtype_jacfwd)

from jax.api_util import argnums_partial
from jax.util import partial
from jax.interpreters import ad

# pylint: disable=no-else-return, missing-function-docstring


def jacrev_and_value(fun, argnums=0, holomorphic=False, allow_int=False):
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int),
             dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    jac = tree_transpose(tree_structure(example_args), tree_structure(y), jac)
    return jac, y

  return jacfun


def jvp(fun: Callable, primals, tangents, **kwargs) -> Tuple[Any, Any]:
  _check_callable(fun)
  return _jvp(lu.wrap_init(fun), primals, tangents, **kwargs)


def _jvp(fun: lu.WrappedFun, primals, tangents, **kwargs):
  """Variant of jvp() that takes an lu.WrappedFun."""
  has_aux = kwargs.pop('has_aux', False)

  if (not isinstance(primals, (tuple, list)) or
      not isinstance(tangents, (tuple, list))):
    msg = ("primal and tangent arguments to jax.jvp must be tuples or lists; "
           "found {} and {}.")
    raise TypeError(msg.format(type(primals).__name__, type(tangents).__name__))

  ps_flat, tree_def = tree_flatten(primals)
  ts_flat, tree_def_2 = tree_flatten(tangents)
  if tree_def != tree_def_2:
    msg = ("primal and tangent arguments to jax.jvp must have the same tree "
           "structure; primals have tree structure {} whereas tangents have "
           "tree structure {}")
    raise TypeError(msg.format(tree_def, tree_def_2))
  for p, t in safe_zip(ps_flat, ts_flat):
    if _dtype(p) != _dtype(t):
      msg = ("primal and tangent arguments to jax.jvp must have equal types; "
             "type mismatch primal {} vs tangent {}")
      raise TypeError(msg.format(_dtype(p), _dtype(t)))
  if not has_aux:  # pylint:disable=no-else-return
    flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
    out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)  # pylint: disable=no-member
    return (tree_unflatten(out_tree(), out_primals), \
            tree_unflatten(out_tree(), out_tangents))
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, tree_def)
    jvp_fun, aux = ad.jvp(flat_fun, has_aux=True)
    out_primals, out_tangents = jvp_fun.call_wrapped(ps_flat, ts_flat)  # pylint: disable=no-member
    out_tree, aux_tree = out_aux_trees()
    out_primals = tree_unflatten(out_tree, out_primals)
    out_tangents = tree_unflatten(out_tree, out_tangents)
    aux = tree_unflatten(aux_tree, aux())
    return out_primals, out_tangents, aux


def value_and_jacfwd(fun: Callable,
                     argnums: Union[int, Sequence[int]] = 0,
                     holomorphic: bool = False,
                     has_aux: bool = False) -> Callable:
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    pushfwd = partial(functools.partial(_jvp, has_aux=has_aux), f_partial,
                      dyn_args)
    if not has_aux:
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    else:
      y, jac, aux = vmap(pushfwd,
                         out_axes=(None, -1, None))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)
    if not has_aux:
      return y, jac
    else:
      return y, jac, aux

  return jacfun


def value_jac_and_hessian(fun, argnums=0, holomorphic=False):

  def wrapped(*args, **kwargs):
    jac, hessian, value = value_and_jacfwd(jacrev_and_value(
        fun, argnums=argnums, holomorphic=holomorphic),
                                           argnums=argnums,
                                           holomorphic=holomorphic,
                                           has_aux=True)(*args, **kwargs)

    return value, jac, hessian

  return wrapped


if __name__ == "__main__":
  import jax.numpy as jnp

  def f(x1, x2):
    return x1 + x2[0] + x2[1]

  x = jnp.array(1.)
  y = jnp.array([1., 2.])

  print(value_jac_and_hessian(f, argnums=(0, 1))(x, y))
