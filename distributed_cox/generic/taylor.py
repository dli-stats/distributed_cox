"""Taylor Approximation through JAX's transformations."""

from typing import Union, Sequence

import functools
import operator

import jax.numpy as jnp
import jax.api as api
import jax.experimental.jet as jet
from jax.interpreters import ad
import jax.linear_util as lu

import oryx

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant

jet.jet_rules[
    oryx.core.interpreters.harvest.sow_p] = lambda *args, **kwargs: args


def taylor_approx(val, *, name):
  """Marks a term to be taylor approximated."""
  return sow(val, tag="taylor_approx", name=name, mode="clobber")


def _factorial(i):
  return functools.reduce(operator.mul, range(1, i + 1), 1)


def taylor_expand_fun(fun, argnums, order: int = 1):
  """Perform taylor expansion on fun."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = api.argnums_partial(f, argnums, args[:-len(argnums)])
    dyn_args0 = args[-len(argnums):]

    dyn_args_flat, in_tree = api.tree_flatten(dyn_args)
    dyn_args0_flat, in_tree2 = api.tree_flatten(dyn_args0)
    if in_tree != in_tree2:
      raise TypeError("Invalid input arguments for taylor expand")
    del in_tree2
    f_flat, out_tree = api.flatten_fun_nokwargs(f_partial, in_tree)

    dparams = api.safe_map(jnp.subtract, dyn_args_flat, dyn_args0_flat)
    # pylint: disable=protected-access,no-member,no-value-for-parameter
    if order == 1:
      # f0, vjp_fun = ad.vjp(f_flat, dyn_args0_flat)
      # f1 = vjp_fun(*dparams)
      f0, f1 = ad.jvp(f_flat).call_wrapped(dyn_args0_flat, dparams)
      out_val = api.safe_map(operator.add, f0, f1)
    else:
      series = [([d] + [jnp.zeros_like(d)] * (order - 1)) for d in dparams]
      f0, f_terms = jet.jet_fun(jet.jet_subtrace(f_flat),
                                order).call_wrapped(dyn_args0_flat, series)
      out_val = api.safe_map(
          lambda f0, f_terms: f0 + sum(f_terms[i] / _factorial(i + 1)
                                       for i in range(order)), f0, f_terms)
    return api.tree_unflatten(out_tree(), out_val)

  return wrapped_fun


def taylor_approx_expand(fun,
                         *,
                         name: str,
                         argnums: Union[Sequence[int], int] = 0,
                         order: int = 1):
  """Expands a function by expanding a specified marked sub-expression."""
  if isinstance(argnums, int):
    argnums = (argnums,)

  @functools.wraps(fun)
  def full_expanded_fun(*args, **kwargs):
    fun_expanded = taylor_expand_fun(reap(fun,
                                          tag="taylor_approx",
                                          allowlist=[name]),
                                     argnums,
                                     order=order)
    orig_args = args[:-len(argnums)]

    intermediates = fun_expanded(*args, **kwargs)
    return plant(fun, tag="taylor_approx",
                 allowlist=[name])(intermediates, *orig_args, **kwargs)

  return full_expanded_fun
