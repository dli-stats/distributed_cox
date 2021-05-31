"""Taylor Approximation through JAX's transformations.

This module provides functionality to approximate a function through taylor
expansion on itself (:py:func:`taylor_expand_fun`), or, more generally, an
internal sub-expression of the function (:py:func:`taylor_approx_expand`).

For the former functionality, please refer to the documentation of
:py:func:`taylor_expand_fun`.

We will explain here the semantics of second functionality below:

WLOG, Given a function, we are interested in an approximated version of
the function, by taylor expanding some of the intermediate terms the function.
Without loss of generality, assume that the given function is

.. math::
  f(x, y) = g(h(x), x, y)

and we would like to obtain the approximated version

.. math::
  \\tilde{f}(x, y, x_0) = g(\\bar{h^i}(x, x_0), x, y)

where we used :math:`\\bar{h^i}(x, x_0)` to denote the approximated version of
:math:`h(x)` by taylor expanding a taylor series around :math:`x_0`, up to order
:math:`i`.

The function :py:func:`taylor_approx_expand` essentially implements the
transformation :math:`f \\mapsto \\tilde{f}`. To use the function, one needs to
inform the transformation which part of the input function corresponds to
:math:`h`. This is done by invoking a special marker primitive
:py:func:`taylor_approx`.

Example:
  One first defines the to-be-approximated function::

    def f(x, y):
      sin_x = taylor_approx(np.sin(x), "sin_x")
      return sin_x + y

  In this case, our :math:`h(x)` is :math:`sin(x)`. Now perform the expansion::

    f_approx = taylor_approx_expand(f, name="sin_x", argnums=0, order=1)

  This obtains us :math:`\\tilde{f}`, which is equivalent to::

    def f_approx_2(x, y, x0):
      sin_x = np.sin(x0) + np.cos(x0) * (x - x0) # taylor expand to approximate
      return sin_x + y

  Except that we obtained `f_approx` automatically.
  We can also verify its corretness::

    assert np.allclose(f_approx(0.1, 10, 0), f_approx_2(0.1, 10, 0)) == True

For more detail on the functions', please see their corrresponding docstrings.
"""

from typing import Callable, Union, Sequence, Tuple

import functools
import operator

import jax
import jax.numpy as jnp
import jax.api as api
import jax.experimental.jet as jet
from jax.interpreters import ad
import jax.linear_util as lu

import oryx

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant

# Let the jet rules know about ``sow``.
jet.jet_rules[oryx.core.interpreters.harvest.sow_p] = lambda *args, **_: args


def _sow_with_jvp(x, *, tag, name, **kwargs):
  """Sow and preserve the sown value's jvp."""

  @jax.custom_jvp
  def custom_sow(x):
    return sow(x, tag=tag, name=name, **kwargs)

  @custom_sow.defjvp
  def custom_jvp(primals, tangents):
    x, = primals
    g, = tangents
    g = sow(g, tag=tag, name=f'{name}_jvp', **kwargs, key=x)
    return custom_sow(x), g

  return custom_sow(x)


def taylor_approx(val, *, name):
  """Marks a term to be taylor approximated."""
  return _sow_with_jvp(val, tag="taylor_approx", name=name, mode="strict")


def _factorial(i):
  return functools.reduce(operator.mul, range(1, i + 1), 1)


def taylor_approx_expand(fun: Callable,
                         *,
                         name: str,
                         argnums: Union[Sequence[int], int] = 0,
                         order: int = 1):
  """Expands a marked function by expanding a specified marked sub-expression.

  Given a function ``fun`` that internally calls :py:func:`taylor_approx`,
  taylor expands the marked value with respect to the arguments of ``fun``.

  This attempts to be compatible with :py:mod:`jax`'s transformation system.
  Specifically, applying :py:func`taylor_approx_expand` first, and then
  transformations such as :py:func:`jax.grad`, :py:func:`jax.vmap`,
  :py:func:`jax.jit` should work as expected. On the other hand, applying
  :py:func:`jax.jacfwd`, followed by :py:func:`taylor_approx_expand` also works.

  For the semantics of the taylor approximation expansion, please see the module
  docstring :py:mod:`distributed_cox.generic.taylor`.

  Args:
    fun: the function to be expanded.
    name: the tag of the marked value to be expanded.
    argnums:  An int or collection of ints to that which position arguments the
      taylor expansion should be performed with respect to.
    order: the order the of the taylor expansion.

  Returns:
    The taylor expanded function. The function has ``|argnums|`` more arguments
    (appended to the last), and has the same return type as the input function.
    For more detail of the returned function's signature, please check
    :py:func:`taylor_expand_fun`.
  """
  if isinstance(argnums, int):
    argnums = (argnums,)

  @functools.wraps(fun)
  def full_expanded_fun(*args, **kwargs):
    allowlist = [name, f"{name}_jvp"]
    fun_expanded = taylor_expand_fun(reap(fun,
                                          tag="taylor_approx",
                                          allowlist=allowlist),
                                     argnums,
                                     order=order)
    orig_args = args[:-len(argnums)]

    intermediates = fun_expanded(*args, **kwargs)
    return plant(fun, tag="taylor_approx",
                 allowlist=allowlist)(intermediates, *orig_args, **kwargs)

  return full_expanded_fun


def taylor_expand_fun(fun: Callable,
                      argnums: Tuple[int],
                      order: int = 1) -> Callable:
  """Performs taylor expansion on ``fun``.

  Performs taylor expansion on ``fun``, with respect to argument specified in
  ``argnums``, and with expansion order of ``order``.
  Assuming that ``fun`` has signature :math:`(x_0, ... x_k) \\rightarrow y`, the
  returned function has the signature
  :math:`(x_0, ... x_k, x^0_0, ... x^0_{|\\text{argnums}|}) \\rightarrow y`,
  where :math:`x^0_{i}` are the arguments' values around which the taylor
  expansion is performed.

  Note:
    When the expansion ``order`` is greater than 1, the function uses
    :py:mod:`jax.experimental.jet` to calculate higher order expansions.
    Since :py:mod:`jax.experimental.jet` is experimental, ``order > 1`` should
    also be considered as experimental as well.

  Args:
    fun: the function to be expanded.
    argnums: a tuple sequence of integer arguments positions.
    order: the expansion order of the taylor expansion.
  """

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