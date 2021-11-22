"""Taylor Approximation through JAX's transformations.

This module provides functionality to approximate a function through taylor
expansion on itself (:py:func:`taylor_expand_fun`), or, more generally, an
internal sub-expression of the function (:py:func:`taylor_approx_expand`).

For the former functionality, please refer to the documentation of
:py:func:`taylor_expand_fun`.

We will explain here the semantics of second functionality below:

Given a function, we are interested in an approximated version of
the function, by taylor expanding some of the intermediate terms the function.
Without loss of generality, we assume that the given function is

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
import warnings

import jax
import jax.numpy as jnp

from jax import tree_util as tu
from jax import api_util
from jax import linear_util as lu
from jax.experimental import jet
from jax.interpreters import ad

import oryx

sow = oryx.core.sow
reap = oryx.core.reap
nest = oryx.core.nest
plant = oryx.core.plant
tie_all = oryx.core.tie_all

# Let the jet rules know about ``sow``.
jet.jet_rules[oryx.core.interpreters.harvest.sow_p] = lambda *args, **_: args

TAYLOR_APPROX_TAG = "taylor_approx"


def taylor_approx(val, *, name):
  """Marks a term to be taylor approximated."""
  return sow(val, tag=TAYLOR_APPROX_TAG, name=name, mode="strict")


def _recursive_sow_values(sow_f, values):
  """Recurively sow all values, while respecting scoping."""
  ret = {}
  for k, v in values.items():
    if isinstance(v, dict):
      ret[k] = nest(functools.partial(_recursive_sow_values, sow_f), scope=k)(v)
    else:
      ret[k] = sow_f(k, v)
  return ret


def grad(f,
         argnums: Union[int, Sequence[int]] = 0,
         has_aux: bool = False,
         holomorphic: bool = False,
         allow_int: bool = False):
  """Computes gradient of ``f`` and preserves the taylor approximation tags.

  This function is meant to be a drop-in replacement for :py:func:`jax.grad`.
  With :py:func:`jax.grad`, the intermediates marked by :py:func:`taylor_approx`
  are "lost". This function preserves those intermediates, by essentially
  splitting ``f`` into composition of two functions (before and after the
  intermediates), and applies autodiff with chain rule.

  Please see :py:func:`jax.grad` for documentation on the arguments.
  """
  assert not has_aux, "aux not supported yet"

  if isinstance(argnums, int):
    argnums = (argnums,)

  def wrapped(*args, **kwargs):
    in_to_intermediates = reap(f, tag=TAYLOR_APPROX_TAG)

    if not in_to_intermediates:
      # There's no marked values, back off and resort to normal jax.grad
      warnings.warn(
          "It appears that you have invoked taylor-approx customized grad"
          " function but there is no taylor_approx marks in the function.")
      del in_to_intermediates
      return jax.grad(f,
                      argnums=argnums,
                      holomorphic=holomorphic,
                      allow_int=allow_int)(*args, **kwargs)

    intermediates_to_out = plant(f, tag=TAYLOR_APPROX_TAG)

    in_to_intermediates_d = jax.jacfwd(in_to_intermediates,
                                       argnums=argnums,
                                       holomorphic=holomorphic)
    intermediates_to_out_d = jax.grad(intermediates_to_out,
                                      argnums=(0, *[a + 1 for a in argnums]),
                                      holomorphic=holomorphic,
                                      allow_int=allow_int)
    sow_f = lambda name, value: taylor_approx(value, name=name)
    intermediates = in_to_intermediates(*args, **kwargs)
    # f'(g(x))
    term1 = intermediates_to_out_d(_recursive_sow_values(sow_f, intermediates),
                                   *args, **kwargs)
    # g'(x)
    term2 = _recursive_sow_values(
        sow_f, {"grad": in_to_intermediates_d(*args, **kwargs)})
    # f'(g(x)) * g'(x)
    return tu.tree_reduce(operator.add, [
        jax.tree_multimap(lambda x, ys: tuple(x @ y for y in ys), term1[0],
                          term2["grad"]), term1[1:]
    ])

  return wrapped


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
    allowlist = [name, "jvp", "grad"]  # TODO: Currently later two is not used.
    fun_expanded = taylor_expand_fun(reap(fun,
                                          tag=TAYLOR_APPROX_TAG,
                                          allowlist=allowlist),
                                     argnums,
                                     order=order)
    orig_args = args[:-len(argnums)]

    intermediates = fun_expanded(*args, **kwargs)
    return plant(fun, tag=TAYLOR_APPROX_TAG,
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
    f_partial, dyn_args = api_util.argnums_partial(f, argnums,
                                                   args[:-len(argnums)])
    dyn_args0 = args[-len(argnums):]

    dyn_args_flat, in_tree = tu.tree_flatten(dyn_args)
    dyn_args0_flat, in_tree2 = tu.tree_flatten(dyn_args0)
    if in_tree != in_tree2:
      raise TypeError("Invalid input arguments for taylor expand")
    del in_tree2
    f_flat, out_tree = api_util.flatten_fun_nokwargs(f_partial, in_tree)

    dparams = api_util.safe_map(jnp.subtract, dyn_args_flat, dyn_args0_flat)
    # pylint: disable=protected-access,no-member,no-value-for-parameter
    if order == 1:
      # f0, vjp_fun = ad.vjp(f_flat, dyn_args0_flat)
      # f1 = vjp_fun(*dparams)
      f0, f1 = ad.jvp(f_flat).call_wrapped(dyn_args0_flat, dparams)
      out_val = api_util.safe_map(operator.add, f0, f1)
    else:
      series = [([d] + [jnp.zeros_like(d)] * (order - 1)) for d in dparams]
      f0, f_terms = jet.jet_fun(jet.jet_subtrace(f_flat),
                                order).call_wrapped(dyn_args0_flat, series)
      out_val = api_util.safe_map(
          lambda f0, f_terms: f0 + sum(f_terms[i] / _factorial(i + 1)
                                       for i in range(order)), f0, f_terms)
    return api_util.tree_unflatten(out_tree(), out_val)

  return wrapped_fun
