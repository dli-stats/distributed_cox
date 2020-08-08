"""Taylor Approximation through JAX's transformations."""

from typing import Callable, Union, Sequence, Dict, Any
import functools
import math

import jax.core as core
from jax.interpreters import ad, xla
from jax.util import partial
import jax.api_util as api_util
import jax.linear_util as lu
import jax.tree_util as tu
from jax.util import safe_map
from jax.api import _dtype, _vjp
from jax import source_info_util
import jax.interpreters.partial_eval as pe
import jax.numpy as np
from jax import jacfwd, make_jaxpr, grad, jacrev, vjp, vmap
import jax.experimental.jet as jet

taylor_approx_call_p = core.CallPrimitive("taylor_approx_call")


def taylor_approx_p_impl(
    fun: lu.WrappedFun,
    *args,
    #  order=1,
    #  expand=False,
    **kwargs):
  """Taylor approximate call primitive.

  Applies taylor expansion to the wrapped function if expand=True."""
  # assert order >= 1, "Taylor expansion order must be higher than 1"
  # if expand:
  #   original_args, taylor_args = args[:len(args) // 2], args[len(args) // 2:]
  #   diff = safe_map(np.subtract, original_args, taylor_args)
  #   # if order == 1:
  #   #   breakpoint()
  #   #   f0, vjp_fun = _vjp(fun, *taylor_args)
  #   #   f1 = vjp_fun(diff)
  #   #   return safe_map(np.add, f0, f1)
  #   #   # return fun.call_wrapped(*taylor_args)
  #   # else:
  #   series = [([d] + [np.zeros_like(d)] * (order - 1)) for d in diff]

  #   def wrapped(*args):
  #     return fun.call_wrapped(*args)[0]

  #   f0, taylor_terms = jet.jet(wrapped, taylor_args, series)
  #   breakpoint()
  #   ret = f0 + sum(
  #       taylor_terms[i] / math.factorial(i + 1) for i in range(order))
  #   return (ret,)

  return fun.call_wrapped(*args)


taylor_approx_call_p.def_impl(taylor_approx_p_impl)
ad.primitive_transposes[taylor_approx_call_p] = partial(ad.call_transpose,
                                                        taylor_approx_call_p)

# def call_translation(*args, tag=None, **kwargs):
#   return xla.call_translations[xla.xla_call_p](*args,
#                                                donated_invars=None,
#                                                **kwargs)

# xla.call_translations[taylor_approx_call_p] = call_translation


def taylor_approx(fun: Callable, tag=None) -> Callable:

  @api_util.wraps(fun)  # pylint: disable=no-value-for-parameter
  def wrapped(*args, **kwargs):
    args_flat, in_tree = tu.tree_flatten((args, kwargs))
    flat_fun, out_tree = api_util.flatten_fun(lu.wrap_init(fun), in_tree)
    out_flat = taylor_approx_call_p.bind(flat_fun,
                                         *args_flat,
                                         name=flat_fun.__name__,
                                         tag=tag)
    return tu.tree_unflatten(out_tree(), out_flat)

  return wrapped


def taylor_approx_expand(fun: Callable,
                         taylor_argnums: Union[int, Sequence[int]] = 0,
                         orders: Dict[str, int] = {}):
  if isinstance(taylor_argnums, int):
    taylor_argnums = (taylor_argnums,)

  def wrapped(*args):
    original_args = tuple(
        arg[0] if i in taylor_argnums else arg for i, arg in enumerate(args))
    taylor_args = tuple(
        arg[1] if i in taylor_argnums else arg for i, arg in enumerate(args))

    wrapped_fun = lu.wrap_init(fun)
    original_args_flat, in_tree = tu.tree_flatten((original_args, {}))
    taylor_args_flat, in_tree_2 = tu.tree_flatten((taylor_args, {}))
    if in_tree != in_tree_2:
      raise TypeError("Original arguments must match with taylor arguments")
    fun_flat, out_tree = api_util.flatten_fun(wrapped_fun, in_tree)
    in_pvals = tuple((pe.PartialVal.unknown(pe.get_aval(origin_arg)))
                     for origin_arg in original_args_flat)
    jaxpr, _, consts = pe.trace_to_jaxpr(fun_flat, in_pvals)
    out_vals = _eval_jaxpr_taylor_expand(
        jaxpr, consts, *(original_args_flat + taylor_args_flat), orders=orders)
    return tu.tree_unflatten(out_tree(), out_vals)

  return wrapped


def _eval_jaxpr_taylor_expand(jaxpr: core.Jaxpr,
                              consts,
                              *args,
                              orders: Dict[str, int] = {}):

  def read(v):
    if type(v) is core.Literal:
      return (v.val, v.val)
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  original_args, taylor_args = args[:len(args) // 2], args[len(args) // 2:]

  args = tuple(zip(original_args, taylor_args))
  del original_args, taylor_args

  env: Dict[core.Var, Any] = {}
  write(core.unitvar, (core.unit, core.unit))
  safe_map(write, jaxpr.constvars, tuple((c, c) for c in consts))
  safe_map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:

    call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)

    original_in_vals, taylor_in_vals = zip(*safe_map(read, eqn.invars))
    if call_jaxpr:
      if eqn.primitive == taylor_approx_call_p:
        order = orders.get(params.get('tag', None), 1)

        def eval_call(*args):
          return core.eval_jaxpr(call_jaxpr, (), *args)

        diff = safe_map(np.subtract, original_in_vals, taylor_in_vals)
        series = [([d] + [np.zeros_like(d)] * (order - 1)) for d in diff]
        f0, taylor_terms = jet.jet(eval_call, taylor_in_vals, series)
        ans = [
            f0 + sum(taylor_terms[j][i] / math.factorial(i + 1)
                     for i in range(order))
            for j, f0 in enumerate(f0)
        ]
        ans = tuple(zip(ans, ans))
      else:
        subfuns = (lu.wrap_init(
            partial(_eval_jaxpr_taylor_expand, call_jaxpr, (), orders=orders)),)
        with source_info_util.user_context(eqn.source_info):
          ans = eqn.primitive.bind(
              *(subfuns + original_in_vals + taylor_in_vals), **params)
    else:
      with source_info_util.user_context(eqn.source_info):
        ans0 = eqn.primitive.bind(*original_in_vals, **params)
        ans1 = eqn.primitive.bind(*taylor_in_vals, **params)
        if eqn.primitive.multiple_results:
          ans = tuple(zip(ans0, ans1))
        else:
          ans = (ans0, ans1)

    if eqn.primitive.multiple_results:
      safe_map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)

  return tuple(read(outvar)[0] for outvar in jaxpr.outvars)


def jvp_taylor(fun, primals, series):
  # Computes the Taylor series the slow way, with nested jvp.
  order, = set(map(len, series))

  def composition(eps):
    taylor_terms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1)
             for i in range(len(terms))])
        for terms in series
    ]
    nudged_args = [
        (x + t).astype(x.dtype) for x, t in zip(primals, taylor_terms)
    ]
    return fun(*nudged_args)

  primal_out = fun(*primals)
  terms_out = [repeated(jacfwd, i + 1)(composition)(0.) for i in range(order)]
  return primal_out, terms_out


def repeated(f, n):

  def rfun(p):
    return functools.reduce(lambda x, _: f(x), range(n), p)

  return rfun


import jax.lax as lax


def fact(n):
  return lax.exp(lax.lgamma(n + 1.))


if __name__ == "__main__":

  def g(X, beta):

    @functools.partial(taylor_approx, tag="a")
    def f(alpha):
      return np.exp(np.dot(X, alpha))

    return np.sum(f(np.sin(beta)), axis=0)

  import numpy as onp
  X = onp.random.uniform(size=(10, 3))
  beta = np.array([1, 2, 2.])
  beta0 = np.array([1, 2, 3.])
  print(make_jaxpr(jacfwd(g, argnums=1))(X, beta))
  # g1 = jacrev(g, argnums=1)
  # print((taylor_approx_expand(g, taylor_argnums=1,
  #                             orders={'a': 1}))(X, (beta, beta0)))

  # f = lambda x: x**3
  # f1 = lambda x: 3 * x**2
  # f2 = lambda x: 6 * x
  # f3 = lambda x: 6
  # print(
  #     jvp_taylor(f, [np.array(0.)], [[
  #         np.array(1.),
  #         np.array(0.),
  #         np.array(0.),
  #     ]]))
  # print(f(0.) + f1(0.) + f2(0) / 2 + f3(0) / 6)
