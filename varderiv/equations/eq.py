from typing import Sequence, Union, Dict
import functools
import operator
import math

import jax.numpy as np
import jax.linear_util as lu
import jax.api as api
import jax.experimental.jet as jet
from jax.interpreters import ad
import jax.lax as lax
import jax.ops

import oryx

sow = oryx.core.sow
reap = oryx.core.reap
plant = oryx.core.plant
nest = oryx.core.nest


def taylor_approx(val, *, name):
  """Marks a term to be taylor approximated."""
  return sow(val, tag="taylor_approx", name=name)


def cumsum(vals, *args, name: str):
  """Custom cumsum for modeling."""
  return sow(np.cumsum(vals, axis=0, dtype=None), tag="cumsum_outer", name=name)


def factorial(i):
  return functools.reduce(operator.mul, range(1, i + 1), 1)


def taylor_expand_fun(fun, argnums, order: int = 1):
  """Perform taylor expansion on fun."""

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

    dparams = api.safe_map(np.subtract, dyn_args_flat, dyn_args0_flat)
    # pylint: disable=protected-access,no-member,no-value-for-parameter
    if order == 1:
      # f0, vjp_fun = ad.vjp(f_flat, dyn_args0_flat)
      # f1 = vjp_fun(*dparams)
      f0, f1 = ad.jvp(f_flat).call_wrapped(dyn_args0_flat, dparams)
      out_val = api.safe_map(operator.add, f0, f1)
    else:
      series = [([d] + [np.zeros_like(d)] * (order - 1)) for d in dparams]
      f0, f_terms = jet.jet_fun(jet.jet_subtrace(f_flat),
                                order).call_wrapped(dyn_args0_flat, series)
      out_val = api.safe_map(
          lambda f0, f_terms: f0 + sum(f_terms[i] / factorial(i + 1)
                                       for i in range(order)), f0, f_terms)
    return api.tree_unflatten(out_tree(), out_val)

  return wrapped_fun


def reap_fun(fun, *, tag, name):

  def wrapped(*args, **kwargs):
    intermediate_vals = reap(fun, tag=tag, allowlist=[name])(*args, **kwargs)
    return intermediate_vals[name]

  return wrapped


def taylor_approx_expand(fun,
                         *,
                         name: str,
                         argnums: Union[Sequence[int], int] = 0,
                         order: int = 1):
  if isinstance(argnums, int):
    argnums = (argnums,)

  def full_expanded_fun(*args, **kwargs):
    allowlist = ["jvp", "vjp", name]
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


def _model_inject_grad(fn_to_approx, name, args):
  fn_to_approx_custom_deriv = jax.custom_jvp(fn_to_approx)

  def custom_tagged_jvp(primals, tangents):
    primal_out, tangent_out = jax.jvp(fn_to_approx, primals, tangents)
    return primal_out, taylor_approx(tangent_out, name=name)

  fn_to_approx_custom_deriv.defjvp(nest(custom_tagged_jvp, scope="jvp"))
  fn_to_approx_custom_deriv = jax.custom_vjp(fn_to_approx_custom_deriv)

  wrapped_fun = lu.wrap_init(fn_to_approx)
  _, in_tree = api.tree_flatten(args)
  flat_fun, _ = api.flatten_fun_nokwargs(wrapped_fun, in_tree)
  _, pvals, jaxpr, _ = ad.linearize(flat_fun, *args)

  def custom_tagged_vjp_fwd(*primals):
    _, in_tree = api.tree_flatten(primals)
    flat_fun, out_tree = api.flatten_fun_nokwargs(wrapped_fun, in_tree)
    out_primals, _, _, consts = ad.linearize(flat_fun, *primals)
    return (api.tree_unflatten(out_tree(), out_primals), consts)

  def custom_tagged_vjp_bwd(consts, cts):
    cts, _ = api.tree_flatten(cts)
    cts = tuple(map(ad.ignore_consts, cts, pvals))
    dummy_args = [ad.UndefinedPrimal(v.aval) for v in jaxpr.invars]
    arg_cts = ad.backward_pass(jaxpr, consts, dummy_args, cts)
    res = map(ad.instantiate_zeros, arg_cts)
    res = api.tree_unflatten(in_tree, res)
    return taylor_approx(res, name=name)

  fn_to_approx_custom_deriv.defvjp(custom_tagged_vjp_fwd,
                                   nest(custom_tagged_vjp_bwd, scope="vjp"))
  return fn_to_approx_custom_deriv


def model(fun, tag="taylor_approx"):
  """High level wrapper that preserves taylor_approx tag."""

  def model_fun(*args, **kwargs):
    intermediate_vals = reap(fun, tag=tag)(*args, **kwargs)
    for name in intermediate_vals:
      fn_to_approx = reap_fun(fun, tag=tag, name=name)
      fn_to_approx_custom_deriv = _model_inject_grad(fn_to_approx, name, args)
      intermediate_vals[name] = fn_to_approx_custom_deriv(*args, **kwargs)
    return plant(fun, tag=tag)(intermediate_vals, *args, **kwargs)

  return model_fun


def distribute(fun, *, name: str):
  """Partitions a function into distributed version."""

  pt1_fun = reap_fun(fun, tag="cumsum_outer", name=name)

  def pt2_fun(intermediates, group_labels, *args, **kwargs):
    K, *_ = intermediates.shape

    def groupped_cumsum(group_cnts, group_label):
      group_cnts = jax.ops.index_add(group_cnts, group_label, 1)
      cur_sum = np.sum(np.where(group_cnts >= 0,
                                intermediates[np.arange(K), group_cnts], 0),
                       axis=0)
      return group_cnts, cur_sum

    _, s = lax.scan(groupped_cumsum,
                    np.zeros(K, dtype=np.int32) - 1, group_labels)
    return plant(fun, tag="cumsum_outer", allowlist=[name])({
        name: s
    }, *args, **kwargs)

  return pt1_fun, pt2_fun


if __name__ == "__main__":
  from jax import grad, vmap
  import varderiv.data as vdata

  @functools.partial(model, tag="cumsum_outer")
  @functools.partial(model, tag="taylor_approx")
  def eq1_log_likelihood(X, delta, beta):
    breakpoint()
    bx = np.dot(X, beta)
    ebx = np.exp(bx)
    ebx = taylor_approx(ebx, name="ebx")
    ebx_cs = cumsum(ebx, 0, name="ebx_cs")
    log_term = np.log(ebx_cs)
    return np.sum((bx - log_term) * delta, axis=0)

  approx_fn = taylor_approx_expand(grad(eq1_log_likelihood, argnums=2),
                                   argnums=2,
                                   name="ebx")
  eq2_score_fn_pt1, eq2_score_fn_pt2 = distribute(approx_fn, name="ebx_cs")

  gen = vdata.data_generator(500, 3, (166, 167, 167))
  X, delta, beta, group_labels = gen(vdata.data_generation_key)
  X_groups, delta_groups = vdata.group_data_by_labels(group_labels,
                                                      X,
                                                      delta,
                                                      K=3,
                                                      group_size=167)
  # print(approx_fn(X, delta, beta, beta))
  intermediates = vmap(eq2_score_fn_pt1,
                       in_axes=(0, 0, None, None))(X_groups, delta_groups, beta,
                                                   beta)

  # pt1f, pt2f = distribute(f, name="ex")
  # xs = np.arange(30).reshape((3, 10))
  # group_labels = np.array([[i] * 10 for i in range(3)]).reshape(30)
  # cs_pt1 = vmap(pt1f)(xs)
  # print(pt2f(cs_pt1, group_labels, np.zeros((30,))))
  # print(np.cumsum(np.exp(xs.reshape(30))))
