"""Automate Distributed."""
from typing import Sequence, Union, Callable, Optional, Iterable

import functools

import jax.core as core
import jax.lax as lax
from jax.interpreters import xla, ad, batching
import jax.interpreters.partial_eval as pe
from jax import source_info_util
import jax.api
import jax.tree_util as tu
import jax.linear_util as lu
from jax.api_util import safe_map, flatten_fun, wraps
from jax.experimental import callback

import jax.numpy as np
from jax import make_jaxpr, grad, jacfwd, jacrev, pmap

dist_prim_lookup = {}


def distributed_prim(name: str, prim: core.Primitive):
  dist_prim = core.Primitive(name)
  # After distributed transformation, dist_prim is transformed into two steps
  # This is the second step that is ran by master process
  dist_prim_2 = core.Primitive(name + "_2")

  dist_prim.def_impl(prim.bind)
  dist_prim_2.def_impl(prim.bind)

  dist_prim.def_abstract_eval(prim.abstract_eval)
  dist_prim_2.def_abstract_eval(prim.abstract_eval)

  # def dist_prim_2_partial_eval_rule(trace: pe.JaxprTrace, *tracers, **params):
  #   avals = [t.aval for t in tracers]
  #   out_aval = dist_prim_2.abstract_eval(*avals, **params)
  #   source = source_info_util.current()
  #   out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(out_aval), None)
  #   out_tracer.recipe = pe.new_eqn_recipe(tracers, [out_tracer], dist_prim_2,
  #                                         params, source)
  #   return out_tracer

  # pe.custom_partial_eval_rules[dist_prim_2] = dist_prim_2_partial_eval_rule

  def translation_rule(*args, **kwargs):
    return xla.translations[prim](*args, **kwargs)

  xla.translations[dist_prim] = translation_rule
  xla.translations[dist_prim_2] = translation_rule

  def jvp_rule(*args, **kwargs):
    return ad.primitive_jvps.get(prim)(*args, **kwargs)

  ad.primitive_jvps[dist_prim] = jvp_rule
  ad.primitive_jvps[dist_prim_2] = jvp_rule

  def vjp_rule(*args, **kwargs):
    return ad.primitive_transposes.get(prim)(*args, **kwargs)

  ad.primitive_transposes[dist_prim] = vjp_rule
  ad.primitive_transposes[dist_prim_2] = vjp_rule

  def batching_rule(*args, **kwargs):
    return batching.primitive_batchers.get(prim)(*args, **kwargs)

  batching.primitive_batchers[dist_prim] = batching_rule
  batching.primitive_batchers[dist_prim_2] = batching_rule

  dist_prim_lookup[dist_prim] = (prim, dist_prim_2)

  return dist_prim


distributed_sum_p = distributed_prim("distributed_sum", lax.reduce_sum_p)


def rewrite_with_distributed(fun: Callable, K: int = 1) -> Callable:

  def rewrite_callback(prim: core.Primitive, in_vals, params):
    if prim in dist_prim_lookup:
      orig_prim, dist_prim_2 = dist_prim_lookup[prim]
      out_vals = prim.bind(*in_vals, **params)
      if not orig_prim.multiple_results:
        out_vals = [out_vals]
      out_vals = [v.reshape((K,) + v.aval.shape) for v in out_vals]
      return dist_prim_2.bind(*out_vals, **params)
    return prim.bind(*in_vals, **params)

  return callback.callback_transform(fun, rewrite_callback)


if __name__ == "__main__":

  def distributed_sum(x: lax.lax.Array, axes: Sequence[int]):
    return distributed_sum_p.bind(x, axes=axes)

  jaxpr = make_jaxpr(
      rewrite_with_distributed(lambda x, y: y + distributed_sum(x * 2, (0,))))(
          np.ones(10), np.ones(10))

  # print(jaxpr)
  print(pe.partial_eval_jaxpr(jaxpr, [True, False], True, None))
  # # print(jaxpr)
