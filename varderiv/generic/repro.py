import jax.core as core
import jax.interpreters.partial_eval as pe
from jax import source_info_util
from jax import make_jaxpr
import jax.numpy as np
import jax.lax as lax

yield_p = core.Primitive("yield_prim")
yield_p.def_impl(lambda x: x)
yield_p.def_abstract_eval(lambda x: x)


def do_yield(x):
  return lax.tie_in(x, yield_p.bind(x))


def yield_p_partial_eval_rule(trace: pe.JaxprTrace, tracer: pe.JaxprTracer,
                              **params):
  # const = tracer.pval.get_known()
  # if const is not None:
  #   return yield_p.bind(const, **params)
  tracer = trace.instantiate_const(tracer)
  out_aval = yield_p.abstract_eval(tracer.aval, **params)
  source = source_info_util.current()
  out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(out_aval), None)
  out_tracer.recipe = pe.new_eqn_recipe([tracer], [out_tracer], yield_p, params,
                                        source)
  return out_tracer


pe.custom_partial_eval_rules[yield_p] = yield_p_partial_eval_rule


def f1(x):
  return do_yield(x) * 2


def f2(x):
  return do_yield(np.sin(x)) * 2


def test_yield_func(f):
  print('-----' + f.__name__ + "-----")
  jaxpr = make_jaxpr(f)(1.)
  (jaxpr_known, jaxpr_unknown,
   out_unknowns) = pe.partial_eval_jaxpr(jaxpr, [False], True, None)

  print("jaxpr:", jaxpr)
  print("jaxpr_known:", jaxpr_known)
  print("jaxpr_unknown:", jaxpr_unknown)
  print("out_unknowns:", out_unknowns)
  print("")


test_yield_func(f1)
test_yield_func(f2)
