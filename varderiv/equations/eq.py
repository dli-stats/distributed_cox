import functools

import jax.numpy as np
from jax import jacfwd, grad, vmap, jit, make_jaxpr
from jax.core import eval_jaxpr
import jax.tree_util as tu
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
import jax.api as api_util

import varderiv.generic.tagging as tagging
import varderiv.generic.taylor_approx as taylor_approx


def eq1_log_likelihood(scope, X, delta, beta):
  bx = np.dot(X, beta)

  def ebxf(beta):
    return np.exp(np.dot(X, beta))

  ebx = taylor_approx.taylor_approx(ebxf)(beta)
  ebx_cs = tagging.tag(np.cumsum(ebx, 0), scope, name="cumsum")
  log_term = np.log(ebx_cs)
  return tagging.tag(np.sum((bx - log_term) * delta, axis=0), scope, name="sum")


def inline_call(fun):

  def wrapped(*args):
    wrapped_fun = lu.wrap_init(fun)
    args_flat, in_tree = tu.tree_flatten((args, {}))
    fun_flat, out_tree = api_util.flatten_fun(wrapped_fun, in_tree)
    in_pvals = tuple((pe.PartialVal.unknown(pe.get_aval(origin_arg)))
                     for origin_arg in args_flat)
    jaxpr, _, consts = pe.trace_to_jaxpr(fun_flat, in_pvals)
    out_vals = eval_jaxpr(jaxpr, consts, *args)
    return tu.tree_unflatten(out_tree(), out_vals)

  return wrapped


def eq4_score_fn(X, delta, beta, X_groups, delta_groups, beta_k_hat):
  scope = tagging.Scope()

  def wrapped_fun(X, delta, beta):
    _, collected = tagging.collect(eq1_log_likelihood)(scope, X, delta, beta)
    return collected['sum']

  wrapped_fun = taylor_approx.taylor_approx_expand(wrapped_fun,
                                                   2,
                                                   orders={"ebx": 1})

  def wrapped_fun3(X, delta, beta, beta_k):
    breakpoint()
    return wrapped_fun(X, delta, (beta, beta_k))

  pt1_batch_loglik = vmap(wrapped_fun3,
                          in_axes=(0, 0, None, 0))(X_groups, delta_groups, beta,
                                                   beta_k_hat)
  pt1_batch_score = vmap(jacfwd(wrapped_fun3, argnums=2),
                         in_axes=(0, 0, None, 0))(X_groups, delta_groups, beta,
                                                  beta_k_hat)

  def wrapped_fun2(pt1_score, beta):
    return tagging.inject(eq1_log_likelihood, {'sum': pt1_score})(scope, X,
                                                                  delta, beta)

  ds, db = grad(wrapped_fun2, argnums=(0, 1))(pt1_batch_loglik)
  return np.dot(ds, pt1_batch_score) + db


if __name__ == "__main__":
  import varderiv.data as vdata
  import varderiv.generic.modeling as modeling
  import varderiv.equations.eq1 as eq1

  X, delta, beta, group_labels = vdata.data_generator(500, 3, (166, 167, 167))(
      vdata.data_generation_key)
  X_groups, delta_groups = vdata.group_data_by_labels(group_labels,
                                                      X,
                                                      delta,
                                                      K=3,
                                                      group_size=167)
  scope = tagging.Scope()
  sol = vmap(modeling.solve_single(eq1.eq1_log_likelihood),
             in_axes=(0, 0, None))(X_groups, delta_groups, beta)
  beta_k_hat = sol.guess
  score = eq4_score_fn(X, delta, beta, X_groups, delta_groups, beta_k_hat)
  print(score)