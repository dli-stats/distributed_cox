# """Higher order functions for creating models."""

# from typing import Union, Callable, Sequence

# import functools
# import collections
# import itertools

# import networkx as nx

# import jax.core
# import jax.linear_util as lu
# from jax import vmap, mask, make_jaxpr
# from jax.api_util import argnums_partial

# import jax.numpy as np

# from varderiv.generic.solver import solve_newton

# FitResult = collections.namedtuple("FitResult",
#                                    "guess covs loglik steps converged")

# # pylint: disable=missing-function-docstring

# def fit_fn(model_loglik_fn, **kwargs):
#   return functools.partial(solve_newton, model_loglik_fn, **kwargs)

# def fit_and_cov(model_loglik_fn, **kwargs):

#   def wrapped(initial_guess):
#     solver_result = solve_newton(model_loglik_fn, initial_guess, **kwargs)
#     cov_H = -np.linalg.inv(solver_result.hessian)
#     return FitResult(guess=solver_result.guess,
#                      covs={"cov_H": cov_H},
#                      loglik=solver_result.loglik,
#                      steps=solver_result.step,
#                      converged=solver_result.converged)

#   return wrapped

# @functools.partial(mask, in_shapes=[('(n,_)', '(n,_)'), '_'], out_shape='')
# def model(X, beta):
#   return np.sum(np.exp(np.dot(X[0], beta)))

# talyor_approx_p = jax.core.Primitive("taylor_approx_p")
# talyor_approx_p.def_abstract_eval(lambda x: x)
# talyor_approx_p.def_impl(lambda x: x)

# def taylor_approx(x):
#   return talyor_approx_p.bind(x)

# def _hash_var(var):
#   return (var.count, var.suffix)

# def _build_dataflow_graph(typed_jaxpr):
#   jaxpr = typed_jaxpr.jaxpr
#   dataflow_graph = nx.DiGraph()
#   defining_eqns_map = {}
#   for invar in itertools.chain(jaxpr.invars, jaxpr.constvars):
#     h_in_var = _hash_var(invar)
#     dataflow_graph.add_node(h_in_var)
#     defining_eqns_map[h_in_var] = h_in_var
#   for eqn_i, eqn in enumerate(jaxpr.eqns):
#     for outvar in eqn.outvars:
#       defining_eqns_map[_hash_var(outvar)] = eqn_i
#     dataflow_graph.add_node(eqn_i)
#     dataflow_graph.add_edges_from(
#         (defining_eqns_map[_hash_var(invar)], eqn_i) for invar in eqn.invars)
#   return dataflow_graph

# def _extract_subjaxpr_by_invars(typed_jaxpr, accessible_invar_idxs,
#                                 dataflow_graph):
#   all_invar_idxs = set(range(len(typed_jaxpr.jaxpr.invars)))
#   accessible_invar_idxs = set(accessible_invar_idxs)
#   unaccesible_invar_idxs = all_invar_idxs - accessible_invar_idxs

#   # Find all unaccessible nodes to be dropped
#   source = "source"
#   dataflow_graph.add_node(source)
#   dataflow_graph.add_edges_from(
#       (_hash_var(typed_jaxpr.invars[invar_idx]), source)
#       for invar_idx in unaccesible_invar_idxs)
#   drop_eqns = nx.algorithms.dag.descendants(dataflow_graph, source)
#   drop_eqns.remove(set())
#   dataflow_graph.remove_node(source)

#   for eqn in typed_jaxpr.jaxpr.eqns:
#     pass

#   ret_jaxpr = jax.core.Jaxpr(
#       constvars,
#       invars,
#   )

# def split_distributed_model(model_fn, group_vars_argnums):

#   def wrapped(*args):
#     typed_jaxpr = make_jaxpr(model_fn)(*args)
#     dataflow_graph = _build_dataflow_graph(typed_jaxpr)

#   return wrapped

# def sum(f):

#   def wrapped(*args):
#     group_args = tuple(arg.group for arg in args)
#     return np.sum(mask(f, in_shapes=None, out_shape=None)(*group_args))

#   return wrapped

# if __name__ == "__main__":
#   import jax.random as jrandom
#   from jax import jit
#   import varderiv.equations.eq1 as eq1
#   import varderiv.data as data

#   N = 500
#   K = 3
#   X_DIM = 3
#   k1, k2 = jrandom.split(jrandom.PRNGKey(0))
#   group_sizes = data.group_sizes_generator(N, K, "same")
#   T, X, delta, beta, group_labels = data.data_generator(N,
#                                                         X_DIM,
#                                                         group_sizes,
#                                                         return_T=True)(k1)
#   X_groups, delta_groups = data.group_data_by_labels(1, K, X, delta,
#                                                      group_labels)

#   eq1_fit_and_cov = jit(
#       fit_and_cov(functools.partial(eq1.eq1_log_likelihood, X, delta),
#                   max_num_steps=10))

#   print(eq1_fit_and_cov(np.ones_like(beta)))