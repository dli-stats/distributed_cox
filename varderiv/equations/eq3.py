"""Equation 3."""

import functools

import jax.numpy as np
from jax import jacfwd
from jax import random as jrandom

from varderiv.solver import solve_newton

from varderiv.data import group_data_by_labels

import varderiv.equations.eq1 as eq1

#########################################################
# BEGIN eq3
#########################################################


def eq3_log_likelihood(X_groups, delta_groups, beta):
  return np.sum(eq1.eq1_log_likelihood(
    X_groups, 
    delta_groups, 
    np.broadcast_to(beta, (X_groups.shape[0],) + beta.shape)), axis=(0,)
  )


def eq3_log_likelihood_grad(eq1_ll_grad_fn, X_groups, delta_groups, beta):
  return np.sum(eq1_ll_grad_fn(
      X_groups, delta_groups,
      np.broadcast_to(beta, (X_groups.shape[0],) + beta.shape)),
                axis=(0,))


@functools.lru_cache(maxsize=None)
def get_eq3_solver(eq1_log_likelihood_grad_fn, solver_max_steps=10, norm_stop_thres=1e-3):
  """Solves equation 3 given eq1's ll_grad function."""

  @functools.partial(np.vectorize, signature="(K,N,p),(K,N),(p)->(p),(p),()")
  def wrapped(X_groups, delta_groups, initial_guess):
    return solve_newton(functools.partial(eq3_log_likelihood_grad,
                                          eq1_log_likelihood_grad_fn, X_groups,
                                          delta_groups),
                        initial_guess,
                        sym_pos=True,
                        max_num_steps=solver_max_steps, 
                        norm_stop_thres=norm_stop_thres)

  return wrapped


def solve_eq3(X,
              delta,
              K,
              group_labels,
              X_groups=None,
              delta_groups=None,
              initial_guess=None,
              eq1_use_ad=False,
              **solver_args):
  """Solves equation 3, NOT batched."""

  if initial_guess is None:
    initial_guess = np.zeros(X.shape[-1])

  if X_groups is None or delta_groups is None:
    X_groups, delta_groups = group_data_by_labels(1, K, X, delta, group_labels)

  if eq1_use_ad:
    eq1_ll_grad_fn = eq1.eq1_log_likelihood_grad_manual
  else:
    eq1_ll_grad_fn = eq1.eq1_log_likelihood_grad_ad

  sol = get_eq3_solver(eq1_ll_grad_fn, **solver_args)(X_groups, delta_groups, initial_guess)

  return sol


#########################################################
# BEGIN eq3 cov
#########################################################

eq3_compute_H = jacfwd(eq3_log_likelihood_grad, -1)


@functools.lru_cache(maxsize=None)
def get_eq3_cov_fn(eq1_ll_grad_fn):

  @functools.partial(np.vectorize, signature="(K,N,p),(K,N),(p)->(p,p)")
  def wrapped(X_groups, delta_groups, beta):
    H = eq3_compute_H(eq1_ll_grad_fn, X_groups, delta_groups, beta)
    return np.linalg.inv(-H)

  return wrapped
