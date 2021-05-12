"""Newton solver."""

from typing import Optional, Callable

import collections
import functools

import jax.lax
import jax.numpy as jnp
import jax.scipy as scipy
from jax import value_and_grad

from distributed_cox.generic.hess import value_jac_and_hessian, value_and_jacfwd

NewtonSolverResult = collections.namedtuple(
    "NewtonSolverResult", "guess loglik score hessian step converged")


def do_halving(args):
  state, _ = args
  halving_ = state.halving + 1
  new_guess_ = ((state.new_guess + halving_ * state.guess) / (halving_ + 1.0))
  return state._replace(new_guess=new_guess_, halving=halving_)


def do_normal_update(use_likelihood, args):
  if use_likelihood:
    state, (new_loglik, new_score, _, cho_factor, _, _) = args
    state = state._replace(loglik=new_loglik)
  else:
    state, (new_score, _, cho_factor, _) = args

  u = scipy.linalg.cho_solve((cho_factor, False), new_score)
  new_guess_ = state.new_guess + u
  return state._replace(guess=state.new_guess, new_guess=new_guess_, halving=0)


def do_converged(use_likelihood, args):
  if use_likelihood:
    state, (new_loglik, new_score, new_hessian, _, _, _) = args
    state = state._replace(loglik=new_loglik)
  else:
    state, (new_score, new_hessian, _, _) = args
  return state._replace(guess=state.new_guess,
                        score=new_score,
                        hessian=new_hessian,
                        converged=True)


def do_work(use_likelihood, args):
  if use_likelihood:
    is_finite = args[1][-2]
    loglik_increased = args[1][-1]
    should_do_normal_update = jnp.logical_and(is_finite, loglik_increased)
  else:
    is_finite = args[1][-1]
    should_do_normal_update = is_finite
  state = jax.lax.cond(should_do_normal_update,
                       functools.partial(do_normal_update, use_likelihood),
                       do_halving,
                       operand=args)
  return state._replace(step=state.step + 1)


def solve_newton(likelihood_or_score_fn,
                 initial_guess,
                 use_likelihood=True,
                 hessian_fn: Optional[Callable] = None,
                 loglik_eps=1e-6,
                 score_norm_eps=1e-3,
                 max_num_steps=10) -> NewtonSolverResult:
  """HOF for newton's method solver."""

  if use_likelihood:
    if hessian_fn is None:
      value_jac_and_hessian_fn = value_jac_and_hessian(likelihood_or_score_fn)
    else:
      value_jac_and_hessian_fn = lambda guess: (value_and_grad(
          likelihood_or_score_fn)(guess) + (hessian_fn(guess),))
    InternalState = collections.namedtuple(
        "InternalState",
        "guess new_guess loglik score hessian step halving converged")
  else:
    if hessian_fn is None:
      jac_and_hessian_fn = value_and_jacfwd(likelihood_or_score_fn)
    else:
      jac_and_hessian_fn = lambda guess: (likelihood_or_score_fn(guess),
                                          hessian_fn(guess))
    InternalState = collections.namedtuple(
        "InternalState", "guess new_guess score hessian step halving converged")

  X_DIM = initial_guess.shape[0]

  def newton_update(state: InternalState):
    new_guess = state.new_guess
    if use_likelihood:
      loglik = state.loglik
      new_loglik, new_score, new_hessian = value_jac_and_hessian_fn(new_guess)
    else:
      new_score, new_hessian = jac_and_hessian_fn(new_guess)
    cho_factor = scipy.linalg.cholesky(-new_hessian, lower=False)

    is_finite = jnp.logical_and(jnp.all(jnp.isfinite(new_score)),
                                jnp.all(jnp.isfinite(cho_factor)))
    if use_likelihood:
      is_finite = jnp.logical_and(jnp.all(jnp.isfinite(new_loglik)), is_finite)
      loglik_increased = new_loglik > loglik
      converged = jnp.logical_and(is_finite,
                                  jnp.abs(1 - loglik / new_loglik) < loglik_eps)
    else:
      converged = jnp.logical_and(
          is_finite,
          jnp.linalg.norm(new_score, ord=jnp.inf) < score_norm_eps)

    state = state._replace(converged=converged)

    if use_likelihood:
      args = (state, (new_loglik, new_score, new_hessian, cho_factor, is_finite,
                      loglik_increased))
    else:
      args = (state, (new_score, new_hessian, cho_factor, is_finite))

    return jax.lax.cond(jnp.logical_and(converged, state.step > 0),
                        functools.partial(do_converged, use_likelihood),
                        functools.partial(do_work, use_likelihood),
                        operand=args)

  def loop_cond(state):
    return jnp.logical_or(
        state.step == 0,  # Run at least one iteration
        jnp.logical_and(state.step < max_num_steps,
                        jnp.logical_not(state.converged)))

  if use_likelihood:
    initial_state = InternalState(initial_guess, initial_guess, -jnp.inf,
                                  jnp.zeros_like(initial_guess),
                                  jnp.zeros((X_DIM, X_DIM)), 0, 0, False)
  else:
    initial_state = InternalState(initial_guess, initial_guess,
                                  jnp.zeros_like(initial_guess),
                                  jnp.zeros((X_DIM, X_DIM)), 0, 0, False)

  state = jax.lax.while_loop(loop_cond, newton_update, initial_state)

  def do_recover_last(state):
    if use_likelihood:
      loglik, score, hessian = value_jac_and_hessian_fn(state.guess)
      return state._replace(loglik=loglik, score=score, hessian=hessian)
    else:
      score, hessian = jac_and_hessian_fn(state.guess)
      return state._replace(score=score, hessian=hessian)

  state = jax.lax.cond(state.converged,
                       lambda state: state,
                       do_recover_last,
                       operand=state)
  return NewtonSolverResult(state.guess,
                            state.loglik if use_likelihood else None,
                            state.score, state.hessian, state.step,
                            state.converged)


if __name__ == "__main__":

  def test_fun(x):
    return jnp.squeeze(x**3 + x**2 - x + jnp.cos(x))

  print(solve_newton(test_fun, jnp.array([1.])))
