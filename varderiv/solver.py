"""Newton solver."""

import collections

import jax.lax
import jax.numpy as np
from jax import jacrev, jacfwd, jit
from jax import random as jrandom

NewtonSolverState = collections.namedtuple("NewtonSolverState",
                                           "guess value step")


def solve_newton(fn,
                 key,
                 initial_guess,
                 norm_stop_thres=1e-3,
                 jac_mode='reverse',
                 max_reset_steps=20,
                 max_num_steps=1000):
  """HOF for newton's method solver."""

  if jac_mode == 'forward':
    jac_fn = jacfwd(fn, 0)
  else:
    jac_fn = jacrev(fn, 0)

  initial_value = fn(initial_guess)

  InternalState = collections.namedtuple("InternalState",
                                         "guess value step key")

  def newton_update(state: InternalState):

    def reset(state: InternalState):
      key, subkey = jrandom.split(state.key, 2)
      guess = initial_guess + jrandom.normal(subkey, shape=initial_guess.shape)
      value = fn(guess)
      return InternalState(guess, value, step + 1, key)

    def update(state: InternalState):
      guess, value, step, key = state
      jacobian = jac_fn(guess)
      guess = guess - np.linalg.solve(jacobian, value)
      value = fn(guess)
      return InternalState(guess, value, step + 1, key)

    step = state.step
    return jax.lax.cond(((step + 1) % max_reset_steps == 0), state, reset,
                        state, update)

  def stop_cond(state):
    return np.logical_and(
        state.step < max_num_steps,
        np.linalg.norm(state.value, ord=np.inf) > norm_stop_thres)

  ret = jax.lax.while_loop(stop_cond, newton_update,
                           InternalState(initial_guess, initial_value, 0, key))

  return NewtonSolverState(ret.guess, ret.value, ret.step)


solve_newton_jit = jit(solve_newton)
