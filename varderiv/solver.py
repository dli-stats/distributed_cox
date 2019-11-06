"""Newton solver."""

import collections
import functools

import jax.lax
import jax.numpy as np
from jax import jacrev, jacfwd, jit
from jax import random as jrandom
import jax.scipy as scipy

NewtonSolverState = collections.namedtuple("NewtonSolverState",
                                           "guess value step")


def solve_newton(fn,
                 key,
                 initial_guess,
                 sym_pos=False,
                 norm_stop_thres=1e-3,
                 jac_mode='forward',
                 max_reset_steps=20,
                 max_num_steps=1000):
  """HOF for newton's method solver."""

  if jac_mode == 'forward':
    jac_fn = jacfwd(fn, 0)
  else:
    jac_fn = jacrev(fn, 0)

  norm_fn = functools.partial(np.linalg.norm, ord=np.inf)

  initial_value = fn(initial_guess)
  norm = norm_fn(initial_value)

  InternalState = collections.namedtuple("InternalState",
                                         "guess value norm step key")

  def newton_update(state: InternalState):

    def reset(state: InternalState):
      key, subkey = jrandom.split(state.key, 2)
      guess = initial_guess + jrandom.normal(subkey, shape=initial_guess.shape)
      value = fn(guess)
      norm = norm_fn(state.value)
      return InternalState(guess, value, norm, step + 1, key)

    def update(state: InternalState):
      guess, value, _, step, key = state
      jacobian = jac_fn(guess)
      guess = guess + scipy.linalg.solve(-jacobian, value, sym_pos=sym_pos)
      value = fn(guess)
      norm = norm_fn(state.value)
      return InternalState(guess, value, norm, step + 1, key)

    step = state.step
    norm = state.norm
    return jax.lax.cond(
        np.logical_or((step + 1) % max_reset_steps == 0, np.isnan(norm)), state,
        reset, state, update)

  def loop_cond(state):
    return np.logical_and(state.step < max_num_steps,
                          state.norm > norm_stop_thres)

  ret = jax.lax.while_loop(
      loop_cond, newton_update,
      InternalState(initial_guess, initial_value, norm, 0, key))

  return NewtonSolverState(ret.guess, ret.value, ret.step)


solve_newton_jit = jit(solve_newton)
