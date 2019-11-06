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
  initial_norm = norm_fn(initial_value)

  InternalState = collections.namedtuple("InternalState",
                                         "guess value norm step key")

  def reset(state: InternalState):
    key, subkey = jrandom.split(state.key, 2)
    guess = initial_guess + jrandom.normal(subkey, shape=initial_guess.shape)
    value = fn(guess)
    norm = norm_fn(value)
    return InternalState(guess, value, norm, state.step, key)

  def newton_update(state: InternalState):

    guess, value, _, step, key = state
    jacobian = jac_fn(guess)
    new_guess = guess + scipy.linalg.solve(-jacobian, value, sym_pos=sym_pos)
    new_value = fn(new_guess)
    new_norm = norm_fn(new_value)
    new_step = step + 1
    new_state = InternalState(guess=new_guess,
                              value=new_value,
                              norm=new_norm,
                              step=new_step,
                              key=key)

    return jax.lax.cond(
        np.logical_or(new_step % max_reset_steps == 0, np.isnan(new_norm)),
        new_state,
        reset,  # reset if exceeded steps
        new_state,
        lambda s: s  # else return new_state
    )

  def loop_cond(state):
    return np.logical_and(state.step < max_num_steps,
                          state.norm > norm_stop_thres)

  ret = jax.lax.while_loop(
      loop_cond, newton_update,
      InternalState(initial_guess, initial_value, initial_norm, 0, key))

  return NewtonSolverState(ret.guess, ret.value, ret.step)


solve_newton_jit = jit(solve_newton)
