"""Basic Common configs for experiments."""

import jax.random as jrandom
from sacred import Ingredient

# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring

ingredient = Ingredient('base')


@ingredient.config
def config():
  num_experiments = 100000
  num_threads = 1
  batch_size = 128

  N = 500
  X_DIM = 2

  seed = 0
  key = jrandom.PRNGKey(seed)
  experiment_rand_key, data_generation_key = jrandom.split(key, 2)
  del key
