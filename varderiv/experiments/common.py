"""Basic Common configs for experiments."""

import jax.random as jrandom
from sacred import Ingredient

# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring

ingredient = Ingredient('base')


@ingredient.config
def config():
  num_experiments = 10000
  num_threads = 1
  batch_size = 256

  N = 500
  X_DIM = 3
  K = 3
  group_labels_generator_kind = "same"
  group_labels_generator_kind_kwargs = {}
  group_X_same = True
  T_star_factors = None

  seed = 0
  key = jrandom.PRNGKey(seed)
  experiment_rand_key, data_generation_key = jrandom.split(key, 2)
  del key
