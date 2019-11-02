"""Basic Common configs for experiments."""

from sacred import Ingredient

import jax.random as jrandom

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


@ingredient.config_hook
def hook(config, command_name, logger):  # pylint: disable=redefined-outer-name
  del command_name
  seed = config["base"].pop("seed")
  logger.info("Pruned seed={} off from config".format(seed))
  return config["base"]
