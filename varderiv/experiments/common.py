"""Basic Common configs for experiments."""

import functools

import jax.random as jrandom
from sacred import Ingredient

from varderiv.data import (grouping_X_generator, default_X_generator,
                           X_group_generator_indep_dim, floatt,
                           T_star_factors_gamma_gen)

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


# def grouping_Xi_generator_2(N, dim, key, group_label=0):
#   if group_label == 0:
#     bernoulli_theta, normal_mean, normal_variance = 0.5, 0, 1
#   elif group_label == 1:
#     bernoulli_theta, normal_mean, normal_variance = 0.3, 1, 0.5
#   elif group_label == 2:
#     bernoulli_theta, normal_mean, normal_variance = 0.7, -1, 1.5
#   return jax.lax.cond(
#       dim % 2 == 0,
#       key, \
#       lambda key: jrandom.bernoulli(key, p=bernoulli_theta,
#                                     shape=(N,)).astype(np.float32),
#       key, \
#       lambda key: jrandom.normal(key, shape=(N,))) * normal_variance + normal_mean

# grouping_X_generator_2 = functools.partial(X_group_generator_indep_dim,
#                                            Xi_generator=grouping_Xi_generator_2)

# def X_group_generator_joint(N, X_dim, key, group_label=0):
#   assert X_dim == 3
#   if group_label == 0:
#     return default_X_generator(N, X_dim, key, group_label=0)
#   elif group_label == 1:
#     key, *subkeys = jrandom.split(key, 3)
#     X02 = jrandom.multivariate_normal(subkeys[0],
#                                       np.array([0, 0]),
#                                       np.array([[1, 0.25], [0.25, 0.25]]),
#                                       shape=(N,))
#     X1 = jrandom.bernoulli(subkeys[1], p=0.3, shape=(N,)).astype(floatt)
#     return np.stack(([X02[:, 0], X1, X02[:, 1]]), axis=1)
#   else:
#     key, *subkeys = jrandom.split(key, 3)
#     X02 = jrandom.multivariate_normal(subkeys[0],
#                                       np.array([0, 0]),
#                                       np.array([[1.5, 0.5], [0.5, 0.25]]),
#                                       shape=(N,))
#     X1 = jrandom.normal(subkeys[1], shape=(N,)) * 1.5
#     return np.stack(([X02[:, 0], X1, X02[:, 1]]), axis=1)


def process_params(**params):

  if not params.pop("group_X_same", False):
    assert params["K"] == 3, "other than 3 groups not supported"
    params["X_generator"] = grouping_X_generator
  else:
    params["X_generator"] = default_X_generator

  T_star_factors = params.pop("T_star_factors", None)
  if T_star_factors is "gamma":
    T_star_factors = T_star_factors_gamma_gen(1, 1)
  elif T_star_factors is "fixed":
    K = params["K"]
    T_star_factors = tuple((k + 1) // 2 for k in range(K))
  params["T_star_factors"] = T_star_factors

  return params
