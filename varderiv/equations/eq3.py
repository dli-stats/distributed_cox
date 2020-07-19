"""Equation 3."""

import functools

import jax.numpy as np
from jax import jacfwd

from varderiv.solver import solve_newton
from varderiv.generic.model_solve import sum_log_likelihood

from varderiv.data import group_data_by_labels

import varderiv.equations.eq1 as eq1

#########################################################
# BEGIN eq3
#########################################################


@functools.partial(np.vectorize, signature="(K,S,P),(K,S),(P)->(K,S)")
def batch_eq3_log_likelihood(X_groups, delta_groups, beta):
  return eq1.batch_eq1_log_likelihood(
      X_groups, delta_groups,
      np.broadcast_to(beta, (X_groups.shape[0],) + beta.shape))


eq3_log_likelihood = np.vectorize(sum_log_likelihood(batch_eq3_log_likelihood),
                                  signature="(K,S,P),(K,S),(P)->()")
