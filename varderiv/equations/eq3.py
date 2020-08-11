"""Equation 3."""

import jax.numpy as np
from jax import vmap

from varderiv.generic.modeling import sum_loglik

import varderiv.equations.eq1 as eq1

#########################################################
# BEGIN eq3
#########################################################

batch_eq3_log_likelihood = np.vectorize(vmap(eq1.batch_eq1_log_likelihood,
                                             in_axes=(0, 0, None)),
                                        signature="(K,S,P),(K,S),(P)->(K,S)")

eq3_log_likelihood = np.vectorize(sum_loglik(batch_eq3_log_likelihood),
                                  signature="(K,S,P),(K,S),(P)->()")

batch_eq3_robust_cox_correction_score = np.vectorize(
    vmap(eq1.batch_eq1_robust_cox_correction_score, in_axes=(0, 0, None)),
    signature="(K,S,P),(K,S),(P)->(K,S,P)")
