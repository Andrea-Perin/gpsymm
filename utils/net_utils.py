# Utilities for jax networks
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import PRNGKeyArray, PyTree


def kaiming_uniform_pytree(key: PRNGKeyArray, params: PyTree) -> PyTree:
    """
    Create a new PyTree with Kaiming uniform initialization for an ensemble of networks,
    preserving empty tuples.

    Args:
        key: JAX random key
        params: PyTree to match structure, with leading ensemble dimension on arrays

    Returns:
        New PyTree with Kaiming uniform arrays
    """
    def init_array(key, param):
        if isinstance(param, tuple) and len(param) == 0:
            return ()  # preserve empty tuples

        # Handle the ensemble dimension - use second dim for fan_in calculation
        if len(param.shape) == 2:  # Linear layer weights for ensemble
            fan_in = param.shape[1]
        elif len(param.shape) == 3:  # Weight matrices with ensemble dimension
            fan_in = param.shape[2]
        else:  # Biases with ensemble dimension
            fan_in = 1

        bound = 1 / jnp.sqrt(fan_in)
        return jr.uniform(key, shape=param.shape, minval=-bound, maxval=bound)

    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    random_leaves = [init_array(k, leaf) for k, leaf in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(tree, random_leaves)
