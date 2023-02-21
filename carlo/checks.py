import numpy as np
from scipy import stats
import warnings


def _check_posterior(posterior):
    if not callable(posterior):
        raise ValueError(
            f"Posterior should be a callable. Current posterior is:{type(posterior)}"
        )


def _check_posterior_gradient(gradient):
    if not callable(gradient):
        raise ValueError(
            f"Posterior gradient should be a callable. Current posterior gradient is:{type(gradient)}"
        )


def _check_sampling_distributions(distributions):
    if not isinstance(distributions, np.ndarray):
        distributions = np.asarray(distributions)
    for distribution in distributions:
        if not callable(distributions[distribution]):
            raise ValueError(
                f"Distribution {distribution} should be callable. {distribution} is {type(distribution)}"
            )
    return distributions


def _check_dimensions(theta, distributions):
    if theta.shape[0] != distributions.shape[0]:
        raise ValueError(
            f"There should be one and only one parameter per sampling distribution. \
                Currently there are {theta.shape[0]} parameters and {distributions.shape[0]} distributions."
        )
