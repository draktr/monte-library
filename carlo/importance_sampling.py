"""
Module that contains function performing importance sampling.
"""


def importance_sampling(importance_sampler, improtance_density, target_density, n):
    """
    Performs importance sampling from `target_density` distribution using `importance`
    distribution as proposal distribution.

    :param importance_sampler: Function that samples from our importance distribution.
    Should be a distribution that is easy to sample from.
    :type importance_sampler: callable
    :param improtance_density: Probability density/mass function of importance distribution.
    :type improtance_density: callble
    :param target_density: Probability density/mass function of the distribution we want
    to have samples from. Intended to be a distribution that is difficult to sample from.
    :type target_density: callable
    :param n: Number of samples to be sampled.
    :type n: itn
    :return: Samples from our `target_distribution` that are obtained indirectly through
    importance sampling
    :rtype: ndarray
    """

    x_proposed = importance_sampler(n)
    w = target_density(x_proposed) / improtance_density(x_proposed)
    x = x_proposed * w

    return x
