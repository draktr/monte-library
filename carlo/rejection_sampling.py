"""
Module that contains function that performs rejection sampling.

:raises ValueError: Error raised if bounds provided are of different dimensions
:return: Accepted samples
:rtype: ndarray
"""
import numpy as np


def rejection_sampling(pdf, lower_bounds, upper_bounds, n):
    """
    Performs rejection sampling from a distribution whose probability density/mass
    function is specified in `pdf`.

    :param pdf: Probability density/mass function of a distribution that we want
    to sample from
    :type pdf: callable
    :param lower_bounds: Lower bound(s) of the sampling region
    :type lower_bounds: ndarray
    :param upper_bounds: Upper bound(s) of the sampling region
    :type upper_bounds: ndarray
    :param n: Total number of samples (both rejected and accepted)
    :type n: int
    :raises ValueError: Error raised if bounds provided are of different dimensions
    :return: Accepted samples
    :rtype: ndarray
    """

    if lower_bounds.shape[0] != upper_bounds.shape[0]:
        raise ValueError("Lower and upper bounds are of different dimensions.")

    x = np.random.uniform(lower_bounds, upper_bounds, (n, lower_bounds.shape[0]))
    p = np.zeros(n)
    for i in range(n):
        p[i] = pdf(x[i])
    u = np.random.uniform(0, 1, n)
    return p[p < u]
