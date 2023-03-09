"""
Module containing Monte Carlo integrator

:raises ValueError: Lower and upper bounds are of different dimensions.
:return: Approximate integral value
:rtype: float
"""
import numpy as np


def integrator(integrand, lower_bounds, upper_bounds, n):
    """
    Integrates n-dimensional function using Monte Carlo simulation

    :param integrand: Integrand, function to be integrated
    :type integrand: function
    :param lower_bounds: Lower bound(s) of the integration region
    :type lower_bounds: ndarray
    :param upper_bounds: Upper bound(s) of the integration region
    :type upper_bounds: ndarray
    :param n: Number of simulation samples
    :type n: int
    :raises ValueError: Error raised if bounds provided are of different dimensions
    :return: Estimated integral value
    :rtype: float
    """

    if lower_bounds.shape[0] != upper_bounds.shape[0]:
        raise ValueError("Lower and upper bounds are of different dimensions.")

    x = np.random.uniform(lower_bounds, upper_bounds, (n, lower_bounds.shape[0]))

    summation = 0
    for i in x:
        summation += integrand(i)
    domain = np.prod(np.subtract(upper_bounds, lower_bounds))

    integral = domain / n * summation

    return integral
