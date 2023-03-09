import pytest
import numpy as np
from monte import integrator


def test_one_dimensional():
    def objective(args):
        return args[0] ** 2

    result = integrator(objective, lower_bounds=[-3], upper_bounds=[3], n=10000000)

    assert np.abs(result - 18) <= 10 ** (-2)


def test_two_dimensional():
    def objective(args):
        return args[0] ** 2 + args[1] ** 3

    result = integrator(
        objective, lower_bounds=[-3, -3], upper_bounds=[3, 3], n=10000000
    )

    assert np.abs(result - 108) <= 10 ** (-2)
