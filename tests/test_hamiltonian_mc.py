import pytest
import numpy as np
from monte import HamiltonianMC


def target(theta):
    return -0.5 * np.sum(theta**2)


def target_gradient(theta):
    return -theta


@pytest.fixture
def sampler():

    sampler = HamiltonianMC(target, target_gradient)
    return sampler


def test(sampler):

    sampler.sample(
        iter=10000,
        warmup=10,
        theta=np.array([8.0, -3.0]),
        epsilon=0.01,
        l=10,
        metric=None,
        lag=1,
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - np.array([0.0, 0.0])) <= 0.5)
