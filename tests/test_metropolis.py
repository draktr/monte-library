import pytest
import numpy as np
from scipy import stats
from gaussian_metropolis import GaussianMetropolis
from generalized_metropolis import GeneralizedMetropolis


def generate_data(true_param, n):
    x = np.linspace(-20, 20, n)
    y = true_param[0] * x + true_param[1] + stats.norm(loc=0, scale=true_param[2])
    return x, y


@pytest.fixture
def gaussian_sampler(target):
    sampler = GaussianMetropolis(target)
    return sampler


@pytest.fixture
def generalized_sampler(target):
    sampler = GeneralizedMetropolis(target)
    return sampler


def test_gaussian(gaussian_sampler):

    true_param = np.array([5, 10, 2])
    x, y = generate_data(true_param=true_param, n=50)
    gaussian_sampler.sample(iter=100000, warmup=1000, theta=0, step_size=0.1, lag=1)

    mean = gaussian_sampler.mean()
    assert mean - true_param <= np.repeat(10 ** (-2), len(mean))


def test_generalized(generalized_sampler):

    true_param = np.array([5, 10, 2])
    x, y = generate_data(true_param=true_param, n=50)
    generalized_sampler.sample(iter=100000, warmup=1000, theta=0, step_size=0.1, lag=1)

    mean = generalized_sampler.mean()
    assert mean - true_param <= np.repeat(10 ** (-2), len(mean))
