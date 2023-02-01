import pytest
import numpy as np
from scipy import stats
from carlo.gibbs_sampler import GibbsSampler


def generate_data(true_param, n):

    x = np.linspace(-20, 20, n)
    y = true_param[0] * x + true_param[1] + stats.norm(loc=0, scale=true_param[2])
    return x, y


def distributions():
    pass


@pytest.fixture
def sampler(distributions):
    sampler = GibbsSampler(distributions)
    return sampler


def test(sampler):

    true_param = np.array([5, 10, 2])
    x, y = generate_data(true_param=true_param, n=50)
    sampler.sample(iter=100000, warmup=1000, theta=0, step_size=0.1, lag=1)

    mean = sampler.mean()
    assert mean - true_param <= np.repeat(10 ** (-2), len(mean))
