import pytest
import numpy as np
from scipy import stats
from carlo.gibbs_sampler import GibbsSampler


@pytest.fixture
def data():

    true_theta = np.array([5, 10, 2, 2, 4])
    n = 1000
    x = np.zeros((n, 4))
    x[:, 0] = np.repeat(1, n)
    x[:, 1:4] = stats.norm(loc=0, scale=1).rvs(size=(n, 3))

    mu = np.matmul(x, true_theta[0:-1])
    y = stats.norm(loc=mu, scale=true_theta[-1]).rvs(size=n)

    return true_theta, x, y


def distributions():
    pass


@pytest.fixture
def sampler(distributions):
    sampler = GibbsSampler(distributions)
    return sampler


def test(sampler, data):

    sampler.sample(iter=100000, warmup=1000, theta=0, lag=1, x=data[1], y=data[2])

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 10 ** (-2))
