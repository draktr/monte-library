import pytest
import numpy as np
from scipy import stats
from carlo.gaussian_metropolis import GaussianMetropolis
from carlo.generalized_metropolis import GeneralizedMetropolis


def generate_data(true_param, n):

    x = np.zeros((n, 4))
    x[:, 0] = np.repeat(1, n)
    x[:, 1:4] = stats.norm(loc=0, scale=1).rvs(size=(n, 3))

    mu = np.matmul(x, true_param[0:-1])
    y = stats.norm(loc=mu, scale=true_param[-1]).rvs(size=n)

    return x, y


def likelihood(param, x, y):
    a = param[0]
    b = param[1]
    sd = param[2]

    pred = a * x + b
    singlelikelihoods = stats.norm(loc=pred, scale=sd).pdf(y)
    prodlikelihood = np.prod(singlelikelihoods)

    return prodlikelihood


def prior(param):
    a = param[0]
    b = param[1]
    sd = param[2]
    aprior = stats.norm(loc=0, scale=30).pdf(a)
    bprior = stats.norm(loc=0, scale=30).pdf(b)
    sdprior = stats.norm(loc=0, scale=30).pdf(sd)

    return aprior * bprior * sdprior


def target(param, x, y):
    return likelihood(param, x, y) * prior(param)


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
