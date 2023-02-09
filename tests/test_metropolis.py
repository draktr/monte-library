import pytest
import numpy as np
from scipy import stats
from carlo.gaussian_metropolis import GaussianMetropolis
from carlo.generalized_metropolis import GeneralizedMetropolis


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


def target(theta, x, y):

    b0prior = stats.norm(loc=0, scale=30).logpdf(theta[0])
    b1prior = stats.norm(loc=0, scale=30).logpdf(theta[1])
    b2prior = stats.norm(loc=0, scale=30).logpdf(theta[2])
    b3prior = stats.norm(loc=0, scale=30).logpdf(theta[3])
    sdprior = stats.uniform(loc=0, scale=30).logpdf(theta[-1])

    mu = np.matmul(x, theta[0:-1])
    singlelikelihoods = stats.norm(loc=mu, scale=theta[-1]).logpdf(y)
    prodlikelihood = np.sum(singlelikelihoods)

    return b0prior + b1prior + b2prior + b3prior + sdprior + prodlikelihood


def cauchy_proposal(location):
    return stats.cauchy(loc=location, scale=1).rvs(size=location.shape[0])


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
