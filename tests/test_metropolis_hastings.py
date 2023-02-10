import pytest
import numpy as np
from scipy import stats
from carlo.metropolis_hastings import MetropolisHastings


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

    beta_prior = stats.multivariate_normal(
        mean=np.repeat(0, len(theta[0:-1])),
        cov=np.diag(np.repeat(30, len(theta[0:-1]))),
    ).logpdf(theta[0:-1])
    sd_prior = stats.uniform(loc=0, scale=30).logpdf(theta[-1])

    mu = np.matmul(x, theta[0:-1])
    likelihood = np.sum(stats.norm(loc=mu, scale=theta[-1]).logpdf(y))

    return np.sum(beta_prior) + sd_prior + likelihood


def skewnorm_proposal(location):
    return stats.skewnorm(a=2, loc=location, scale=1).rvs(size=location.shape[0])


def skewnorm_density(x, location):
    return stats.skewnorm(a=2, loc=location, scale=1).pdf(x)


@pytest.fixture
def sampler(target):
    sampler = MetropolisHastings(target)
    return sampler


def test_gaussian_proposal(sampler, data):

    sampler.sample(
        iter=5000,
        warmup=1000,
        theta=np.array([0, 0, 0, 0, 1]),
        step_size=1,
        lag=1,
        x=data[1],
        y=data[2],
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 10 ** (-2))


def test_hastings_ratio(sampler, data):

    sampler.sample(
        iter=5000,
        warmup=1000,
        theta=np.array([0, 0, 0, 0, 1]),
        proposal_sampler=skewnorm_proposal,
        proposal_density=skewnorm_density,
        lag=1,
        x=data[1],
        y=data[2],
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 10 ** (-2))
