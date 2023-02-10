import pytest
import numpy as np
from scipy import stats
from carlo.hamiltonian_mc import HamiltonianMC


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


def target_gradient(theta, x, y):
    delta = 0.0005
    gradient = np.zeros(len(theta))

    for k in range(len(theta)):
        theta_delta = theta
        theta_delta[k] = theta[k] + delta
        gradient[k] = (target(theta_delta, x, y) - target(theta, x, y)) / delta

    return gradient


@pytest.fixture
def sampler(target, target_gradient):

    sampler = HamiltonianMC(target, target_gradient)
    return sampler


def test(sampler, data):

    sampler.sample(
        iter=100000,
        warmup=1000,
        theta=0,
        epsilon=0.1,
        l=20,
        metric=None,
        lag=1,
        x=data[1],
        y=data[2],
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 10 ** (-2))
