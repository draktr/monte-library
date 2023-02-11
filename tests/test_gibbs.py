import pytest
import numpy as np
from scipy import stats
from carlo.gibbs_sampler import GibbsSampler


@pytest.fixture
def data():

    true_theta = np.array([5, 10, 2])
    n = 1000
    x = np.zeros((n, 2))
    x[:, 0] = np.repeat(1, n)
    x[:, 1] = stats.norm(loc=0, scale=1).rvs(size=n)

    mu = np.matmul(x, true_theta[0:-1])
    y = stats.norm(loc=mu, scale=true_theta[-1]).rvs(size=n)

    return true_theta, x, y


@pytest.fixture
def distributions():
    def sample_alpha(theta, hyperparameters, y, x):
        precision = hyperparameters[2] + theta[2] * y.shape[0]
        mean = (
            hyperparameters[2] * hyperparameters[0]
            + theta[2] * np.sum(y - theta[1] * x[:, 1])
        ) / precision
        return stats.norm(loc=mean, scale=1 / np.sqrt(precision)).rvs()

    def sample_beta(theta, hyperparameters, y, x):
        precision = hyperparameters[3] + theta[2] * np.sum(x[:, 1] ** 2)
        mean = (
            hyperparameters[3] * hyperparameters[1]
            + theta[2] * np.sum((y - theta[0]) * x[:, 1])
        ) / precision
        return stats.norm(loc=mean, scale=1 / np.sqrt(precision)).rvs()

    def sample_tau(theta, hyperparameters, y, x):
        chi_updated = hyperparameters[4] + y.shape[0] / 2
        psi_updated = (
            hyperparameters[5] + np.sum((y - theta[0] - theta[1] * x[:, 1]) ** 2) / 2
        )
        return stats.gamma(a=chi_updated, scale=1 / psi_updated).rvs()

    return np.array([sample_alpha, sample_beta, sample_tau])


@pytest.fixture
def sampler(distributions):
    sampler = GibbsSampler(distributions)
    return sampler


def test(sampler, data):

    sampler.sample(
        iter=100,
        warmup=10,
        theta=np.array([0, 0, 2]),
        lag=1,
        hyperparameters=np.array([0, 0, 1, 1, 2, 1]),
        x=data[1],
        y=data[2],
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 10 ** (-2))
