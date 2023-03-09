import pytest
import numpy as np
from scipy import stats
from monte import GibbsSampler


@pytest.fixture
def data():
    true_theta = np.array([30, 7])
    y = stats.norm(loc=true_theta[0], scale=true_theta[1]).rvs(size=200000)
    return true_theta, y


@pytest.fixture
def distributions():
    def sample_mu(theta, y):
        n = y.shape[0]
        bary = np.mean(y)

        mu_prior = 0
        s_prior = 100
        w = s_prior**2 / (theta[1] ** 2 / n + s_prior**2)
        m = w * bary + (1 - w) * mu_prior
        s = np.sqrt(w * theta[1] ** 2 / n)
        return stats.norm(loc=m, scale=s).rvs()

    def sample_sigma(theta, y):
        n = y.shape[0]

        a_prior = 0.001
        b_prior = 0.001

        a = a_prior + 0.5 * n
        b = b_prior + 0.5 * np.sum((y - theta[0]) ** 2)
        tau = stats.gamma(
            a=a,
            scale=1 / b,
        ).rvs()
        sigma = np.sqrt(1 / tau)
        return sigma

    return np.array([sample_mu, sample_sigma])


@pytest.fixture
def sampler(distributions):
    sampler = GibbsSampler(distributions)
    return sampler


def test(sampler, data):

    sampler.sample(
        iter=50000,
        warmup=5000,
        theta=np.array([0, 1]),
        lag=1,
        y=data[1],
    )

    expected_theta = sampler.mean()
    print(expected_theta)
    assert np.all(np.abs(expected_theta - data[0]) <= 0.5)
