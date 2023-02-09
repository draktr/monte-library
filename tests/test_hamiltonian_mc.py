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


def target(x, y, theta):
    mu = np.matmul(x, theta[0:-1])

    prioralpha = 0.001
    priorbeta = 0.001

    likelihood = stats.multivariate_normal(
        mean=mu, cov=np.diag(np.repeat(theta[-1] ** 2, len(y)))
    ).logpdf(y)
    priorb = stats.multivariate_normal(
        mean=np.repeat(0, len(theta[0:-1])),
        cov=np.diag(np.repeat(100, len(theta[0:-1]))),
    ).logpdf(theta[0:-1])
    priors2 = stats.gamma(a=prioralpha, scale=1 / priorbeta).logpdf(1 / theta[-1] ** 2)
    posterior = likelihood * priorb * priors2

    return posterior


def target_gradient(x, y, theta):
    delta = 0.0001
    gradient = np.zeros(len(theta))

    for k in range(len(theta)):
        theta_hi = theta
        theta_lo = theta
        theta_hi[k] = theta[k] + delta
        theta_lo[k] = theta[k] - delta
        gradient[k] = (target(x, y, theta_hi) - target(x, y, theta_lo)) / (2 * delta)

    return gradient


@pytest.fixture
def sampler(target, target_gradient):

    sampler = HamiltonianMC(target, target_gradient)
    return sampler


def test(sampler):

    true_param = np.array([5, 10, 2])
    x, y = generate_data(true_param=true_param, n=50)
    sampler.sample(iter=100000, warmup=1000, theta=0, step_size=0.1, lag=1, x=x, y=y)

    mean = sampler.mean()
    assert mean - true_param <= np.repeat(10 ** (-2), len(mean))
