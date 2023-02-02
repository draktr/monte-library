import pytest
import numpy as np
from scipy import stats
from carlo.hamiltonian_mc import HamiltonianMC


def generate_data(true_param, n):

    x = np.linspace(-20, 20, n)
    y = true_param[0] * x + true_param[1] + stats.norm(loc=0, scale=true_param[2])
    return x, y


def target(X, y, th):
    beta = th[0:-1]
    sigma = th[-1]
    sigma2 = sigma**2
    mu = X * beta

    prioralpha = 0.001
    priorbeta = 0.001

    likelihood = stats.multivariate_normal(
        mean=mu, cov=np.diag(np.repeat(sigma2, len(y)))
    ).pdf(y)
    priorb = stats.multivariate_normal(
        mean=np.repeat(0, len(beta)), cov=np.diag(np.repeat(100, len(beta)))
    ).pdf(beta)
    priors2 = stats.gamma(a=prioralpha, scale=1 / priorbeta).pdf(1 / sigma2)
    posterior = likelihood * priorb * priors2

    return posterior


def target_gradietn(X, y, th):
    d = len(th)
    e = 0.0001
    diffs = np.zeros(d)

    for k in range(d):
        th_hi = th
        th_lo = th
        th_hi[k] = th[k] + e
        th_lo[k] = th[k] - e
        diffs[k] = (target(X, y, th_hi) - target(X, y, th_lo)) / (2 * e)

    return diffs


@pytest.fixture
def sampler(target, target_gradient):

    sampler = HamiltonianMC(target, target_gradient)
    return sampler


def test(sampler):

    true_param = np.array([5, 10, 2])
    x, y = generate_data(true_param=true_param, n=50)
    sampler.sample(iter=100000, warmup=1000, theta=0, step_size=0.1, lag=1)

    mean = sampler.mean()
    assert mean - true_param <= np.repeat(10 ** (-2), len(mean))
