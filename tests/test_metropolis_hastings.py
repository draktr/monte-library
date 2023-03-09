import pytest
import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal as mvn, norm
from scipy.stats._multivariate import _squeeze_output
from monte import MetropolisHastings


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

    return beta_prior + sd_prior + likelihood


class Skewnorm:

    # By [Gregory Gundersen](https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/)

    def __init__(self, shape, mean=None, cov=None):
        self.dim = len(shape)
        self.shape = np.asarray(shape)
        self.mean = np.zeros(self.dim) if mean is None else np.asarray(mean)
        self.cov = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x, mean):
        return np.exp(self.logpdf(x, mean))

    def logpdf(self, x, mean):
        x = mvn._process_quantiles(x, self.dim)
        pdf = mvn(mean, self.cov).logpdf(x)
        cdf = norm(0, 1).logcdf(np.dot(x, self.shape))
        return _squeeze_output(np.log(2) + pdf + cdf)


def proposal_sampler(location):
    dim = location.shape[0]
    proposals = np.zeros(dim)
    for i in range(dim):
        proposals[i] = stats.skewnorm(a=1, loc=location[i], scale=1).rvs()
    return proposals


@pytest.fixture
def proposal_density():
    proposal = Skewnorm(shape=np.array([2, 2, 2, 2, 2]))
    return proposal


@pytest.fixture
def sampler():
    sampler = MetropolisHastings(target)
    return sampler


def test_gaussian_proposal(sampler, data):

    sampler.sample(
        iter=10000,
        warmup=5000,
        theta=np.array([0, 0, 0, 0, 1]),
        lag=1,
        x=data[1],
        y=data[2],
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 0.5)


def test_hastings_ratio(sampler, data, proposal_density):

    sampler.sample(
        iter=10000,
        warmup=5000,
        theta=np.array([0, 0, 0, 0, 1]),
        proposal_sampler=proposal_sampler,
        proposal_density=proposal_density.pdf,
        lag=1,
        x=data[1],
        y=data[2],
    )

    expected_theta = sampler.mean()
    assert np.all(np.abs(expected_theta - data[0]) <= 0.5)
