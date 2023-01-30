import pytest
from gibbs_sampler import GibbsSampler


def distributions():
    pass


@pytest.fixture
def sampler(distributions):
    sampler = GibbsSampler(distributions)
    return sampler


def test(sampler):
    sampler.sample(iter=100000, warmup=1000, theta=0, lag=1)

    assert sampler.mean() - 10 <= 10 ** (-2)
