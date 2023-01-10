import pytest
from metropolis_hastings import MetropolisHastings


def target():
    pass

@pytest.fixture
def sampler(target):
    sampler = MetropolisHastings(target)
    return sampler

def test(sampler):
    sampler.sample(iter = 100000,
                   warmup = 1000,
                   theta = 0,
                   step_size = 0.1,
                   lag = 1)

    assert sampler.mean() - 10 <= 10**(-2)
