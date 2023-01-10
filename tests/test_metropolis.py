import pytest
from gaussian_metropolis import GaussianMetropolis
from generalized_metropolis import GeneralizedMetropolis


def target():
    pass

@pytest.fixture
def gaussian_sampler(target):
    sampler = GaussianMetropolis(target)
    return sampler

@pytest.fixture
def generalized_sampler(target):
    sampler = GeneralizedMetropolis(target)
    return sampler

def test_gaussian(gaussian_sampler):
    gaussian_sampler.sample(iter = 100000,
                            warmup = 1000,
                            theta = 0,
                            step_size = 0.1,
                            lag = 1)

    assert gaussian_sampler.mean() - 10 <= 10**(-2)

def test_generalized(generalized_sampler):
    generalized_sampler.sample(iter = 100000,
                               warmup = 1000,
                               theta = 0,
                               step_size = 0.1,
                               lag = 1)

    assert generalized_sampler.mean() - 10 <= 10**(-2)
