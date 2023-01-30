import pytest
from hamiltonian_mc import HamiltonianMC


def target_lp():
    pass


def target_lp_gradietn():
    pass


@pytest.fixture
def sampler(target_lp, target_lp_gradient):
    sampler = HamiltonianMC(target_lp, target_lp_gradient)
    return sampler


def test(sampler):
    sampler.sample(
        iter=100000, warmup=1000, theta=0, epsilon=0.1, l=10, metric=None, lag=1
    )

    assert sampler.mean() - 10 <= 10 ** (-2)
