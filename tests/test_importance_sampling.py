import pytest
import numpy as np
from scipy import stats
from monte import importance


@pytest.fixture
def importance():
    return stats.t(loc=5, scale=2, df=16)


@pytest.fixture
def target():
    return stats.norm(loc=5, scale=2)


@pytest.fixture
def multiimportance():
    return stats.multivariate_t(
        loc=np.array([5, 2, 3]), shape=np.diag(np.array([1.5, 1, 1])), df=16
    )


@pytest.fixture
def multitarget():
    return stats.multivariate_normal(
        mean=np.array([5, 2, 3]), cov=np.diag(np.array([1.5, 1, 1]))
    )


def test_one_dimensional(importance, target):

    samples = importance(
        importance_sampler=importance.rvs,
        importance_density=importance.pdf,
        target_density=target.pdf,
        n=100000,
    )

    assert np.abs(5 - np.mean(samples)) <= 0.1 and np.abs(2 - np.std(samples)) <= 0.1


def test_multidimensiona(multiimportance, multitarget):
    samples = importance(
        importance_sampler=multiimportance.rvs,
        importance_density=multiimportance.pdf,
        target_density=multitarget.pdf,
        n=100000,
    )

    assert np.all(
        np.abs(np.array([5, 2, 3]) - np.mean(samples, axis=0)) <= 0.1
    ) and np.all(np.abs(np.array([1.5, 1, 1]) - np.std(samples, axis=0)) <= 0.1)
