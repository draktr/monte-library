import pytest
import numpy as np
from scipy import stats
from carlo import rejection


@pytest.fixture
def distribution():
    return stats.norm(loc=5, scale=2)


@pytest.fixture
def multinorm():
    return stats.multivariate_normal(
        mean=np.array([5, 2, 3]), cov=np.diag(np.array([1.5, 1, 1]))
    )


def test_one_dimensional(distribution):

    samples = rejection(
        pdf=distribution.pdf,
        lower_bounds=np.array([-1]),
        upper_bounds=np.array([11]),
        n=10000,
    )

    assert np.abs(5 - np.mean(samples)) <= 0.5
    assert np.abs(2 - np.std(samples)) <= 0.5


def test_multidimensional(multinorm):
    samples = rejection(
        pdf=multinorm.pdf,
        lower_bounds=np.array([1, -1, 0]),
        upper_bounds=np.array([9, 5, 6]),
        n=10000,
    )

    assert np.all(np.abs(np.array([5, 2, 3]) - np.mean(samples, axis=0)) <= 0.5)
    assert np.all(np.abs(np.array([1.5, 1, 1]) - np.std(samples, axis=0)) <= 0.5)
