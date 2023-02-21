import numpy as np
from scipy import stats
import warnings


def _check_posterior(posterior):
    if not callable(posterior):
        raise ValueError(
            f"Posterior should be a callable. Current posterior is:{type(posterior)}"
        )
