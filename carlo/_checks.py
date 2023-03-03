import numpy as np
from scipy import stats
import warnings


def _check_posterior(posterior):
    if not callable(posterior):
        raise ValueError(
            f"Posterior should be a callable. Current posterior is:{type(posterior)}"
        )


def _check_posterior_gradient(gradient):
    if not callable(gradient):
        raise ValueError(
            f"Posterior gradient should be a callable. Current posterior gradient is:{type(gradient)}"
        )


def _check_sampling_distributions(distributions):
    if not isinstance(distributions, np.ndarray):
        distributions = np.asarray(distributions)
    for distribution in distributions:
        if not callable(distribution):
            raise ValueError(
                f"Distribution {distribution} should be callable. {distribution} is {type(distribution)}"
            )
    return distributions


def _check_dimensions(theta, distributions):
    if theta.shape[0] != distributions.shape[0]:
        raise ValueError(
            f"There should be one and only one parameter per sampling distribution. \
                Currently there are {theta.shape[0]} parameters and {distributions.shape[0]} distributions."
        )


def _check_parameters(
    iter=None,
    warmup=None,
    step_size=None,
    lag=None,
    proposal_sampler=None,
    proposal_density=None,
    epsilon=None,
    l=None,
):
    if not isinstance(iter, (int, type(None))):
        raise ValueError(
            f"Number of iterations should be an integer. Curent iter is:{type(iter)}."
        )
    if iter <= 1 or None:
        raise ValueError(
            f"Number of iterations should be greater than or equal to 1. Curent iter={iter}."
        )
    if not isinstance(warmup, (int, type(None))):
        raise ValueError(
            f"Number of warmup steps should be an integer. Curent warmup is:{type(warmup)}."
        )
    if warmup < 0 or None:
        raise ValueError(
            f"Number of warmup steps should be non-negative. Current warmup={warmup}."
        )
    if not isinstance(step_size, (float, int, type(None))):
        raise ValueError(
            f"Step size should be a float or an integer. Current step_size is:{type(step_size)}"
        )
    if step_size is not None:
        if step_size < 0:
            raise ValueError(
                f"Step size should be positive. Current step_size={step_size}"
            )
    if not isinstance(lag, (int, type(None))):
        raise ValueError(f"Lag should be an integer. Current lag is:{type(lag)}")
    if lag is not None:
        if lag < 1:
            raise ValueError(
                f"Lag should be greater than or equal to 1. Curent lag={lag}."
            )
    if not callable(proposal_sampler) and not isinstance(proposal_sampler, type(None)):
        raise ValueError(
            f"Proposal sampler should be a callable. Current proposal_sampler is:{type(proposal_sampler)}"
        )
    if not callable(proposal_density) and not isinstance(proposal_density, type(None)):
        raise ValueError(
            f"Proposal density should be a callable. Current proposal_density is:{type(proposal_density)}"
        )
    if not isinstance(epsilon, (float, int, type(None))):
        raise ValueError(
            f"Epsilon should be a float or an integer. Current epsilon is:{type(epsilon)}"
        )
    if epsilon is not None:
        if epsilon < 0:
            raise ValueError(f"Epsilon should be positive. Current epsilon={epsilon}")
    if not isinstance(l, (int, type(None))):
        raise ValueError(f"l should be an integer. Current l is:{type(l)}")
    if l is not None:
        if l <= 0:
            raise ValueError(f"l should be positive. Current l={l}")


def _check_theta(theta):
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    return theta


def _check_metric(metric, theta):
    if metric is None:
        metric = np.identity(theta.shape[0])
    if not isinstance(metric, np.ndarray):
        metric = np.asarray(metric)
    if np.array_equal(metric, metric.T):
        try:
            np.linalg.cholesky(metric)
            return metric
        except np.linalg.LinAlgError:
            raise ValueError("Metric M must be symmetric positive definite matrix")
    else:
        raise ValueError("Metric M must be symmetric positive definite matrix")


def _check_proposal(sampler, density):
    if sampler is None and density is None:
        warnings.warn(
            "Proposal sampler and density are not provided. Normal distribution is used as proposal by default.",
            UserWarning,
        )

        def sampler(location):
            return stats.multivariate_normal(
                mean=location, cov=np.identity(location.shape[0])
            ).rvs()

        def density(x, location):
            return stats.multivariate_normal(
                mean=location, cov=np.identity(location.shape[0])
            ).pdf(x)

    if sampler is None and density is not None:
        raise ValueError("Proposal sampler is not provided.")
    if sampler is not None and density is None:
        raise ValueError("Proposal density is not provided.")
    return sampler, density
