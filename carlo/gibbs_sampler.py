"""
Module containing Markov Chains Monte Carlo sampler utilizing a special case of
single-component Metropolis-Hastings algorithm called Gibbs sampler. Since Gibbs
sampler utilizes full conditional posterior distribution as proposal density, all
proposals are accepted.
"""

import numpy as np
import carlo._checks
from carlo import BaseSampler


class GibbsSampler(BaseSampler):
    def __init__(self, sampling_distributions) -> None:
        """
        Initializes the problem sampler object.

        :param sampling_distributions: Array of full conditional posterior
        distributions :math:`f(\theta_j|\theta_{\backslash j, y})`.
        The order of the distributions should match the order of parameters
        in parameter vector passed to `sample()` method. Each respective
        distribution should take all other parameters as arguments and return
        the sample of the particular parameter conditional on those arguments.
        :type sampling_distributions: ndarray
        """

        super().__init__()
        sampling_distributions = carlo._checks._check_sampling_distributions(
            sampling_distributions
        )
        self.sampling_distributions = sampling_distributions

    def _iterate(self, theta, **kwargs):
        """
        Single iteration of the sampler

        :param theta: Vector of current values of parameters
        :type theta: ndarray
        :return: New value of parameter vector, acceptance information
        :rtype: ndarray, int
        """

        for i in range(theta.shape[0]):
            theta[i] = self.sampling_distributions[i](theta, **kwargs)
        a = 1

        return theta, a

    def sample(self, iter, warmup, theta, lag=1, **kwargs):
        """
        Samples from the posterior distribution

        :param iter: Number of iterations of the algorithm
        :type iter: int
        :param warmup: Number of warmup steps of the algorithm. These are discarded
        so that the only samples recorded are the ones obtained after the Markov chain
        has reached the stationary distribution
        :type warmup: int
        :param theta: Vector of initial values of parameters
        :type theta: ndarray
        :param lag: Sampler lag. Parameter specifying every how many iterations will the sample
        be recorded. Used to limit autocorrelation of the samples. If `lag=1`, every sample is
        recorded, if `lag=3` each third sample is recorded, etc. , defaults to 1
        :type lag: int, optional
        :raises ValueError: Returns error if number of parameters doesn't match the number
        of sampling distributions
        :return: Numpy array of samples for every parameter, for every algorithm iteration,
        numpy array of acceptance information for every algorithm iteration.
        :rtype: ndarray, ndarray
        """

        carlo._checks._check_parameters(iter=iter, warmup=warmup, lag=lag)
        theta = carlo._checks._check_theta(theta)
        carlo._checks._check_dimensions(theta, self.sampling_distributions)

        samples = np.zeros((iter, theta.shape[0]))
        acceptances = np.zeros(iter)

        for i in range(warmup):
            theta, a = self._iterate(theta, **kwargs)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta, **kwargs)
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
