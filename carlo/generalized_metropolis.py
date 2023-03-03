"""
Module containing Markov Chains Monte Carlo sampler utilizing Metropolis algorithm
with arbitrary, symetrical proposal distribution. It should be noted, that the proposal
distribution should be symetrical for the sampler to be usable. Otherwise,
Metropolis-Hastings algorithm should be used which can be found in `metropolis_hastings.py`.
"""

import numpy as np
import carlo.checks
from carlo import BaseSampler


class GeneralizedMetropolis(BaseSampler):
    def __init__(self, log_posterior) -> None:
        """
        Initializes the problem sampler object.

        :param log_posterior: Log-probability of the target distribution to be
        sampled from. This should either be posterior distribution of the model
        or a product of prior distribution and likelihood.
        :type log_posterior: callable
        """

        super().__init__()
        carlo.checks._check_posterior(log_posterior)
        self.log_posterior = log_posterior

    def _iterate(self, theta_current, proposal_sampler, **kwargs):
        """
        Single iteration of the sampler

        :param theta_current: Vector of current values of parameter(s)
        :type theta_current: ndarray
        :param proposal_sampler: Function that returns a random value from a
        desired proposal distribution given current value of parameter.
        The only argument should be the sampling distribution location.
        :type proposal_sampler: callable
        :return: New value of parameter vector, acceptance information
        :rtype: ndarray, int
        """

        theta_proposed = proposal_sampler(theta_current)
        alpha = min(
            1,
            np.exp(
                self.log_posterior(theta_proposed, **kwargs)
                - self.log_posterior(theta_current, **kwargs)
            ),
        )
        u = np.random.rand()
        if u <= alpha:
            theta_new = theta_proposed
            a = 1
        else:
            theta_new = theta_current
            a = 0

        return theta_new, a

    def sample(self, iter, warmup, theta, proposal_sampler=None, lag=1, **kwargs):
        """
        Samples from the log_posterior distribution

        :param iter: Number of iterations of the algorithm
        :type iter: int
        :param warmup: Number of warmup steps of the algorithm. These are discarded
        so that the only samples recorded are the ones obtained after the Markov chain
        has reached the stationary distribution
        :type warmup: int
        :param theta: Vector of initial values of parameter(s)
        :type theta: ndarray
        :param proposal_sampler: Function that returns a random value from a
        desired proposal distribution given current value of parameter.
        The only argument should be the sampling distribution location.
        :type proposal_sampler: callable
        :param lag: Sampler lag. Parameter specifying every how many iterations will the sample
        be recorded. Used to limit autocorrelation of the samples. If `lag=1`, every sample is
        recorded, if `lag=3` each third sample is recorded, etc. , defaults to 1
        :type lag: int, optional
        :return: Numpy array of samples for every parameter, for every algorithm iteration,
        numpy array of acceptance information for every algorithm iteration.
        :rtype: ndarray, ndarray
        """

        carlo.checks._check_parameters(
            iter=iter, warmup=warmup, proposal_sampler=proposal_sampler, lag=lag
        )
        theta = carlo.checks._check_theta(theta)

        samples = np.zeros((iter, theta.shape[0]))
        acceptances = np.zeros(iter)
        lp = np.zeros(iter)

        if proposal_sampler is None:

            def sampler(location):
                return np.random.normal(loc=location, scale=1)

            proposal_sampler = sampler

        for i in range(warmup):
            theta, a = self._iterate(theta, proposal_sampler, **kwargs)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta, proposal_sampler, **kwargs)
            samples[i] = theta
            acceptances[i] = a
            lp[i] = self.log_posterior(theta, **kwargs)

        self.samples = samples
        self.acceptances = acceptances
        self.lp = lp

        return samples, acceptances
