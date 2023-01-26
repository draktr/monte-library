"""
Blockwise-only for multiple parameters Metropolis algorithm,
no Hastings Ratio, that takes proposal distributions as an argument
"""

import numpy as np
from base_sampler import BaseSampler


class GeneralizedMetropolis(BaseSampler):
    def __init__(self, target) -> None:

        super().__init__()
        self.target = target

    def _iterate(self, theta_current, proposal_sampler, **proposal_parameters):

        theta_proposed = proposal_sampler(theta_current, **proposal_parameters)
        alpha = min(1, self.target(theta_proposed) / self.target(theta_current))
        u = np.random.rand()
        if u <= alpha:
            theta_new = theta_proposed
            a = 1
        else:
            theta_new = theta_current
            a = 0

        return theta_new, a

    def sample(
        self, iter, warmup, theta, proposal_sampler, lag=1, **proposal_parameters
    ):

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)

        for i in range(warmup):
            theta, a = self._iterate(theta, proposal_sampler, **proposal_parameters)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta, proposal_sampler, **proposal_parameters)
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
