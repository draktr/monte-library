"""
Blockwise-only for multiple parameters Metropolis algorithm,
no Hastings Ratio, with normal proposal distribution
"""

import numpy as np
from base_sampler import BaseSampler


class GaussianMetropolis(BaseSampler):

    def __init__(self,
                 target) -> None:

        self.target = target

    def _iterate(self,
                 theta_current,
                 step_size):

        theta_proposed = np.random.normal(loc = theta_current,
                                          scale = step_size)
        alpha = self.target(theta_proposed) / self.target(theta_current)
        u = np.random.rand()
        if u <= alpha:
            theta_new = theta_proposed
            a = 1
        else:
            theta_new = theta_current
            a = 0

        return theta_new, a

    def sample(self,
               iter,
               warmup,
               theta,
               step_size,
               lag = 1):

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)

        for i in range(warmup):
            theta, a = self._iterate(theta, step_size)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta, step_size)
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
