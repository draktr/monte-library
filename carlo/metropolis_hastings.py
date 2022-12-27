"""
Metropolis-Hastings algorithm that takes proposal distributions as an argument.
"""

import numpy as np


class MetropolisHastings():

    def __init__(self,
                 target) -> None:

        self.target = target

    def _iterate(self,
                 theta_current,
                 proposal_sampler,
                 proposal_density,
                 **proposal_parameters):

        theta_proposed = proposal_sampler(theta_current,
                                          **proposal_parameters)
        metropolis_ratio = self.target(theta_proposed) / self.target(theta_current)
        hastings_ratio = proposal_density(theta_current, theta_proposed, **proposal_parameters) / \
                         proposal_density(theta_proposed, theta_current, **proposal_parameters)
        alpha = metropolis_ratio * hastings_ratio
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
               proposal_sampler,
               proposal_density,
               lag = 1,
               **proposal_parameters):

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)

        for i in range(warmup):
            theta, a = self._iterate(theta,
                                     proposal_sampler,
                                     proposal_density,
                                     **proposal_parameters)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta,
                                         proposal_sampler,
                                         proposal_density,
                                         **proposal_parameters)
            samples[i] = theta
            acceptances[i] = a

        return samples, acceptances
