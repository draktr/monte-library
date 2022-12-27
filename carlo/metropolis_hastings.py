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



########################################################################

# Specifying target distribution
def targetdist(x):
        probX = np.exp(-x**2) * (2 + np.sin(x*5) + np.sin(x*2))
        return probX

from scipy.stats import rv_continuous

class gaussian_gen(rv_continuous):
    "Gaussian distribution, N(0, 1)"

    def _pdf(self, x):

        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)

gaussian = gaussian_gen(name='gaussian')
gaussian.rvs(loc = 3, scale = 1)    # samples 3 datapoints from our distribution

gaussian.pdf(3, loc=3, scale=5)


#TODO: piecewise vs blockwise
#TODO: random_state for rvs

def targetdist(theta):
        return np.exp(-theta**2) * (2 + np.sin(theta*5) + np.sin(theta*2))

mcmc = MetropolisHastings(target = targetdist)
mcmc.sample(iter = 1000,
            warmup=10,
            step_size = 1,
            theta = -1,
            proposal_sampler = gaussian.rvs,
            proposal_density = gaussian.pdf)


def foo(*args):
    print(args)

foo(1, 2)

"""
proposal_density is formated in such as way as for the
first arguments to be quantiles,
second to be mean of the distribution
`proposal_parameters` to be the other arguments
"""
