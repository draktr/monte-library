import numpy as np
from base_sampler import BaseSampler


class GibbsSampler(BaseSampler):

    def __init__(self,
                 sampling_distributions) -> None:

        self.sampling_distributions = sampling_distributions

    def _iterate(self,
                 theta):

        for i in range(len(theta)):
            theta[i] = self.sampling_distributions[i](theta)
        a = 1

        return theta, a

    def sample(self,
               iter,
               warmup,
               theta,
               lag = 1):

        if len(theta) != len(self.sampling_distributions):
            raise ValueError("There should be one and only one parameter per sampling distribution")

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)

        for i in range(warmup):
            theta, a = self._iterate(theta)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta)
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
