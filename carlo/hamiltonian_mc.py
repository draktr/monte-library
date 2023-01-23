import numpy as np
from base_sampler import BaseSampler


class HamiltonianMC(BaseSampler):

    def __init__(self,
                 target,
                 target_gradient) -> None:

        self.target = target
        self.target_gradient = target_gradient

    def sample(self,
               iter,
               warmup,
               theta,
               timestep,
               n_steps,
               lag = 1):

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)

        for i in range(warmup):
            theta, a = self._iterate(theta, timestep, n_steps)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta, timestep, n_steps)
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
