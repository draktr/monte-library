import numpy as np
from base_sampler import BaseSampler


class HamiltonianMC(BaseSampler):

    def __init__(self,
                 target_lp,
                 target_lp_gradient) -> None:

        self.target_lp = target_lp
        self.target_lp_gradient = target_lp_gradient

    def _iterate(self,
                 theta_current,
                 epsilon,
                 l,
                 metric,
                 inverse_metric,):

        rho_current = np.random.multivariate_normal(mean=np.zeros(theta_current.shape[0]),
                                                    cov=metric, size=theta_current.shape[0])
        theta_proposed, rho_updated = self._leapfrog(theta_current.copy(), rho_current.copy(),
                                                     epsilon, l, inverse_metric)

        alpha = self._acceptance_step(theta_proposed, rho_updated, theta_current, rho_current)
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
               epsilon,
               l,
               metric = None,
               lag = 1):

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)
        if metric is None:
            metric = np.identity(theta.shape[0])
        inverse_metric = np.linalg.inv(metric)

        for i in range(warmup):
            theta, a = self._iterate(theta, epsilon, l, metric, inverse_metric)

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(theta, epsilon, l, metric, inverse_metric)
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
