"""
Module containing Markov Chains Monte Carlo sampler utilizing Hamiltonian Monte Carlo (HMC)
algorithm. HMC uses Hamiltonian dynamics to sample by moving a particle through the posterior
to perform numerical integration that is  then corrected by Metropolis acceptance criterion.
"""

import numpy as np
from carlo import base_sampler


class HamiltonianMC(base_sampler.BaseSampler):
    def __init__(self, target_lp, target_lp_gradient) -> None:
        """
        Initializes the problem sampler object

        :param target_lp: Log-probability of the target distribution to be sampled from
        :type target_lp: function
        :param target_lp_gradient: Log-probability of the gradient of the target
        distribution to be sampled from
        :type target_lp_gradient: list
        """

        super().__init__()
        self.target_lp = target_lp
        self.target_lp_gradient = target_lp_gradient

    def _hamiltonian(self, theta, rho, inverse_metric, **kwargs):
        """
        Calculates the Hamiltonian of the current particle. Mass of the particle is taken as
        :math:`m=1`. The most general form of kinetic energy of a physical object is used:
        :math:`K(\rho) = \frac{|\rho|^2}{2m}`.

        :param theta: Vector of current values of parameter(s)
        :type theta: ndarray
        :param rho: Particle momentum vector
        :type rho: ndarray
        :return: Hamiltonian of the current particle
        :rtype: float
        """

        return 0.5 * np.dot(np.matmul(inverse_metric, rho), rho) - self.target_lp(
            theta, **kwargs
        )

    def _acceptance(
        self,
        theta_proposed,
        rho_updated,
        theta_current,
        rho_current,
        inverse_metric,
        **kwargs
    ):
        """
        Metropolis acceptance criterion

        :param theta_proposed: Proposed parameter vector
        :type theta_proposed: ndarray
        :param rho_updated: Updated momentum vector
        :type rho_updated: ndarray
        :param theta_current: Current parameter vector
        :type theta_current: ndarray
        :param rho_current: Current momentum vector
        :type rho_current: ndarray
        :return: Acceptance probability
        :rtype: float
        """

        return min(
            1,
            np.exp(
                (
                    self._hamiltonian(
                        theta_proposed, rho_updated, inverse_metric, **kwargs
                    )
                    - self._hamiltonian(
                        theta_current, rho_current, inverse_metric, **kwargs
                    )
                )
            ),
        )

    def _leapfrog(self, theta, rho, epsilon, l, inverse_metric, **kwargs):
        """
        Leapfrom integrator used to simulate the movement of the particle

        :param theta: Vector of current values of parameter(s)
        :type theta: ndarray
        :param rho: Particle momentum vector
        :type rho: ndarray
        :param epsilon: Discretization time or timestep
        :type epsilon: float
        :param l: Number of leapfrog steps taken
        :type l: int
        :param inverse_metric: Diagonal estimate of the covariance of the posterior
        :type inverse_metric: ndarray
        :return: Parameter and momentum vectors
        :rtype: ndarray, ndarray
        """

        rho -= 0.5 * epsilon * -self.target_lp_gradient(theta, **kwargs)
        for _ in range(l - 1):
            theta += epsilon * np.matmul(inverse_metric, rho)
            rho -= epsilon * -self.target_lp_gradient(theta, **kwargs)
        theta += epsilon * np.matmul(inverse_metric, rho)
        rho -= 0.5 * epsilon * -self.target_lp_gradient(theta, **kwargs)

        return theta, rho

    def _iterate(self, theta_current, epsilon, l, metric, inverse_metric, **kwargs):
        """
        Single iteration of the sampler

        :param theta_current: Current parameter vector
        :type theta_current: ndarray
        :param epsilon: Discretization time or timestep
        :type epsilon: float
        :param l: Number of leapfrog steps taken
        :type l: int
        :param metric: Covariance matrix of the momentum vector sampling distribution
        :type metric: ndarray
        :param inverse_metric: Diagonal estimate of the covariance of the posterior
        :type inverse_metric: ndarray
        :return: New value of parameter vector, acceptance information
        :rtype: ndarray, int
        """

        rho_current = np.random.multivariate_normal(
            mean=np.zeros(theta_current.shape[0]),
            cov=metric,
            size=theta_current.shape[0],
        )
        theta_proposed, rho_updated = self._leapfrog(
            theta_current.copy(),
            rho_current.copy(),
            epsilon,
            l,
            inverse_metric,
            **kwargs
        )

        alpha = self._acceptance(
            theta_proposed,
            rho_updated,
            theta_current,
            rho_current,
            inverse_metric,
            **kwargs
        )
        u = np.random.rand()
        if u <= alpha:
            theta_new = theta_proposed
            a = 1
        else:
            theta_new = theta_current
            a = 0

        return theta_new, a

    def sample(self, iter, warmup, theta, epsilon, l, metric=None, lag=1, **kwargs):
        """
        Samples from the target distributions

        :param iter: Number of iterations of the algorithm
        :type iter: int
        :param warmup: Number of warmup steps of the algorithm. These are discarded
        so that the only samples recorded are the ones obtained after the Markov chain
        has reached the stationary distribution
        :type warmup: int
        :param theta: Vector of initial values of parameter(s)
        :type theta: ndarray
        :param epsilon: Discretization time or timestep
        :type epsilon: float
        :param l: Number of leapfrog steps taken
        :type l: int
        :param metric: Covariance matrix of the momentum vector sampling distribution
        :type metric: ndarray
        :param lag: Sampler lag. Parameter specifying every how many iterations will the sample
        be recorded. Used to limit autocorrelation of the samples. If `lag=1`, every sample is
        recorded, if `lag=3` each third sample is recorded, etc. , defaults to 1
        :type lag: int, optional
        :return: Numpy array of samples for every parameter, for every algorithm iteration,
        numpy array of acceptance information for every algorithm iteration.
        :rtype: ndarray, ndarray
        """

        samples = np.zeros(iter)
        acceptances = np.zeros(iter)
        if metric is None:
            metric = np.identity(theta.shape[0])
        inverse_metric = np.linalg.inv(metric)

        for i in range(warmup):
            theta, a = self._iterate(
                theta, epsilon, l, metric, inverse_metric, **kwargs
            )

        for i in range(iter):
            for _ in range(lag):
                theta, a = self._iterate(
                    theta, epsilon, l, metric, inverse_metric, **kwargs
                )
            samples[i] = theta
            acceptances[i] = a

        self.samples = samples
        self.acceptances = acceptances

        return samples, acceptances
