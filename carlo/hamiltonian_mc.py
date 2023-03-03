"""
Module containing Markov Chains Monte Carlo sampler utilizing Hamiltonian Monte Carlo (HMC)
algorithm. HMC uses Hamiltonian dynamics to sample by moving a particle through the posterior
to perform numerical integration that is  then corrected by Metropolis acceptance criterion.
"""

import numpy as np
import carlo._checks
from carlo import BaseSampler


class HamiltonianMC(BaseSampler):
    def __init__(self, log_posterior, log_posterior_gradient) -> None:
        """
        Initializes the problem sampler object

        :param log_posterior: Log-probability of the target distribution to be sampled from
        :type log_posterior: callable
        :param log_posterior_gradient: Log-probability of the gradient of the target
        distribution to be sampled from
        :type log_posterior_gradient: callable
        """

        super().__init__()
        carlo._checks._check_posterior(log_posterior)
        carlo._checks._check_posterior_gradient(log_posterior_gradient)
        self.log_posterior = log_posterior
        self.log_posterior_gradient = log_posterior_gradient

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

        return 0.5 * np.dot(np.matmul(rho, inverse_metric), rho) - self.log_posterior(
            theta, **kwargs
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
        rho -= 0.5 * epsilon * self.log_posterior_gradient(theta, **kwargs)
        for _ in range(l - 1):
            theta += epsilon * np.matmul(inverse_metric, rho)
            rho -= epsilon * self.log_posterior_gradient(theta, **kwargs)
        theta += epsilon * np.matmul(inverse_metric, rho)
        rho -= 0.5 * epsilon * self.log_posterior_gradient(theta, **kwargs)
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
            mean=np.zeros(theta_current.shape[0]), cov=metric
        )
        theta_proposed, rho_updated = self._leapfrog(
            theta_current.copy(),
            rho_current.copy(),
            epsilon,
            l,
            inverse_metric,
            **kwargs,
        )

        alpha = (
            min(
                0,
                -(
                    self._hamiltonian(
                        theta_proposed, rho_updated, inverse_metric, **kwargs
                    )
                    - self._hamiltonian(
                        theta_current, rho_current, inverse_metric, **kwargs
                    )
                ),
            ),
        )

        u = np.log(np.random.random())
        if u <= alpha:
            theta_new = theta_proposed
            a = 1
        else:
            theta_new = theta_current
            a = 0

        return theta_new, a

    def sample(self, iter, warmup, theta, epsilon, l, metric=None, lag=1, **kwargs):
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

        carlo._checks._check_parameters(
            iter=iter, warmup=warmup, epsilon=epsilon, l=l, lag=lag
        )
        theta = carlo._checks._check_theta(theta)
        metric = carlo._checks._check_metric(metric, theta)

        samples = np.zeros((iter, theta.shape[0]))
        acceptances = np.zeros(iter)
        lp = np.zeros(iter)

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
            lp[i] = self.log_posterior(theta, **kwargs)

        self.samples = samples
        self.acceptances = acceptances
        self.lp = lp

        return samples, acceptances
