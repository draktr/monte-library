########
Examples
########

Example 1: Monte Carlo Integration
----------------------------------

The following example is a simple Monte Carlo implementation to solve
the following integral:

.. math::  \int_{-3}^{3} \int_{-3}^{3} x^2 + y^3 dxdy

.. code:: python

   from monte import integrator

   def integrand(args):
           return args[0] ** 2 + args[1] ** 3

   result = integrator(integrand, lower_bounds=[-3, -3], upper_bounds=[3, 3], n=10000000)
   result

Example 2: Bayesian Linear Regression with Metropolis Algorithm
---------------------------------------------------------------

Example 2 is using Metropolis algorithm (with Gaussian proposal) to
estimate parameters of a multivariate linear regression.

.. code:: python

   import numpy as np
   from scipy import stats
   from monte import GaussianMetropolis

   # First, we generate some data
   true_theta = np.array([5, 10, 2, 2, 4])
   n = 1000
   x = np.zeros((n, 4))
   x[:, 0] = np.repeat(1, n)
   x[:, 1:4] = stats.norm(loc=0, scale=1).rvs(size=(n, 3))

   mu = np.matmul(x, true_theta[0:-1])
   y = stats.norm(loc=mu, scale=true_theta[-1]).rvs(size=n)

   # Define the posterior
   def posterior(theta, x, y):

       beta_prior = stats.multivariate_normal(
           mean=np.repeat(0, len(theta[0:-1])),
           cov=np.diag(np.repeat(30, len(theta[0:-1]))),
       ).logpdf(theta[0:-1])
       sd_prior = stats.uniform(loc=0, scale=30).logpdf(theta[-1])

       mu = np.matmul(x, theta[0:-1])
       likelihood = np.sum(stats.norm(loc=mu, scale=theta[-1]).logpdf(y))

       return beta_prior + sd_prior + likelihood

   # Lastly, we sample
   gaussian_sampler = GaussianMetropolis(posterior)
   gaussian_sampler.sample(
       iter=10000,
       warmup=5000,
       theta=np.array([0, 0, 0, 0, 1]),
       step_size=1,
       lag=1,
       x=x,
       y=y,
       )

Using methods from the ``BaseSampler`` class we can perform posterior
analytics. These are some of the analytics methods:

.. code:: python

   # Checking parameter estimates and their credible intervals
   gaussian_sampler.mean()
   gaussian_sampler.credible_interval()

   # Checking Metropolis acceptance rate
   gaussian_sampler.acceptance_rate()

   # Plotting KDE plot with histogram
   gaussian_sampler.parameter_kde()

   # Plotting traceplots and ergodic means, and calculating effective sample sizes as convergence diagnostics
   gaussian_sampler.traceplots()
   gaussian_sampler.plot_ergodic_mean()

Example 3: Sampling from a Multivariate Distribution using Hamiltonian Monte Carlo
----------------------------------------------------------------------------------

In the following example we use Hamiltonian Monte Carlo (HMC) algorithm
to sample from a distribution. Note that this is a toy example, and HMC
is more appropriate to be used for higher-dimensional model parameter
estimation. Also note that analytical gradient is not necessary.

.. code:: python

   import numpy as np
   from monte import HamiltonianMC

   # Defining the distribution that we are going to sample from...
   def posterior(theta):
       return -0.5 * np.sum(theta**2)

   # ... and its gradient
   def posterior_gradient(theta):
       return -theta

   # Sampling
   sampler = HamiltonianMC(posterior, posterior_gradient)
   sampler.sample(
       iter=10000,
       warmup=10,
       theta=np.array([8.0, -3.0]),
       epsilon=0.01,
       l=10,
       metric=None,
       lag=1,
       )
