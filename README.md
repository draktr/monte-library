# carlo

Carlo is a set of Monte Carlo methods in Python. The package is written to be flexible, clear to understand and encompass variety of Monte Carlo methods.

* Free software: MIT license

## Installation

Preferred method to install carlo is through Python's package installer pip. To install carlo, run this command in your terminal

```shell
$ pip install carlo
```

Alternativelly, you can install the package directly from GitHub:

```shell
$ git clone -b development https://github.com/draktr/carlo.git
$ cd carlo
$ python setup.py install
```

## Features

### Base module

- saving samples and log probability values as `.csv` file
- posterior mean, standard devition and quantiles
- diagnostic tools: effective sample size (ESS), autocorrelation plots, ergodic mean plots, acceptance rate
- visualizations: histograms, kernel density estimation plots, traceplots

### General Monte Carlo methods

- multidimensional Monte Carlo integration
- multidimensional rejection sampling
- multidimensional importance sampling

### Markov Chain Monte Carlo modelling methods

- symmetric proposal Metropolis algorithm
- Metropolis-Hastings algorithm
- Gibbs sampler
- vanilla Hamiltonian Monte Carlo

## Advantages

- **FLEXIBILITY** - the package allows users to use various existing Monte Carlo methods for their needs without needing to write the whole algorithm. At the same time, `carlo` allows users to specify their own hyperparameters, posterior and proposal distributions as needed. Furthermore `BaseSampler` class can be used as parent class for any proprietary Monte Carlo algorithm thus utilizing its features for visualizations, posterior analysis and convergence checks

- **SIMPLE AND CLEAR CODE BASE** - code was intentionally kept simple to be understandable to those with limited exposure to Statistical Computing. `carlo` is a great tool to supplement learning as it is generality matches mathematical formulations of algorithm and simple syntaxt helps focus on the algorith itself.

- **COMPREHENSIVE** - includes Monte Calor methods for various applications. Bayesian modeling methods include both classical methods (Metropolis algorithm) as well as more advanced methods such as Hamiltonian Monte Carlo.

## Usage

Package contains variety of Monte Carlo methods that can be applied to problems ranging from integration to modelling. Importantly, code is both simple and generalized as to match the respective mathematical formulations of algorithms. As such it can be a great supplement when learning these topics. Finally, the package is flexible and `BaseSampler` class can be used as a parent class to any user-defined sampler. Furthermore, it is easy to modify existing algorithms with proprietary improvements.


### Example 1: Monte Carlo Integration

The following example is a simple Monte Carlo implementation to solve the following integral:
$$
    \int_{-3}^{3} \int_{-3}^{3} x^2 + y^3 dxdy
$$
```python
from carlo.monte_carlo_integrator import integrator


def integrand(args):
        return args[0] ** 2 + args[1] ** 3

result = integrator(integrand, lower_bounds=[-3, -3], upper_bounds=[3, 3], n=10000000)
result

```

### Example 2: Bayesian Linear Regression with Metropolis Algorithm

Example 2 is using Metropolis algorithm (with Gaussian proposal) to estimate parameters of a multivariate linear regression.

```python
import numpy as np
from scipy import stats
from carlo.gaussian_metropolis import GaussianMetropolis


# First, we create data
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

```

Using methods from the `BaseSampler` class we can perform posterior analytics. These are some of the analytics methods:

```python
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
```

### Example 3 Sampling from a multivariate distribution using Hamiltonian Monte Carlo

In the following example we use Hamiltonian Monte Carlo (HMC) algorithm to sample from a distribution. Note that this is a toy example, and HMC is more appropriate to be used for higher-dimensional model parameter estimation. Also note that analytical gradient is not neccesary.

```python
import numpy as np
from carlo.hamiltonian_mc import HamiltonianMC


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
```
