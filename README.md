# Monte Library

`monte-library` is a set of Monte Carlo methods in Python. The package is written to be flexible, clear to understand and encompass variety of Monte Carlo methods.

* Free software: MIT license
* Documentation: <https://monte-library.readthedocs.io/en/latest/>

## Installation

Preferred method to install `monte-library` is through Python's package installer pip. To install `monte-library`, run this command in your terminal

```shell
pip install monte-library
```

Alternatively, you can install the package directly from GitHub:

```shell
git clone -b development https://github.com/draktr/monte-library.git
cd monte-library
python setup.py install
```

## Features

### Base module

* saving samples and log probability values as `.csv` file
* posterior mean, standard deviation and quantiles
* diagnostic tools: effective sample size (ESS), autocorrelation plots, ergodic mean plots, acceptance rate
* visualizations: histograms, kernel density estimation plots, traceplots

### General Monte Carlo Methods

* multidimensional Monte Carlo integration
* multidimensional rejection sampling
* multidimensional importance sampling

### Markov Chain Monte Carlo Modelling Methods

* symmetric proposal Metropolis algorithm
* Metropolis-Hastings algorithm
* Gibbs sampler
* vanilla Hamiltonian Monte Carlo

## Advantages

* **FLEXIBLE** - the package allows users to use various existing Monte Carlo methods for their needs without needing to write the whole algorithm. At the same time, `monte-library` allows users to specify their own hyperparameters, posterior and proposal distributions as needed. Furthermore `BaseSampler` class can be used as parent class for any proprietary Monte Carlo algorithm thus utilizing its features for visualizations, posterior analysis and convergence checks.

* **SIMPLE AND CLEAR CODE BASE** - code was intentionally kept simple to be understandable to those with limited exposure to Statistical Computing. `monte-library` is a great tool to supplement learning as it generally matches mathematical formulations of algorithms and simple syntax helps focus on the algorithm itself.

* **COMPREHENSIVE** - includes Monte Calor methods for various applications. Bayesian modelling methods include both classical methods (Metropolis algorithm) as well as more advanced methods such as Hamiltonian Monte Carlo.

## Usage

Package contains variety of Monte Carlo methods that can be applied to problems ranging from integration to modelling. Importantly, code is both simple and generalized as to match the respective mathematical formulations of algorithms. As such it can be a great supplement when learning these topics. Finally, the package is flexible and `BaseSampler` class can be used as a parent class to any user-defined sampler. Furthermore, it is easy to modify existing algorithms with proprietary improvements.

### Example 1: Monte Carlo Integration

The following example is a simple Monte Carlo implementation to solve the following integral:
$$\int_{-3}^{3} \int_{-3}^{3} x^2 + y^3 dxdy$$

```python
from monte import integrator

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

### Example 3: Sampling from a Multivariate Distribution using Hamiltonian Monte Carlo

In the following example we use Hamiltonian Monte Carlo (HMC) algorithm to sample from a distribution. Note that this is a toy example, and HMC is more appropriate to be used for higher-dimensional model parameter estimation. Also note that analytical gradient is not necessary.

```python
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
```

## Alternatives and Complements

There are more sophisticated and computationally efficient implementations of Monte Carlo methods for off-the-shelf solutions

* [ArviZ](https://www.arviz.org/en/latest/) - independent library for exploratory analysis of Bayesian models
* [vegas](https://github.com/gplepage/vegas) - uses improved version of the adaptive Monte Carlo vegas algorithm
* [OpenBUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/openbugs/) - open source implementation of BUGS language utilizing Gibbs sampler
* [JAGS](https://mcmc-jags.sourceforge.io/) - cross-platform and more extensible implementation of BUGS language
* [WinBUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/the-bugs-project-winbugs/) - software for Bayesian analysis utilizing Gibbs sampler (available, but discontinued in favour of OpenBUGS)
* [Stan](https://mc-stan.org/) - state-of-the-art probabilistic language implementing advanced version of No-U-Turn Sampler
* [PyMC](https://github.com/pymc-devs/pymc) - supports HMC and Metropolis-Hastings algorithms, as well as Sequential Monte Carlo methods

## Project Principles

* Easy to be understood and used by non-mathematicians
* Potential to be used as pedagogical tool
* Easy to modify algorithms with proprietary improvements
* Flexibility and simplicity over computational efficiency
* Tested
* Dedicated documentation
* Formatting deferred to [Black](https://github.com/psf/black)

## Future Development

Feel free to reach out through Issues forum if you wish to add features or help in any other way. If there are any issues, bugs or improvement recommendations, please raise them in the forum. Especially reach out if you want to contribute with any of the possible features listed below.

### Possible Future Features

#### In `BaseSampler` Class

* Sampling trace animation
* ECDF plot
* Forrest plot of parameter estimates with credible intervals
* Monte Carlo Error
* Support for multiple chains
* Convergence iterations required (Raftery-Lewis 1995)
* Rhat (Gelman-Rubin 1992)
* Means equality test (Geweke 1992)

#### Monte Carlo Methods

* Slice sampling
* Annealed importance sampling
* Component-wise Metropolis
* Independent Metropolis
* Wang-Landau algorithm
* Monte Carlo tree search
* Direct Sampling Monte Carlo
* Monte Carlo statistical distribution test

## Further Reading

The following is the non-exhaustive list of useful sources for learning more about Monte Carlo methods. Some of the code in `monte` has been written based on mathematical formulae from some of these sources.

### General

<a id="1">[1]</a> Ntzoufras, I. (2009). Bayesian Modelling Using WinBUGS. Wiley. \
<a id="2">[2]</a> Metropolis, N., & Ulam, S. (1949). The Monte Carlo Method. Journal of the American Statistical Association, 44(247), 335–341. <https://doi.org/10.1080/01621459.1949.10483310>

### Metropolis

<a id="3">[3]</a> Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (2004). Equation of State Calculations by Fast Computing Machines. The Journal of Chemical Physics, 21(6), 1087. <https://doi.org/10.1063/1.1699114> \
<a id="4">[4]</a> Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1), 97–109. <https://doi.org/10.1093/BIOMET/57.1.97> \
<a id="5">[5]</a> Hartig, F. (n.d.). A simple Metropolis-Hastings MCMC in R | theoretical ecology. Retrieved February 15, 2023, from <https://theoreticalecology.wordpress.com/2010/09/17/metropolis-hastings-mcmc-in-r/> \
<a id="6">[6]</a> Dirty Quant @YouTube. (n.d.). The Metropolis-Hastings Algorithm (MCMC in Python) - YouTube. Retrieved February 15, 2023, from <https://www.youtube.com/watch?v=MxI78mpq_44> \
<a id="7">[7]</a> TWEAG Software Innovation Lab. (n.d.). Markov chain Monte Carlo (MCMC) Sampling, Part 1: The Basics - Tweag. Retrieved February 15, 2023, from <https://www.tweag.io/blog/2019-10-25-mcmc-intro1/> \
<a id="8">[8]</a> Urbanevych, V. (n.d.). VU | Bayesian linear regression and Metropolis-Hastings sampler. Retrieved February 15, 2023, from <https://vitaliiur.github.io/blog/2021/linreg/>

### Gibbs Sampler

<a id="9">[9]</a> Geman, S., & Geman, D. (1984). Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-6(6), 721–741. <https://doi.org/10.1109/TPAMI.1984.4767596> \
<a id="10">[10]</a> Campbell, K. R. (n.d.). Gibbs sampling for Bayesian linear regression in Python | Kieran R Campbell - blog. Retrieved February 15, 2023, from <https://kieranrcampbell.github.io/blog/2016/05/15/gibbs-sampling-bayesian-linear-regression.html>

### Hamiltonian Monte Carlo

<a id="11">[11]</a> Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. <https://doi.org/10.48550/arxiv.1701.02434> \
<a id="12">[12]</a> Neal, R. M. (2012). MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, 1–592. <https://doi.org/10.1201/b10905> \
<a id="13">[13]</a> Stan. (n.d.). 15.1 Hamiltonian Monte Carlo | Stan Reference Manual. Retrieved February 15, 2023, from <https://mc-stan.org/docs/reference-manual/hamiltonian-monte-carlo.html> \
<a id="14">[14]</a> Clark, M. (n.d.). Hamiltonian Monte Carlo | Model Estimation by Example. Retrieved February 15, 2023, from <https://m-clark.github.io/models-by-example/hamiltonian-monte-carlo.html> \
<a id="15">[15]</a> Richard. (n.d.). Markov Chains: Why Walk When You Can Flow? | Elements of Evolutionary Anthropology. Retrieved February 15, 2023, from <http://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/>
