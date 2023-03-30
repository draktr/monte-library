Monte Library
=============

monte-library is a set of Monte Carlo methods in Python. The package is written
to be flexible, clear to understand and encompass variety of Monte Carlo
methods.

-  Free software: MIT license
-  Documentation: https://monte-library.readthedocs.io/en/latest/

Installation
------------

Preferred method to install ``monte-library`` is through Python’s package
installer pip. To install ``monte-library``, run this command in your terminal

.. code:: shell

   pip install monte-library

Alternatively, you can install the package directly from GitHub:

.. code:: shell

   git clone -b development https://github.com/draktr/monte-library.git
   cd monte-library
   python setup.py install

Features
--------

Base module
~~~~~~~~~~~

-  saving samples and log probability values as ``.csv`` file
-  posterior mean, standard deviation and quantiles
-  diagnostic tools: effective sample size (ESS), autocorrelation plots,
   ergodic mean plots, acceptance rate
-  visualizations: histograms, kernel density estimation plots,
   traceplots

General Monte Carlo Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  multidimensional Monte Carlo integration
-  multidimensional rejection sampling
-  multidimensional importance sampling

Markov Chain Monte Carlo Modelling Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  symmetric proposal Metropolis algorithm
-  Metropolis-Hastings algorithm
-  Gibbs sampler
-  vanilla Hamiltonian Monte Carlo

Advantages
----------

-  **FLEXIBLE** - the package allows users to use various existing Monte
   Carlo methods for their needs without needing to write the whole
   algorithm. At the same time, ``monte-library`` allows users to specify their
   own hyperparameters, posterior and proposal distributions as needed.
   Furthermore ``BaseSampler`` class can be used as parent class for any
   proprietary Monte Carlo algorithm thus utilizing its features for
   visualizations, posterior analysis and convergence checks

-  **SIMPLE AND CLEAR CODE BASE** - code was intentionally kept simple
   to be understandable to those with limited exposure to Statistical
   Computing. ``monte-library`` is a great tool to supplement learning as it
   generally matches mathematical formulations of algorithms and simple
   syntax helps focus on the algorithm itself.

-  **COMPREHENSIVE** - includes Monte Calor methods for various
   applications. Bayesian modelling methods include both classical
   methods (Metropolis algorithm) as well as more advanced methods such
   as Hamiltonian Monte Carlo.

Usage
-----

Package contains variety of Monte Carlo methods that can be applied to
problems ranging from integration to modelling. Importantly, code is
both simple and generalized as to match the respective mathematical
formulations of algorithms. As such it can be a great supplement when
learning these topics. Finally, the package is flexible and
``BaseSampler`` class can be used as a parent class to any user-defined
sampler. Furthermore, it is easy to modify existing algorithms with
proprietary improvements. Below is an example of Monte Carlo integration
and more examples can be found by clicking on the `Examples` section
of the documentation

Example 1: Monte Carlo Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example is a simple Monte Carlo implementation to solve
the following integral:

.. math::  \int_{-3}^{3} \int_{-3}^{3} x^2 + y^3 dxdy

.. code:: python

   from monte import integrator

   def integrand(args):
           return args[0] ** 2 + args[1] ** 3

   result = integrator(integrand, lower_bounds=[-3, -3], upper_bounds=[3, 3], n=10000000)
   result

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    monte.base_sampler
    monte.gaussian_metropolis
    monte.generalized_metropolis
    monte.gibbs_sampler
    monte.hamiltonian_mc
    monte.importance_sampling
    monte.metropolis_hastings
    monte.monte_carlo_integrator
    monte.rejection_sampling

.. toctree::
    :maxdepth: 1
    :caption: Other

    examples
    future_dev
    further_reading


Alternatives and Complements
----------------------------

There are more sophisticated and computationally efficient
implementations of Monte Carlo methods for off-the-shelf solutions

-  `ArviZ <https://www.arviz.org/en/latest/>`__ - independent library
   for exploratory analysis of Bayesian models
-  `vegas <https://github.com/gplepage/vegas>`__ - uses improved version
   of the adaptive Monte Carlo vegas algorithm
-  `OpenBUGS <https://www.mrc-bsu.cam.ac.uk/software/bugs/openbugs/>`__
   - open source implementation of BUGS language utilizing Gibbs sampler
-  `JAGS <https://mcmc-jags.sourceforge.io/>`__ - cross-platform and
   more extensible implementation of BUGS language
-  `WinBUGS <https://www.mrc-bsu.cam.ac.uk/software/bugs/the-bugs-project-winbugs/>`__
   - software for Bayesian analysis utilizing Gibbs sampler (available,
   but discontinued in favour of OpenBUGS)
-  `Stan <https://mc-stan.org/>`__ - state-of-the-art probabilistic
   language implementing advanced version of No-U-Turn Sampler
-  `PyMC <https://github.com/pymc-devs/pymc>`__ - supports HMC and
   Metropolis-Hastings algorithms, as well as Sequential Monte Carlo
   methods


Project Principles
------------------

-  Easy to be understood and used by non-mathematicians
-  Potential to be used as pedagogical tool
-  Easy to modify algorithms with proprietary improvements
-  Flexibility and simplicity over computational efficiency
-  Tested
-  Dedicated documentation
-  Formatting deferred to `Black <https://github.com/psf/black>`__

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
