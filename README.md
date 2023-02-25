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
