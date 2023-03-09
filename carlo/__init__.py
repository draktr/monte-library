"""
Carlo

Carlo is a set of Monte Carlo methods in Python. The package is written to be flexible, clear to understand and encompass variety of Monte Carlo methods.
"""

from carlo.base_sampler import BaseSampler
from carlo.gaussian_metropolis import GaussianMetropolis
from carlo.generalized_metropolis import GeneralizedMetropolis
from carlo.gibbs_sampler import GibbsSampler
from carlo.hamiltonian_mc import HamiltonianMC
from carlo.importance_sampling import importance
from carlo.metropolis_hastings import MetropolisHastings
from carlo.monte_carlo_integrator import integrator
from carlo.rejection_sampling import rejection

__all__ = [s for s in dir() if not s.startswith("_")]

__version__ = "0.1.0"
__author__ = "draktr"
