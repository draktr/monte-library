"""
Monte

Monte is a set of Monte Carlo methods in Python. The package is written to be flexible, clear to understand and encompass variety of Monte Carlo methods.
"""

from monte.base_sampler import BaseSampler
from monte.gaussian_metropolis import GaussianMetropolis
from monte.generalized_metropolis import GeneralizedMetropolis
from monte.gibbs_sampler import GibbsSampler
from monte.hamiltonian_mc import HamiltonianMC
from monte.importance_sampling import importance
from monte.metropolis_hastings import MetropolisHastings
from monte.monte_carlo_integrator import integrator
from monte.rejection_sampling import rejection

__all__ = [s for s in dir() if not s.startswith("_")]

__version__ = "0.1.0"
__author__ = "draktr"
