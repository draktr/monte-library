"""
Base sampler for Monte Carlo methods containing some basic utility methods.
"""

import numpy as np
import pandas as pd


class BaseSampler():

    def __init__(self) -> None:

        self.samples = None
        self.acceptances = None

    def save_samples(self,
                     path = None):

        pd.DataFrame(self.samples).to_csv(path)

    def mean(self):

        return np.mean(self.samples, axis=0)

    def std(self):

        return np.std(self.samples, axis=0)

    def acceptance_rate(self):

        return np.mean(self.acceptances)
