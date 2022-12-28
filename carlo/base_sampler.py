"""
Base sampler for Monte Carlo methods containing some basic utility methods.
"""

import pandas as pd


class BaseSampler():

    def __init__(self) -> None:
        pass

    def save_samples(self,
                     samples,
                     path = None):

        pd.DataFrame(samples).to_csv(path)
