"""
Base sampler for Monte Carlo methods containing some basic utility methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    def plot_histogram(self,
                       bins = 100,
                       show = True,
                       save = False):

        dim = self.samples.shape[1]
        fig = plt.figure(constrined_layout = True)
        axs = [plt.subplot(dim, 1, i+1) for i in range (dim)]
        for i in range(dim):
            axs[i].hist(self.samples[:, i], bins = bins)
            axs[i].set_xlabel(f"theta_{i}")
            axs[i].set_ylabel("density")
            axs[i].set_title(f"theta_{i} histogram")
        fig.suptitle("Parameter Histograms")
        if save is True:
            plt.savefig("histograms.png", dpi=300)
        if show is True:
            plt.show()
