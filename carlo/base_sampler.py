"""
Base sampler for Monte Carlo methods containing some basic utility methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
                       figsize = (12, 8),
                       bins = 100,
                       show = True,
                       save = False):

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
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

    def parameter_kde(self,
                      figsize = (12, 8),
                      histogram = True,
                      bins = 100,
                      show = True,
                      save = False,
                      **kwargs):

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i+1) for i in range (dim)]
        for i in range(dim):
            if histogram is True:
                axs[i].hist(self.samples[:, i], bins = bins)
            sns.kdeplot(self.samples[:, i], ax=axs[i], **kwargs)
            axs[i].set_xlabel(f"theta_{i}")
            axs[i].set_ylabel("density")
            axs[i].set_title(f"theta_{i} kde plot")
        fig.suptitle("Parameter KDE Plots")
        if save is True:
            plt.savefig("kde_plots.png", dpi=300)
        if show is True:
            plt.show()

    def traceplots(self,
                   figsize = (12, 8),
                   show = True,
                   save = False,
                   **kwargs):

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i+1) for i in range (dim)]
        for i in range(dim):
            axs[i].plot(self.samples[:, i], **kwargs)
            axs[i].set_xlabel("iteration")
            axs[i].set_ylabel(f"theta_{i}")
            axs[i].set_title(f"theta_{i} traceplot")
        fig.suptitle("Parameter Traceplots")
        if save is True:
            plt.savefig("parameter_traceplots.png", dpi=300)
        if show is True:
            plt.show()
