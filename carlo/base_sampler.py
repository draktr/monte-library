"""
Module containing base sampler for Monte Carlo
sampling methods containing some basic utility methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


class BaseSampler:
    """
    Base sampler for Monte Carlo sampling methods containing basic utility methods
    """

    def __init__(self) -> None:
        """
        Initializes the sampler with two main attributes: samples and acceptances
        which are saved here at each iteration
        """

        self.samples = None
        self.acceptances = None

    def save_samples(self, path=None):
        """
        Exports samples as `.csv` file.

        :param path: Path where to save the `.csv` file. By default, it saves to
        the working directory, defaults to None
        :type path: str, optional
        """

        pd.DataFrame(self.samples).to_csv(path)

    def mean(self):
        """
        Estimates the expected value of the posterior by calculating the mean
        of the samples

        :return: Numpy array containing posterior mean for every parameter
        :rtype: ndarray
        """

        return np.mean(self.samples, axis=0)

    def std(self):
        """
        Estimates the standard deviation of the posterior

        :return: Numpy array containing posterior standard deviation for every parameter
        :rtype: ndarray
        """

        return np.std(self.samples, axis=0)

    def acceptance_rate(self):
        """
        Calculates the sampler acceptance rate

        :return: Sampler acceptance rate
        :rtype: float
        """

        return np.mean(self.acceptances)

    def plot_histogram(self, figsize=(12, 8), bins=100, show=True, save=False):
        """
        Plots histogram(s) (for each parameter) of posterior. Visually estimates
        posterior distribution

        :param figsize: Size of the total figure
        (all histograms together), defaults to (12, 8)
        :type figsize: tuple, optional
        :param bins: Number of histogram bins, defaults to 100
        :type bins: int, optional
        :param show: Whether to show the figure at runtime, defaults to True
        :type show: bool, optional
        :param save: Whether to save the figure as `.png` file, defaults to False
        :type save: bool, optional
        """

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i + 1) for i in range(dim)]
        for i in range(dim):
            axs[i].hist(self.samples[:, i], bins=bins)
            axs[i].set_xlabel(f"theta_{i}")
            axs[i].set_ylabel("density")
            axs[i].set_title(f"theta_{i} histogram")
        fig.suptitle("Parameter Histograms")
        if save is True:
            plt.savefig("histograms.png", dpi=300)
        if show is True:
            plt.show()

    def parameter_kde(
        self, figsize=(12, 8), histogram=True, bins=100, show=True, save=False, **kwargs
    ):
        """
        Plots Kernel Density Estimation(s) (KDE) (for each parameter) of posterior. Visually
        estimates posterior distribution

        :param figsize: Size of the total figure (all plots together), defaults to (12, 8)
        :type figsize: tuple, optional
        :param histogram: Whether to overlay histogram over KDE plots, defaults to True
        :type histogram: bool, optional
        :param bins: Number of histogram bins, defaults to 100
        :type bins: int, optional
        :param show: Whether to show the figure at runtime, defaults to True
        :type show: bool, optional
        :param save: Whether to save the figure as `.png` file, defaults to False
        :type save: bool, optional
        """

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i + 1) for i in range(dim)]
        for i in range(dim):
            if histogram is True:
                axs[i].hist(self.samples[:, i], bins=bins)
            sns.kdeplot(self.samples[:, i], ax=axs[i], **kwargs)
            axs[i].set_xlabel(f"theta_{i}")
            axs[i].set_ylabel("density")
            axs[i].set_title(f"theta_{i} kde plot")
        fig.suptitle("Parameter KDE Plots")
        if save is True:
            plt.savefig("kde_plots.png", dpi=300)
        if show is True:
            plt.show()

    def traceplots(self, figsize=(12, 8), show=True, save=False, **kwargs):
        """
        Plots traceplot(s) (for each parameter)

        :param figsize: Size of the total figure (all plots together), defaults to (12, 8)
        :type figsize: tuple, optional
        :param show: Whether to show the figure at runtime, defaults to True
        :type show: bool, optional
        :param save: Whether to save the figure as `.png` file, defaults to False
        :type save: bool, optional
        """

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i + 1) for i in range(dim)]
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

    def plot_acf(self, figsize=(12, 8), show=True, save=False, **kwargs):
        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i + 1) for i in range(dim)]
        for i in range(dim):
            plot_acf(self.samples[:, i], ax=axs[i], **kwargs)
            axs[i].set_xlabel("iteration")
            axs[i].set_ylabel(f"theta_{i}")
            axs[i].set_title(f"theta_{i} ACF")
        fig.suptitle("Autocorrelation Plot")
        if save is True:
            plt.savefig("parameter_acf.png", dpi=300)
        if show is True:
            plt.show()
