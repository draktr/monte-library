"""
Module containing base sampler for Monte Carlo
sampling methods containing some basic utility methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.stats.weightstats import ztest
from scipy.stats import norm, chi2, cramervonmises


class BaseSampler:
    def __init__(self) -> None:
        """
        Base sampler for Monte Carlo sampling methods containing basic utility methods.
        Initializes the sampler with two main attributes: samples and acceptances
        which are saved here at each iteration
        """

        self._samples = None
        self._acceptances = None
        self._lp = None

    @property
    def samples(self):
        """
        Gets or sets samples property

        :return: Samples property
        :rtype: ndarray
        """

        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def acceptances(self):
        """
        Gets or sets acceptances property

        :return: Acceptances property
        :rtype: ndarray
        """

        return self._acceptances

    @acceptances.setter
    def acceptances(self, acceptances):
        self._acceptances = acceptances

    @property
    def lp(self):
        """
        Gets or sets log-probability (lp) property

        :return: Log-probability (lp) property
        :rtype: ndarray
        """

        return self._lp

    @lp.setter
    def lp(self, lp):
        self._lp = lp

    def save_samples(self, parameter_names=None, path=None):
        """
        Exports samples as `.csv` file.

        :param path: Path where to save the `.csv` file. By default, it saves to the
                     working directory, defaults to None
        :type path: str, optional
        """

        if parameter_names is None:
            column_names = np.array(
                np.concatenate(
                    (
                        np.array(["lp"]),
                        np.array([f"theta_{i}" for i in range(self.samples.shape[1])]),
                    ),
                    axis=None,
                )
            )
        else:
            if parameter_names.shape[0] != self.sample.shape[1]:
                raise ValueError(
                    "There should be one and only one parameter name per parameter"
                )
            else:
                column_names = np.array(
                    np.concatenate((np.array(["lp"]), parameter_names), axis=None)
                )

        pd.DataFrame(
            np.concatenate(
                (np.reshape(self.lp, (self.lp.shape[0], 1)), self.samples), axis=1
            ),
            columns=column_names,
        ).to_csv(path)

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

    def quantiles(self, quantiles=np.array([0.25, 0.5, 0.75])):
        """
        Calculates posterior quantiles.

        :param quantiles: Quantiles, defaults to np.array([0.25, 0.5, 0.75])
        :type quantiles: ndarray, optional
        :return: Parameter values at quantiles
        :rtype: ndarray
        """

        return np.quantile(self.samples, quantiles)

    def credible_interval(self, ci=0.9):
        """
        Calculates

        .. math::
            100 \\times \\text{ci} \%

        credible interval for each parameter

        :param ci: Credible interval in decimal, defaults to 0.9
        :type ci: float, optional
        :return: Parameter values at credible interval
        :rtype: ndarray
        """

        tail = (1 - ci) / 2
        return np.quantile(self.samples, np.array([0 + tail, 1 - tail]))

    def plot_histogram(self, figsize=(12, 8), bins=100, show=True, save=False):
        """
        Plots histogram(s) (for each parameter) of posterior. Visually estimates
        posterior distribution

        :param figsize: Size of the total figure (all histograms together), defaults to (12, 8)
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

    def plot_histogram_fold(
        self, k=2, style="default", rcParams_update={}, show=True, save=False
    ):
        """
        Plots histograms of `k`-subsets of the samples

        :param k: Number of subsets taken from the Markov chain for histogram plotting,
                defaults to 2
        :type k: int, optional
        :param style: `matplotlib` style to be used for plots. User can pass
                    built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                    or a path to a custom style defined in a `.mplstyle` document,
                    defaults to "default"
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot, defaults to False
        :type save: bool, optional
        """

        plt.style.use(style)
        plt.rcParams.update({"figure.figsize": (3, 3 * k)})
        plt.rcParams.update(**rcParams_update)
        fig, axs = plt.subplots(nrows=k, ncols=self.samples.shape[1])
        for i in range(k):
            for j in range(self.samples.shape[1]):
                self.samples[
                    int(i * self.samples.shape[0] / k) : int(
                        (i + 1) * self.samples.shape[0] / k, j
                    )
                ].hist(ax=axs[i, j])
                axs[i, j].set_xlabel("Value")
                axs[i, j].set_ylabel("Frequency")
                axs[i, j].set_title("Probability Density (histogram)")
        fig.suptitle("All Chains Histograms")
        fig.tight_layout()
        plt.show()
        if save:
            plt.savefig("hist_comp.png")
        if show:
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
        """
        Plots autocorrelation function values of the parameter samples

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
            tsaplots.plot_acf(self.samples[:, i], ax=axs[i], **kwargs)
            axs[i].set_xlabel("iteration")
            axs[i].set_ylabel(f"theta_{i}")
            axs[i].set_title(f"theta_{i} ACF")
        fig.suptitle("Autocorrelation Plot")
        if save is True:
            plt.savefig("parameter_acf.png", dpi=300)
        if show is True:
            plt.show()

    def ergodic_mean(self):
        """
        Calculates ergodic mean(s) for each parameter. Ergodic mean refers
        to the mean value until the current iteration

        :return: Ergodic mean
        :rtype: ndarray
        """

        ergodic_mean = np.zeros(self.samples.shape[0])
        for i in range(self.samples.shape[0]):
            ergodic_mean[i] = np.mean(self.samples[:i], axis=0)
        return ergodic_mean

    def plot_ergodic_mean(self, figsize=(12, 8), show=True, save=False, **kwargs):
        """
        Plots ergodic mean(s) (for each parameter)

        :param figsize: Size of the total figure (all plots together), defaults to (12, 8)
        :type figsize: tuple, optional
        :param show: Whether to show the figure at runtime, defaults to True
        :type show: bool, optional
        :param save: Whether to save the figure as `.png` file, defaults to False
        :type save: bool, optional
        """

        ergodic_mean = self.ergodic_mean()

        dim = self.samples.shape[1]
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        axs = [plt.subplot(dim, 1, i + 1) for i in range(dim)]
        for i in range(dim):
            axs[i].plot(ergodic_mean[:, i], **kwargs)
            axs[i].set_xlabel("iteration")
            axs[i].set_ylabel(f"theta_{i}")
            axs[i].set_title(f"theta_{i} ergodic mean")
        fig.suptitle("Parameter erogidc means")
        if save is True:
            plt.savefig("ergodic_means.png", dpi=300)
        if show is True:
            plt.show()

    def ess(self):
        """
        Calculates effective sample size (for each parameter) as per
        Kass et al (1998) and Robert and Casella (2004; pg.500)

        :return: Effective sample size
        :rtype: ndarray
        """

        ess = np.empty((self.samples.shape[1]))
        for param in range(self.samples.shape[1]):
            ess[param] = self.samples.shape[0] / (
                1 + 2 * np.sum(acf(self.samples[:, param])[1:])
            )

        return ess

    def geweke(self, first=0.1, last=0.5):
        """
        Calculates Geweke diagnostic for a Markov chain

        :param first: The fraction of the chain to use for the first segment, defaults to 0.1
        :type first: float, optional
        :param last: The fraction of the chain to use for the last segment, defaults to 0.5
        :type last: float, optional
        :return: Test Z-scores and p-values
        :rtype: float, float
        """

        n = self.samples.shape[0]
        n_params = self.samples.shape[1]
        z_score = np.empty(n_params)
        p_value = np.empty(n_params)

        for param in range(self.samples.shape[1]):
            chain = self.samples[:, param]
            first_segment = chain[: int(first * n)]
            last_segment = chain[-int(last * n) :]

            z_score[param], p_value[param] = ztest(x1=first_segment, x2=last_segment)

        return z_score, p_value

    def heidelberger_welch(self, alpha=0.05):
        """
        Calculates Heidelberger-Welch diagnostic for a Markov chain

        :param alpha: Test significance level, defaults to 0.05
        :type alpha: float, optional
        :return: Diagnostic results, including: stationarity test result,
                halfwidth test result, initial chain mean,
                updated chain start, chain end, updated chain mean
        :rtype: dict
        """

        n = self.samples.shape[0]
        n_params = self.samples.shape[1]

        results = {
            "stationarity": np.repeat(False, n_params),
            "halfwidth": np.repeat(False, n_params),
            "mean": np.mean(self.samples, axis=0),
            "start": np.zeros(n_params),
            "end": np.repeat(n, n_params),
            "mean_final": np.repeat(None, n_params),
        }

        # Stationarity test
        for param in range(n_params):
            for start in range(0, n, n // 10):
                current_chain = self.samples[start:, param]
                sorted_data = np.sort(current_chain)
                cvm_result = cramervonmises(sorted_data, "uniform")

                if cvm_result.pvalue > alpha:
                    results["stationarity"][param] = True
                    results["start"][param] = start
                    break

            if not results["stationarity"][param]:
                return results

            # Halfwidth test
            final_chain = self.samples[results["start"][param] :, param]
            n_final = len(final_chain)
            mean_final = np.mean(final_chain)
            var_final = np.var(final_chain, ddof=1)
            se_final = np.sqrt(var_final / n_final)

            z = chi2.ppf(1 - alpha / 2, df=1)
            halfwidth = z * se_final

            if halfwidth < alpha * abs(mean_final):
                results["halfwidth"][param] = True

            results["mean_final"][param] = mean_final
            results["end"][param] = n

        return results

    def raftery_lewis(self, q=0.025, r=0.005, s=0.95):
        """
        Calculates Raftery-Lewis diagnostic

        :param q: The quantile of interest, defaults to 0.025 for a lower tail quantile
        :type q: float, optional
        :param r: The accuracy required, defaults to 0.005
        :type r: float, optional
        :param s: The probability of achieving the specified accuracy, defaults to 0.95
        :type s: float, optional
        :return: Diagnostic results and specifications, including: quantile of interest,
                requires samples, burn-in samples, total samples, accuracy required,
                probability of reaching that accuracy
        :rtype: dict
        """

        n_params = self.samples.shape[1]
        results = {
            "quantile": np.repeat(q, n_params),
            "required_samples": np.repeat(None, n_params),
            "burn_in": np.repeat(None, n_params),
            "total_samples": np.repeat(None, n_params),
            "accuracy": np.repeat(r, n_params),
            "probability": np.repeat(s, n_params),
        }

        for param in range(n_params):
            quantile_estimate = np.percentile(self.samples[:, param], q * 100)
            exceedance = self.samples[:, param] >= quantile_estimate
            m = np.mean(exceedance)
            z = norm.ppf((1 + s) / 2)

            n_required = int((z**2 * m * (1 - m)) / (r**2))
            burn_in = np.where(exceedance == 1)[0][0] if np.any(exceedance) else 0
            total_samples = burn_in + n_required

            results["required_samples"][param] = n_required
            results["burn_in"][param] = burn_in
            results["total_samples"][param] = total_samples

        return results

    def stationarity_test(self, test="adf", **kwargs):
        """
        Test for the stationarity of Markov chains

        :param test: Statistical test used. Available are Augmented Dickey-Fuller test (`adf`)
                    and Kwiatkowski–Phillips–Schmidt–Shin test (`kpss`), defaults to "adf"
        :type test: str, optional
        :return: Test statistic, p-value
        :rtype: float, float
        """

        n_params = self.samples.shape[1]
        results = np.empty((n_params, 2))

        if test == "adf":
            for param in range(n_params):
                results[param] = adfuller(
                    self.samples[:, self.samples.shape[1]], **kwargs
                )
            return results[param, 0], results[param, 1]
        elif test == "kpss":
            for param in range(n_params):
                results[param] = kpss(self.samples[:, self.samples.shape[1]], **kwargs)
            return results[param, 0], results[param, 1]
        else:
            ValueError("Test unavailable. Available tests are `adf` and `kspp`.")
