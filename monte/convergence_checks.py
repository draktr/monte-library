"""
Module contains convergence checks functions that require data preprocessing.
Most often this means having multiple chains.
"""

import numpy as np


def gelman_rubin(chains):
    """
    Calculates Gelman-Rubin statistic (R-hat) for Markov chains

    :param chains: Markov chains used for the calculation
    :type chains: pd.DataFrame
    :return: Gelman-Rubin statistic (R-hat)
    :rtype: float
    """

    n_samples = chains.shape[0]

    chain_means = np.mean(chains, axis=0)
    between = n_samples * np.var(chain_means, ddof=1)
    within = np.mean(np.var(chains, axis=0, ddof=1))

    var_hat = (1 - 1 / n_samples) * within + (1 / n_samples) * between
    r_hat = np.sqrt(var_hat / within)

    return r_hat


def multivariate_gelman_rubin(chains):
    """
    Calculates multivariate Gelman-Rubin statistic (R-hat) for Markov chains


    :param chains: Markov chains used for the calculation with dimensions
                   (n_chains, n_samples, n_dim), where n_chains is the number
                   of chains, n_samples is the number of samples per chain,
                   and n_dim is the number of dimensions.
    :type chains: pd.DataFrame
    :return: Multivariate Gelman-Rubin statistic (R-hat)
    :rtype: float
    """

    m, n, d = chains.shape

    within = np.zeros((d, d))
    for i in range(m):
        within += np.cov(chains[i], rowvar=False)
    within /= m

    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means, axis=0)
    between = np.zeros((d, d))
    for i in range(m):
        mean_diff = (chain_means[i] - overall_mean).reshape(-1, 1)
        between += mean_diff @ mean_diff.T
    between *= n / (m - 1)

    eigenvalues = np.linalg.eigvals(np.matmul(np.linalg.inv(within), between))
    lambda_max = np.max(eigenvalues)

    # Compute the Multivariate Potential Scale Reduction Factor
    multivariate_r_hat = np.sqrt((d - 1) / d + (m + 1) / m * lambda_max)

    return multivariate_r_hat
