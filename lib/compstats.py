from typing import Tuple

import numpy as np
import pandas as pd


def distribution_midpoint(s1: np.array, s2: np.array) -> np.float64:
    '''
    Computes the midpoint between the two distribution means
    '''
    return (s1.mean() + s2.mean()) / 2


def distribution_intersection(s1: np.array, s2: np.array) -> np.float64:
    '''
    Compute the point where two pdfs cross
    '''
    sd1 = s1.std()
    sd2 = s2.std()
    return (sd1 * s2.mean() + sd2 * s1.mean()) / (sd1 + sd2)


def pooled_std(s1: np.array, s2: np.array) -> np.float64:
    '''
    Computes the pooled standard deviation of two samples
    '''
    n1, n2 = len(s1), len(s2)
    return np.sqrt((n1 * s1.var() + n2 * s2.var()) / (n1 + n2))


def cohen(group1: np.array, group2: np.array) -> np.float64:
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    """
    return (group1.mean() - group2.mean()) / pooled_std(group1, group2)


def var(xs: np.array, mu=None, ddof=0):
    """Computes variance.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """

    if mu is None:
        mu = xs.mean()

    ds = xs - mu
    return np.dot(ds, ds) / (len(xs) - ddof)


def trim(t: np.array, p=0.01) -> np.array:
    """Trims the largest and smallest elements of t.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        sorted sequence of values
    """
    # interpret the proportion in terms of the number of elements in t
    n = int(p * len(t))
    return np.array(sorted(t)[n:-n])


def trimmed_mean(t: np.array, p=0.01) -> np.float64:
    """Computes the trimmed mean of a sequence of numbers.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    return trim(t, p).mean()


def trimmed_mean_var(t, p=0.01) -> Tuple[np.float64, np.float64]:
    """Computes the trimmed mean and variance of a sequence of numbers.

    Side effect: sorts the list.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    t = trim(t, p)
    return t.mean(), var(t)


def moment(xs: np.array, k: int) -> np.float64:
    """
    Computes the kth moment of xs.
    """
    return np.sum(xs**k) / len(xs)


def central_moment(xs: np.array, k: int) -> np.float64:
    """
    Computes the kth central moment of xs.
    """
    mean = moment(xs, 1)
    return np.sum((xs - mean) ** k) / len(xs)


def standardized_moment(xs: np.array, k: int) -> np.float64:
    """
    Computes the kth standardized moment of xs.
    """
    std = np.sqrt(central_moment(xs, 2))
    return central_moment(xs, k) / std**k


def pearson_median_skewness(xs: np.array) -> np.float64:
    """
    Computes the Pearson median skewness.
    """
    median = np.median(xs)
    mean = moment(xs, 1)
    std = np.sqrt(central_moment(xs, 2))
    return 3 * (mean - median) / std


def resample_n(xs: np.array, n: int) -> np.array:
    return np.random.choice(xs, n, replace=True)


def resample(xs: np.array) -> np.array:
    return np.random.choice(xs, len(xs), replace=True)


def sample_rows(df: pd.DataFrame, nrows: int, replace=False) -> pd.DataFrame:
    """Choose a sample of rows from a DataFrame.

    df: DataFrame
    nrows: number of rows
    replace: whether to sample with replacement

    returns: DataFrame
    """
    if nrows is None:
        nrows = len(df)
    return df.loc[np.random.choice(df.index, nrows, replace=replace)]


def resample_rows(df: pd.DataFrame, nrows: int) -> pd.DataFrame:
    return sample_rows(df, nrows, replace=True)


def var(xs: np.array, mu=None, ddof=0) -> float:
    """Computes variance.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """

    if mu is None:
        mu = xs.mean()
    ds = xs - mu
    return np.dot(ds, ds) / (len(xs) - ddof)


def std(xs: np.array, mu=None, ddof=0) -> float:
    """Computes standard deviation.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """
    return np.sqrt(var(xs, mu, ddof))


def cov(xs: np.array, ys: np.array, meanx=None, meany=None):
    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


def corr(xs: np.array, ys: np.array):
    """
    Computes Corr(X, Y).

    Args:
        xs: sequence of values
        ys: sequence of values

    Returns:
        Corr(X, Y)
    """
    meanx = xs.mean()
    meany = ys.mean()

    return cov(xs, ys, meanx, meany) / np.sqrt(xs.var() * ys.var())


def percentile_row(array: np.array, p: float) -> np.array:
    """
    Selects the row from a sorted array that maps to percentile p.

    p: float 0--100

    returns: NumPy array (one row)
    """
    rows, cols = array.shape
    index = int(rows * p / 100)
    return array[index,]


def percentile_rows(ys_seq: np.array, percents: np.array):
    """
    Given a collection of lines, selects percentiles along vertical axis.

    For example, if ys_seq contains simulation results like ys as a
    function of time, and percents contains (5, 95), the result would
    be a 90% CI for each vertical slice of the simulation results.

    ys_seq: sequence of lines (y values)
    percents: list of percentiles (0-100) to select

    returns: list of NumPy arrays, one for each percentile
    """
    nrows = len(ys_seq)
    ncols = len(ys_seq[0])
    array = np.zeros((nrows, ncols))

    for i, ys in enumerate(ys_seq):
        array[i,] = ys
    array = np.sort(array, axis=0)
    rows = [percentile_row(array, p) for p in percents]
    return rows


def odds2p(p: np.array) -> np.array:
    '''
    Converts odds to probabilities
    '''
    return p / (1-p)

def p2odds(o: np.array) -> np.array:
    '''
    Converts probabilities to odds
    '''
    return o / (o + 1)


def normal_qq(ys: np.array) -> Tuple[np.array, np.array]:
    """Generates data for a normal probability plot.

    ys: sequence of values
    jitter: float magnitude of jitter added to the ys 

    returns: numpy arrays xs, ys
    """
    xs = np.random.normal(0, 1, len(ys))
    xs.sort()
    ys = ys.copy()
    ys.sort()
    return xs, ys


def fit_line(xs: np.array, intercept, slope) -> np.array:
    """Fits a straight line fit to the given data.

    xs: sequence of x (in sorted order)

    returns: a numpy array
    """
    return intercept + slope * xs


def rmse(estimates: np.ndarray, actual: np.float64) -> np.float64:
    '''
    Returns the square root of the mean of the squares of the errors
    '''
    return np.sqrt(((estimates-actual)**2).mean())