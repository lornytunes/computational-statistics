import bisect
from typing import Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd


class Cdf:
    
    @classmethod
    def from_hist(cls, values, freqs):
        # convert frequencies to probabilities
        probs = np.cumsum(freqs, dtype=np.float64)
        # and normalize
        probs /= probs[-1]
        # call constructor
        return cls(np.asarray(values), probs)
    
    @classmethod
    def from_dict(cls, data: Dict):
        values, freqs = zip(*sorted(data.items()))
        return cls.from_hist(values, freqs)
    
    @classmethod
    def from_seq(cls, x):
        # x may contain duplicates
        values, freqs = zip(*sorted(Counter(x).items()))
        return cls.from_hist(values, freqs)
    
    def __init__(self, xs: np.ndarray, ps: np.ndarray):
        # values
        self.xs = xs
        # cumulative probabilities
        self.ps = ps
        
    
    def prob(self, x) -> float:
        """
        Returns CDF(x), the probability that corresponds to value x.

        Args:
            x: number

        Returns:
            float probability
        """
        if x < self.xs[0]:
            return 0
        # find x in our values
        index = bisect.bisect(self.xs, x)
        # return the corresponding probability
        return self.ps[index-1]
    
    def probs(self, xs: np.array) -> np.array:
        """
        Gets probabilities for a sequence of values.

        xs: any sequence that can be converted to NumPy array

        returns: NumPy array of cumulative probabilities
        """
        index = np.searchsorted(self.xs, xs, side='right')
        ps = self.ps[index-1]
        # anything less than our smallest value is zero
        ps[xs < self.xs[0]] = 0
        return ps
    
    def value(self, p: np.float64):
        """
        Returns InverseCDF(p), the value that corresponds to probability p.

        Args:
            p: number in the range [0, 1]

        Returns:
            number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        index = bisect.bisect_left(self.ps, p)
        return self.xs[index]
    
    def values(self, ps) -> np.array:
        """
        Returns InverseCDF(p), the value that corresponds to probability p.

        Args:
            ps: NumPy array of numbers in the range [0, 1]

        Returns:
            NumPy array of values
        """

        ps = np.asarray(ps)
        
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('Probability p must be in range [0, 1]')

        indices = np.searchsorted(self.ps, ps, side='left')
        return self.xs[indices]
    
    def percentile(self, p: np.int64) -> np.number:
        """
        Returns the value that corresponds to percentile p (between 0 and 100).

        Args:
            p: number in the range [0, 100]

        Returns:
            number value
        """
        return self.value(p / 100)
    
    def rank(self, x) -> np.number:
        """
        Returns the percentile rank of the value x.

        x: potential value in the CDF

        returns: percentile rank in the range 0 to 100
        """
        return self.prob(x) * 100
    
    def sample(self, n: int):
        """
        Returns a list of n values chosen at random from the cdf.
        
        n: int length of the sample
        returns: NumPy array
        """
        
        # n values between 0 and 1
        ps = np.random.random(n)
        # get the values they correspond to
        return self.values(ps)
    
    def mean(self) -> float:
        """
        Computes the mean of a CDF.
        
        sum(x * diff(p))

        Returns:
            float mean
        """
        old_p = 0
        total = 0
        for x, new_p in zip(self.xs, self.ps):
            # compute dp
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total
    
    def ci(self, percentage=90) -> Tuple[float, float]:
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100) / 2
        return self.value(prob), self.value(1 - prob)
    
    def items(self):
        """
        Returns a sorted sequence of (value, probability) pairs.
        """
        a = self.ps
        # shift probabilities one place to the right
        # e.g [0, 0.1, 0.2, 0.3] becomes [0.3, 0, 0.1, 0.2]
        b = np.roll(a, 1)
        b[0] = 0
        return zip(self.xs, a-b)
    
    @property
    def series(self) -> pd.Series:
        '''
        Return our value and cumulative probabilities as a pandas series
        '''
        return pd.Series(data=self.ps, index=self.xs)
    
    def copy(self):
        return self.__class__(self.xs.copy(), self.ps.copy())
    
    def complement(self):
        '''
        Returns 1-CDF
        '''
        return self.__class__(self.xs.copy(), 1-self.ps)
    
    def scale(self, factor):
        """
        Multiplies the xs by a factor.

        factor: what to multiply by
        """
        return self.__class__(self.xs * factor, self.ps.copy())
    
    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(
            x=self.xs,
            p=self.ps
        ))
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, x):
        return self.prob(x)
    
    
def resample_rows_weighted(df: pd.DataFrame, column='finalwgt') -> pd.DataFrame:
    weights = df[column]
    cdf = Cdf.from_dict(dict(weights))
    indices = cdf.sample(len(weights))
    return df.loc[indices]