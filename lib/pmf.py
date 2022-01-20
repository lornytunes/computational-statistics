import bisect
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

class Pmf:
    
    @classmethod
    def from_seq(cls, values: List[int]):
        return cls(pd.Series(values).value_counts().sort_index())
    
    @classmethod
    def from_dict(cls, data: Dict[int, int]):
        return cls(pd.Series(data=data.values(), index=data.keys()))
        
    
    def __init__(self, series: pd.Series, normalize=True):
        # compute the frequencies
        self._series = series
        # compute the range of x values
        self._min = self._series.index.min()
        self._max = self._series.index.max()
        # normalize the frequencies into probabilities
        if normalize:
            self.normalize()
        
    def normalize(self):
        '''
        Normalizes this Pmf so the sum of all probabilities is 1
        '''
        # divide through by the sum of the values
        self._series /= np.sum(self._series)
        
    def incr(self, x: int, term: np.float):
        '''
        Increments the freq/prob associated with the value x
        '''
        if x in self._series:
            self._series[x] += term
            
    def mult(self, x: int, factor: np.float):
        '''
        Scales the freq/prob associated with the value x
        '''
        if x in self._series:
            self._series[x] *= factor
            
    def prob(self, x: int):
        '''
        Gets the probability associated with the value x
        '''
        return self._series.get(x, 0)
    
    @property
    def total(self) -> np.float:
        return np.sum(self._series)
            
    def __getitem__(self, x):
        '''
        Implements the indexing operator
        '''
        return self.prob(x)
    
    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(probs=self.series))
        
    @property
    def series(self) -> pd.Series:
        return self._series
    
    @property
    def min(self) -> int:
        # return the smallest value
        return self._min
    
    @property
    def max(self) -> int:
        # return the largest value
        return self._max

    def mean(self) -> np.float64:
        '''
        Computes the mean of a PM
        '''
        return sum(p * x for x, p in self.items())
    
    def var(self) -> np.float64:
        """
        Computes the variance of a Pmf.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float variance
        """
        mu = self.mean()
        return sum(p * (x-mu)**2 for x, p in self.items())
    
    def mode(self):
        # returns the value with the highest probability
        return self.series.index[np.argmax(remaining.series.values)]
    
    def arange(self, increment=1) -> np.array:
        # include missing
        return np.arange(self._min, self._max+1, increment)

    @property
    def probs(self, xs) -> np.array:
        return np.array([self[x] for x in xs])
    
    @property
    def values(self) -> np.array:
        return self._series.index.values
    
    @property
    def probabilities(self) -> np.array:
        return self._series.values

    def items(self) -> List[Tuple[int, float]]:
        return self._series.iteritems()
    
    def copy(self):
        return self.__class__(self._series.copy(), normalize=False)
    
    def as_dict(self) -> dict:
        return dict(zip(self._series.index, self._series.values))
    
    def add(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = {}
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                val = v1 + v2
                if val not in pmf:
                    pmf[val] = 0
                pmf[val] += p1 * p2
        return self.__class__.from_dict(pmf)
    
    def sub(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = {}
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                val = v1 - v2
                if val not in pmf:
                    pmf[val] = 0
                pmf[val] += p1 * p2
        return self.__class__.from_dict(pmf)
    
    def add_constant(self, value):
        """Computes the Pmf of the sum a constant and values from self.

        value: a number

        returns: new Pmf
        """
        if value == 0:
            return self.copy()

        pmf = {}
        for v1, p1 in self.items():
            pmf[v1 + value] = p1
        return self.__class__.from_dict(pmf)
    
    def __sub__(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        if isinstance(other, type(self)):
            # its another Pmf - subtract the two
            return self.sub(other)
        else:
            return self.add_constant(-other)
    
    def __getitem__(self, x) -> np.float:
        if x in self._series:
            return self._series[x]
        return 0
    
    def __setitem__(self, x, p):
        if x in self._series:
            self._series[x] = p
    
    def __str__(self):
        return str(self._series)
    
    
def prob_greater(pmf1: Pmf, pmf2: Pmf) -> float:
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 > v2:
                total += p1 * p2
    return total


def prob_equal(pmf1: Pmf, pmf2: Pmf) -> float:
    """Probability that a value from pmf1 equals a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 == v2:
                total += p1 * p2
    return total