from typing import List, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np


class HypothesisTest(ABC):
    '''
    A class that represents the structure of a classical hypothesis test
    '''
    
    def __init__(self, data: List[float]):
        self.data = data
        self.make_model()
        self.actual = self.test_statistic(data)
    
    @abstractmethod
    def test_statistic(self, data: List[float]) -> float:
        '''
        Provides the test statistic of interest
        '''
        
    @abstractmethod
    def make_model(self):
        '''
        Sets up the test
        '''
        
    @abstractmethod
    def run_model(self) -> List[float]:
        '''
        Runs the test - generates the data to pass to test_statistic
        '''
        
    def p_value(self, iters=1000):
        '''
        Computes the p-value
        '''
        self.test_stats = np.array([
            self.test_statistic(self.run_model()) for _ in range(iters)
        ])
        # proportion of stats greater than the actual value
        return sum(self.test_stats >= self.actual) / iters
    
    
class DiceTest(HypothesisTest):
    """Tests whether a six-sided die is fair."""
    
    FACES = [1, 2, 3, 4, 5, 6]

    def test_statistic(self, data):
        # data is what has been observed
        n = sum(data)
        # array of 1/6 values the equal to the number of observations
        expected = np.ones(6) * n / 6
        return sum(abs(data - expected))
    
    def make_model(self):
        pass

    def run_model(self):
        n = sum(self.data)
        rolls = np.random.choice(self.FACES, n, replace=True)
        hist = Counter(rolls)
        # the frequencies are the values. return them in order of the dice values 1-6
        return np.array([hist[i] for i in self.FACES])
    

class DiceChiTest(DiceTest):
    """Tests a six-sided die using a chi-squared statistic."""

    def test_statistic(self, data):
        """Computes the test statistic.

        data: list of observed frequencies
        """
        n = sum(data)
        expected = np.ones(6) * n / 6
        return sum((data - expected)**2 / expected)


@dataclass
class GroupPair:
    
    group1: np.array
    group2: np.array
    
    @property
    def lengths(self) -> Tuple[int, int]:
        return (len(self.group1), len(self.group2))
    
    @property
    def means(self) -> Tuple[float, float]:
        return (self.group1.mean(), self.group2.mean())
    

def mean_diff(gp: GroupPair) -> float:
    return abs(gp.group1.mean() - gp.group2.mean())


def pooled_sampler(data: GroupPair) -> GroupPair:
    # represents null hypothesis
    n, m = data.lengths
    pool = np.hstack((data.group1, data.group2))
    # shuffle the pool
    np.random.shuffle(pool)
    # return as a new grouped pair using the same sizes as the actual sample
    return GroupPair(pool[:n], pool[n:])


# null hypothesis for correlated pairs
def permutation_sampler(gp: GroupPair):
    return GroupPair(
        np.random.permutation(gp.group1),
        gp.group2
    )


def run_model(data: GroupPair, test_stat: Callable = mean_diff, sampler: Callable = pooled_sampler, niters: int = 1000) -> np.ndarray:
    return np.array([
        test_stat(sampler(data)) for i in range(niters)
    ])


def p_value(test_stats: np.ndarray, actual: float) -> float:
    '''
    the proportion of differences that exceed the observed difference
    '''
    return sum(test_stats >= actual) / len(test_stats)