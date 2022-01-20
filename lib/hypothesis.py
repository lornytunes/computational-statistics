from typing import Tuple, Callable
from dataclasses import dataclass

import numpy as np


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


def run_model(data: GroupPair, test_stat: Callable = mean_diff, sampler: Callable = pooled_sampler, niters: int = 1000) -> np.ndarray:
    return np.array([
        test_stat(sampler(data)) for i in range(niters)
    ])


def p_value(test_stats: np.ndarray, actual: float) -> float:
    '''
    the proportion of differences that exceed the observed difference
    '''
    return sum(test_stats >= actual) / len(test_stats)