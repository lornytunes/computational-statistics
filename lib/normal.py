import numpy as np
from scipy import stats

class Normal:
    
    def __init__(self, mu: int, sigma2: float):
        """Initializes.

        mu: mean
        sigma2: variance
        """
        self.mu = mu
        self.sigma2 = sigma2
        
    @property
    def sigma(self) -> float:
        return np.sqrt(self.sigma2)
    
    
    def sum(self, n: int):
        """Returns the distribution of the sum of n values.

        n: int

        returns: new Normal
        """
        return self.__class__(n * self.mu, n * self.sigma2)
    
    def __add__(self, other):
        """Adds a number or other Normal.

        other: number or Normal

        returns: new Normal
        """
        if isinstance(other, type(self)):
            return self.__class__(self.mu + other.mu, self.sigma2 + other.sigma2)
        else:
            return self.__class__(self.mu + other, self.sigma2)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtracts a number or other Normal.

        other: number or Normal

        returns: new Normal
        """
        if isinstance(other, type(self)):
            return self.__class__(self.mu - other.mu, self.sigma2 + other.sigma2)
        else:
            return self.__class__(self.mu - other, self.sigma2)

    __rsub__ = __sub__
    
    def __mul__(self, factor):
        """Multiplies by a scalar.

        factor: number

        returns: new Normal
        """
        return self.__class__(factor * self.mu, factor**2 * self.sigma2)
    __rmul__ = __mul__
    
    def __div__(self, divisor):
        """Divides by a scalar.

        divisor: number

        returns: new Normal
        """
        return 1 / divisor * self
    __truediv__ = __div__
    
    def __str__(self):
        return f'N({self.mu:.2f}, {self.sigma2:.4f})'
    __repr__ = __str__
    
    def prob(self, x):
        """Cumulative probability of x.

        x: numeric
        """
        return stats.norm(loc=self.mu, scale=self.sigma).cdf(x)
    
    def percentile(self, p):
        """Inverse CDF of p.

        p: percentile rank 0-100
        """
        return stats.norm(loc=self.mu, scale=self.sigma).ppf(p)
    
    def render(self, n: int = 101):
        """Returns pair of xs, ys suitable for plotting.
        """
        mean, std = self.mu, self.sigma
        low, high = mean - 3 * std, mean + 3 * std
        xs = np.linspace(low, high, n)
        ps = stats.norm.cdf(xs, mean, std)
        return xs, ps