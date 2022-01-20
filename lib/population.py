import numpy as np
import pandas as pd

def read_population(filename='data/population.csv') -> pd.Series:
    df = pd.read_csv(
        filename,
        header=None,
        skiprows=2,
        encoding='iso-8859-1'
    )
    # the population totals are in the 8th column
    populations = df[7]
    populations.name = 'population'
    populations.replace(0, np.nan, inplace=True)
    return populations.dropna()