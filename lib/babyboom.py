import pandas as pd

from fwf import Column


def read_baby_boom(filename='data/babyboom.dat') -> pd.DataFrame:
    var_info = [
        Column(1, int, 'time', end=8),
        Column(9, int, 'sex', end=16),
        Column(17, int, 'weight_g', end=24),
        Column(25, int, 'minutes', end=32)
    ]
    return pd.read_fwf(
        filename,
        width = [c.width for c in var_info],
        names = [c.name for c in var_info],
        dtype = dict([(c.name, c.vtype,) for c in var_info]),
        skiprows=59
    )