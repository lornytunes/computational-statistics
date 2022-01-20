import numpy as np
import pandas as pd

import fwf

def read_brfss(datafile='data/brfss.dat.gz', nrows=None) -> pd.DataFrame:
    var_info = fwf.read_schema([
        ('age', 101, 102, pd.Int64Dtype()),
        ('sex', 143, 143, int),
        ('wtyrago', 127, 130, float),
        ('finalwt', 799, 808, int),
        ('wtkg2', 1254, 1258, float),
        ('htm3', 1251, 1253, pd.Int64Dtype()),
    ])
    
    df = fwf.read_fixed_width(
        datafile,
        var_info,
        include_dtypes=True,
        nrows=nrows
    )
    # clean height
    df.htm3.replace([999], pd.NA, inplace=True)
    # clean weight
    df.wtkg2.replace([99999], np.nan, inplace=True)
    # convert weight to kg
    df.wtkg2 /= 100
    # weight a year ago
    df.wtyrago.replace([7777, 9999], np.nan, inplace=True)
    df['wtyrago'] = df.wtyrago.apply(lambda x: x/2.2 if x < 9000 else x-9000)
    # clean age
    df.age.replace([7, 9], pd.NA, inplace=True)
    return df.astype({
        'age': pd.Int64Dtype(),
        'htm3': pd.Int64Dtype()
    }).rename(columns={
        'htm3': 'height',
        'wtkg2': 'weight'
    })
    