import os
from collections import defaultdict
from typing import DefaultDict, List

import numpy as np
import pandas as pd


def read_fem_preg(datadir='data') -> pd.DataFrame:
    '''
    Reads the NSFG pregnancy data
    '''
    return pd.read_feather(os.path.join(datadir, '2002FemPreg.feather'))


def read_live_fem_preg(datadir='data') -> pd.DataFrame:
    return pd.read_feather(os.path.join(datadir, 'live_births.feather'))
    


def read_fem_resp(datadir='data') -> pd.DataFrame:
    '''
    Reads the NSFG respondent data.
    '''
    return pd.read_feather(os.path.join(datadir, '2002FemResp.feather'))


def make_preg_map(df: pd.DataFrame) -> DefaultDict[int, List[int]]:
    """
    Make a map from caseid to list of preg indices.

    df: DataFrame

    returns: dict that maps from caseid to list of indices into `preg`
    """
    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    return d