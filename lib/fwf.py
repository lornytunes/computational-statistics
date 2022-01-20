import re
from typing import List, Tuple
from dataclasses import dataclass, field

import pandas as pd

# for stata schema lines
COLUMN_IDENTIFIER = re.compile(r'_column\(([\d]+)\)')
QUOTED_VALUE = re.compile(r'"([^"]+)"')

# maps stata schema types to python types
TYPE_MAP = dict(
    byte=int,
    int=int,
    long=int,
    float=float, 
    double=float,
    numeric=float
)


@dataclass
class Column:
    # start position
    start: int
    # type of data
    vtype: type
    # name of the column
    name: str
    # end position - we may not know this initially
    end: int = field(default=0)
    fstring: str = field(default=None, repr=False)
    description: str = field(default=None, repr=False)
        
    def col_spec(self, start_index=0) -> Tuple[int, int]:
        '''
        Returns the start and end positions, with possible correction for zero or one based start
        '''
        return (self.start - start_index, self.end - start_index)
    
    @property
    def width(self) -> int:
        '''
        Returns the total width (in characters) of this column
        '''
        return (self.end - self.start)
    
    
def read_stata_column(line: str) -> Column:
    column_match = COLUMN_IDENTIFIER.search(line)
    description_match = QUOTED_VALUE.search(line)
    # the end of the match is the start of the line we want
    _, s_start = column_match.span(0)
    # the start of the description match is the end of the portion of the rest of the line
    s_end, _ = description_match.span(0)
    # get the captured values
    position = int(column_match.groups()[0])
    description = description_match.groups()[0]
    # get the three remaining values
    vtype, name, fstring = line[s_start:s_end].split()
    # return them
    return Column(
        position,
        TYPE_MAP.get(vtype, str),
        name.lower(),
        fstring=fstring,
        description=description
    )


def read_stata_dictionary(filepath) -> List[Column]:
    columns = []
    # record both starting position of each column
    with open(filepath) as fp:
        for line in fp:
            if '_column' not in line:
                # doesn't contain any data
                continue
            columns.append(read_stata_column(line))
    # work out the end positions. Start with all columns except the first
    for i in range(1, len(columns)):
        # [start, end), e.g [1, 13],[13, 14]..
        columns[i-1].end = columns[i].start
        
    return columns


def read_schema(schema: List[Tuple[str, int, int, type]]) -> List[Column]:
    '''
    Parses out a list of Columns when the start and end positions are known
    '''
    columns = []
    for name, start, end, t in schema:
        columns.append(Column(start, t, name, end=end+1))
    return columns
        


def read_fixed_width(
    data_file: str, columns: List[Column],
    include_dtypes=False, **kwargs) -> pd.DataFrame:
    # options to pass to read_fwf
    options = {}
    # is it compressed
    if data_file.endswith('.gz'):
        options['compression'] = 'gzip'
    if include_dtypes:
        options['dtype'] = dict([
            (c.name, c.vtype) for c in columns
        ])
    if kwargs:
        options.update(kwargs)
    # zero based indexing
    index_base=1
    col_specs = [c.col_spec(index_base) for c in columns]
    col_names = [c.name for c in columns]
    return pd.read_fwf(
        data_file,
        colspecs=col_specs,
        names=col_names,
        **options
    )
    

def read_stata_fixed_width(dct_file: str, data_file: str, nrows=None) -> pd.DataFrame:
    return read_fixed_width(data_file, read_stata_dictionary(dct_file), nrows)
    