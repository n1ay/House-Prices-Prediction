from random import gauss
from pandas import isnull

def replace_NA_rand_gauss(column):
    mean_val = column.mean(skipna=True)
    std_val = column.std(skipna=True)
    return column.apply(lambda x: gauss(mean_val, std_val) if isnull(x) else x)

