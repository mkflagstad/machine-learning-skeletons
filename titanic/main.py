import pandas
import sklearn
import math

def col_contain_nan(col):
    for val in col.unique():
        if isinstance(val, float) and math.isnan(val):
            return True
    return False

def col_contain_non_numeric(col):
    for val in col.unique():
        if (isinstance(val, str)):
            val = unicode(val, 'utf-8')
        if ((isinstance(val, unicode) and not val.isnumeric()) or
           (not (isinstance(val, float) or isinstance(val, long) or isinstance(val, int)))):
               return True
    return False

def filter_columns(df, condition):
    accum = []
    columns = df.columns
    for col in columns:
        if condition(df[col]):
            accum.append(col)
    return accum

def columns_with_nan(df):
    return filter_columns(df, col_contain_nan)

def columns_with_non_numeric(df):
    return filter_columns(df, col_contain_non_numeric)

def column_type(col):
    ty = None
    for val in col.unique():
        is_first_case_of_non_nan_value = ty is None and non_nan(val)
        if is_first_case_of_non_nan_value:
            ty = type(val)
        is_different_type = non_nan(val) and type(val) is not ty
        if is_different_type:
            return None
    return ty

def non_nan(x):
    return not (isinstance(x, float) and math.isnan(x))

def types_of_columns_with_nan(df):
    types = []
    for col in columns_with_nan(df):
        types.append((col, column_type(df[col])))
    return types

def num_missing_values(col):
    nan_count = 0
    for val in col:
        if not non_nan(val):
            nan_count = nan_count + 1
    return nan_count
