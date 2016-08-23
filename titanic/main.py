import pandas
import sklearn
import math

#Load the data:
titanic_train = pandas.read_csv("train.csv")

def col_contain_nan(col):
    for val in col.unique():
        if isinstance(val, float) and math.isnan(val):
            return True
    return False

def columns_with_nan(df):
    contain_nan = []
    columns = df.columns
    for col in columns:
        if col_contain_nan(df[col]):
            contain_nan.append(col)
    return contain_nan
