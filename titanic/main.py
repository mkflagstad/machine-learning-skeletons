import pandas
import sklearn
import math

#Load the data:
titanic_train = pandas.read_csv("train.csv")

def to_clean(df):
    # Find out the columns with missing values before calling `to_clean`
    # print(columns_with_nan(df))
    copy = df.copy(deep=True)
    copy = copy.drop('Cabin', axis=1)
    copy['Age'] = copy['Age'].fillna(copy['Age'].mean())
    copy['Embarked'] = copy['Embarked'].fillna(copy['Embarked'].mode()[0])
    return copy

def test_cleaning:():
    print "og: ", columns_with_nan(titanic_train)
    new = to_clean(titanic_train)
    print "after clean: ", columns_with_nan(new)


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
