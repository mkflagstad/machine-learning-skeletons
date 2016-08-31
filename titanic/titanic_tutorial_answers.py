from __future__ import print_function
import pandas
import sklearn
import main



num_women_by_family = {}

def look_up_num_women(row):
    return num_women_by_family[row['FamilyId']]

def add_women_in_family(df):
    copy = df.copy(deep=True)
    for fam in copy["FamilyId"].unique():
        num_women_by_family[fam] = 0
    for index, row in copy.iterrows():
        if row["Sex"]==1 and row["FamilyId"] != -1:
            famId = row['FamilyId']
            num_women_by_family[famId] = num_women_by_family[famId] + 1
    num_women_col = copy.apply(look_up_num_women, axis=1)
    copy["WomenInFamily"] = num_women_col
    return copy
