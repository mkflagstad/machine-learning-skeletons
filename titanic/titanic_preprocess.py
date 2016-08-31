import pandas
import sklearn
import main
import operator
import re

def to_clean(df):
    # Find out the columns with missing values before calling `to_clean`
    # print(columns_with_nan(df))
    copy = df.copy(deep=True)
    copy = copy.drop('Cabin', axis=1)
    #copy = copy.drop('Name', axis=1)
    copy = copy.drop('Ticket', axis=1)
    copy['Age'] = copy['Age'].fillna(copy['Age'].mean())
    copy['Embarked'] = copy['Embarked'].fillna(copy['Embarked'].mode()[0])
    copy.loc[copy['Sex'] == 'male', 'Sex'] = 0
    copy.loc[copy['Sex'] == 'female', 'Sex'] = 1
    copy.loc[copy['Embarked'] == 'S', 'Embarked'] = 0
    copy.loc[copy['Embarked'] == 'C', 'Embarked'] = 1
    copy.loc[copy['Embarked'] == 'Q', 'Embarked'] = 2
    return copy

def test_cleaning():
    print " -- before clean -- "
    print "columns with nan: ", columns_with_nan(titanic_train)
    print "columns with non numerics: ", columns_with_non_numeric(titanic_train)
    new = to_clean(titanic_train)
    print " -- after clean --  "
    print "columns with nan: ", columns_with_nan(new)
    print "columns with non numerics: ", columns_with_non_numeric(new)

def add_titles(titanic_df):
    # First, we'll add titles to the test set.
    titles = titanic_df["Name"].apply(get_title)
    # We're adding the Dona title to the mapping, because it's in the test set, but not the training set
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    titanic_df["Title"] = titles
    return titanic_df


family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

def add_family_size_and_familyid(titanic_df):
    titanic_df["FamilySize"] = titanic_df["SibSp"] + titanic_df["Parch"]
    family_ids = titanic_df.apply(get_family_id, axis=1)
    family_ids[titanic_df["FamilySize"] < 3] = -1
    titanic_df["FamilyId"] = family_ids
    return titanic_df

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def preprocess(df):
    #Remove features irrelevant to survival, fill in
    #missing values and transform non numerics
    #These steps are from the first titanic tutorial found here:
    #
    copy = df.copy(deep=True)
    copy = to_clean(copy)

    #These steps are from the previous tutorial found here:
    # https://www.dataquest.io/mission/75/improving-your-submission
    copy = add_titles(copy)
    copy["NameLength"] = copy["Name"].apply(lambda x: len(x))
    copy = add_family_size_and_familyid(copy)
    return copy
