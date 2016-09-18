from __future__ import print_function
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import pandas
import sklearn
import main
import numpy as np



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

def random_forest_ensemble(df):
    algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]

    # Initialize the cross validation folds
    kf = KFold(df.shape[0], n_folds=3, random_state=1)

    predictions = np.asarray([np.asarray([]) for i in range(len(algorithms))])
    for train, test in kf:
        train_target = df["Survived"].iloc[train]
        full_test_predictions = np.asarray([])
        # Make predictions for each algorithm on each fold
        for alg, predictors in algorithms:
            # Fit the algorithm on the training data.
            alg.fit(df[predictors].iloc[train,:], train_target)
            # Select and predict on the test fold.
            # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
            test_predictions = alg.predict_proba(df[predictors].iloc[test,:].astype(float))[:,1]
            full_test_predictions = np.concatenate((full_test_predictions, test_predictions), axis=0)
        print("full_test_predictions size: ", full_test_predictions.T.shape)
        print("predictions size: ", predictions.shape)
        np.concatenate((predictions, full_test_predictions.T), axis=1)
    print("predictions: ", predictions)
