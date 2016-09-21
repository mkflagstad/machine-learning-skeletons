from __future__ import print_function
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import pandas
import sklearn
import main
import numpy as np
from scipy import stats

axis_row = 0
axis_col = 1

num_women_by_family = {}

def classifier(survival_percent):
    return 1 if survival_percent >= 0.5 else 0

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
    copy = df.copy(deep=True)
    algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]],
    [RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]]]
    return ensemble(copy, algorithms)

def ensemble(df, algorithms_with_predictors):
    kf = KFold(df.shape[0], n_folds=3, random_state=1)

    # algorithms x data_points
    predictions = np.asarray([np.asarray([]) for i in range(len(algorithms_with_predictors))])
    for train, test in kf:
        train_target = df["Survived"].iloc[train]

        # algorithms x data_points_per_fold
        full_test_predictions = []

        # Make predictions for each algorithm on each fold
        for alg, predictors in algorithms_with_predictors:
            # Fit the algorithm on the training data.
            alg.fit(df[predictors].iloc[train,:], train_target)
            # Select and predict on the test fold.
            # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
            test_predictions = alg.predict_proba(df[predictors].iloc[test,:].astype(float))[:,1]
            full_test_predictions.append(test_predictions)
        full_test_predictions = np.asarray(full_test_predictions)
        predictions = np.concatenate((predictions, full_test_predictions), axis=1)
    return predictions

def majority_voting(predictions):
    classify_cell = np.vectorize(classifier)
    votes = classify_cell(predictions)
    return stats.mode(votes)[0][0]

def support_vector_machine(df, predictors):
    kf = KFold(df.shape[0], n_folds=3, random_state=1)
    predictions = []
    for train, test in kf:
        alg = SVC(probability=True, random_state=1)
        train_target = df["Survived"].iloc[train]
        alg.fit(df[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(df[predictors].iloc[test,:].astype(float))[:,1]
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    return predictions

