from __future__ import print_function
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import pandas
import sklearn
import main
import operator
import numpy as np


def add_women_in_family(df):
    #Goal: Return the dataframe with a new column "WomenInFamily" representing the number of women in the family.

    #Create a copy of the data frame to avoid side effects:
    copy = df.copy(deep=True)

    #Use the columns "Sex" and "FamilyId". Remember that women are represented by a 1
    #in the sex column. For the -1 family id, make the num women 0.

    #You can iterate over the index and rows in a dataframe using:
    # for index, row in df.iterrows

    #Create a new column by apply a function to our dataframe:
    #num_women_col = df.apply(lambda row: my_cool_function(row))

    #Add the column to the dataframe:
    #df["WomenInFamily"] = num_women_col

    return copy

def random_forest_ensemble(df):
    # #Add the random forest model to algorithms we want to ensemble
    # #Use the same predictors as the GradientBoostingClassifier
    # algorithms = [
    # [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    # [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]
    # #The result of the ensemble is a n x m numpy array where n is the number of algorithms we are ensembling
    # # and m is the number of data points. Each row is the predictions for a given algorithm.
    # return ensemble(df, algorithms)
    algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]],
    [RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]]]
    result = ensemble(df, algorithms)
    print("result size: ", result.shape)
    return result


def ensemble(df, algorithms_with_predictors):
    kf = KFold(df.shape[0], n_folds=3, random_state=1)

    # algorithms x data_points (2 x 891)
    predictions = np.asarray([np.asarray([]) for i in range(len(algorithms_with_predictors))])
    for train, test in kf:
        train_target = df["Survived"].iloc[train]

        # algorithms x data_points_per_fold (2 x 297)
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
