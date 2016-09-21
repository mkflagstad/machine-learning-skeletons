from __future__ import print_function
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import pandas
import sklearn
import operator
import numpy as np
from scipy import stats


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
    #Goal: get the predictions for likelihood of survival using GradientBoostingClassifier, LogisticRegression, and RandomForest

    #Add the random forest model to algorithms we want to ensemble
    #   - Use the same predictors as the GradientBoostingClassifier
    #   - Give the algorithm the following parameters:
    #   - random_state=1, n_estimators=25, max_depth=3
    algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]
    #The result of the ensemble is a n x m numpy array where n is the number of algorithms we are ensembling
    # and m is the number of data points. Each row is the predictions for a given algorithm. Check out the code below!
    return ensemble(df, algorithms)

def majority_voting(predictions):
    #Goal: given an n x m numpy array (where n is the number of algorithms we are ensembling and m is the number of data points),
    # where each row is the predictions for one algorithm and each column is the three predictions for a single data point,
    # classify each datapoint as survived (1) or did not survive (0) using majority voting.
    # This is an alternative to ensembling the algortihms by averaging the predictions.
    #Your result should be a 1 x m array where m is the number of data points.
    return np.array([])

def support_vector_machine(df, predictors):
    #Goal: Use a support vector machine to find likelihood of surival. We are going to use Support Vector Classification(SVC)
    # We are again going to use Kfold for cross validation.
    kf = KFold(df.shape[0], n_folds=3, random_state=1)
    for train, test in kf:
        alg = SVC(probability=True, random_state=1)
    return np.array([])

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

def get_accuracy(df, predictions):
    return sum(predictions[predictions == df["Survived"]]) / len(predictions)
