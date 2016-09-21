import unittest
import titanic_tutorial
import titanic_tutorial_answers
import pandas
import numpy as np
from pandas.util.testing import assert_frame_equal
from titanic_preprocess import preprocess

#Load the data:
titanic_train_og = pandas.read_csv("train.csv")
titanic_train = preprocess(titanic_train_og)

def is_df_equal(df1, df2):
    try:
        assert_frame_equal(df1, df2)
        return True
    except:  # appeantly AssertionError doesn't catch all
        return False

def assert_array_equal(array1, array2):
    if array1.shape != array2.shape:
        return False
    return np.allclose(array1, array2)

class TitanicWorkshop(unittest.TestCase):

    def test_add_women_in_family(self):
        actual = titanic_tutorial.add_women_in_family(titanic_train)
        expected = titanic_tutorial_answers.add_women_in_family(titanic_train)
        self.assertTrue(is_df_equal(actual, expected))

    def test_random_forest_ensemble(self):
        actual = titanic_tutorial.random_forest_ensemble(titanic_train)
        expected = titanic_tutorial_answers.random_forest_ensemble(titanic_train)
        self.assertTrue(assert_array_equal(actual, expected))

    def test_majority_voting(self):
        predictions = titanic_tutorial_answers.random_forest_ensemble(titanic_train)
        actual = titanic_tutorial.majority_voting(predictions)
        expected = titanic_tutorial_answers.majority_voting(predictions)
        self.assertTrue(assert_array_equal(actual, expected))

    def test_support_vector_machine(self):
        predictors = ["Pclass", "Sex", "Age", "Fare", "FamilySize"]
        actual = titanic_tutorial.support_vector_machine(titanic_train, predictors)
        expected = titanic_tutorial_answers.support_vector_machine(titanic_train, predictors)
        self.assertTrue(assert_array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
