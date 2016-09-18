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



class TitanicWorkshop(unittest.TestCase):

    def assert_array_equal(self, actual, expected):
        equal = np.array_equal(expected, actual)
        if not equal:
            print("Expected: ")
            print(expected)
            print("Actual: ")
            print(actual)
        self.assertTrue(equal)

    def test_add_women_in_family(self):
        your_soln = titanic_tutorial.add_women_in_family(titanic_train)
        expected = titanic_tutorial_answers.add_women_in_family(titanic_train)
        self.assertTrue(is_df_equal(your_soln, expected))

    def test_random_forest_ensemble(self):
        your_soln = titanic_tutorial.random_forest_ensemble(titanic_train)
        expected = titanic_tutorial_answers.random_forest_ensemble(titanic_train)
        self.assert_array_equal(your_soln, expected)

    #Try the random forest classifier in the ensemble.

    #A support vector machine might work well with this data.

    #We could try neural networks.

    #Boosting with a different base classifier might work better.

    #Could majority voting be a better ensembling method than averaging probabilities?

if __name__ == '__main__':
    unittest.main()
