import pandas
import sklearn
import main
import operator

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
