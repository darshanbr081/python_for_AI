"""  **Missing Data Imputer:** Build a custom transformer that fills missing values with the median of their specific class/group."""

import numpy as np
import pandas as pd

class MissingDataImputer:
    def __init__(self, group_col, target_col):
        self.group_col = group_col
        self.target_col = target_col
        self.medians = {}

    def fit(self, X):
        # Calculate the median for each group
        self.medians = X.groupby(self.group_col)[self.target_col].median().to_dict() #here X is the dataframe passed to fit method

    def transform(self,X):
        #Fill missing values with the corresponding group median
        X[self.target_col] = X.apply(
            lambda row:self.medians[row[self.group_col]] 
            if pd.isnull(row[self.target_col]) 
            else row[self.target_col],
            axis=1
            )
        return X
    
#example usage:
if __name__ == "__main__":
    data = {'group':['A','A','B','B','C','C'],
            'value':[1, np.nan,3,4,5,np.nan]}
    df = pd.DataFrame(data)
    imputer = MissingDataImputer(group_col='group', target_col='value')
    imputer.fit(df)
    transformed_df = imputer.transform(df)
    print(transformed_df)