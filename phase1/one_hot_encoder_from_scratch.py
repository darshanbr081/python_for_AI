"""**One-Hot Encoder from Scratch:** Implement a one-hot encoder using only NumPy."""

import numpy as np

class OneHotEncoder:
    def fit(self,X):
        self.categories = np.unique(X)

    def transform(self,X):
        one_hot_encoded = np.zeros((X.shape[0], len(self.categories)))
        for i, category in enumerate(self.categories):
            one_hot_encoded[:, i] = (X == category).astype(int)
        return one_hot_encoded
#Example usage:
if __name__ == "__main__":
    data = np.array(['cat','dog','cat','mouse'])
    encoder = OneHotEncoder()
    encoder.fit(data)
    one_hot_encoded_data = encoder.transform(data)
    print(one_hot_encoded_data)

