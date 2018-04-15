from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np

def _get_mask(X, value_to_mask):
        if value_to_mask == "NaN":
            # print(np.isnan(X))
            return np.isnan(X)
        else:
            return X == value_to_mask

class MiceImputer(object):

    def __init__(self, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.axis = axis
        self.verbose = verbose
        self.copy = copy
        self.imp = Imputer(missing_values=self.missing_values, strategy=self.strategy, axis= self.axis, verbose=self.verbose, copy=self.copy)
        self.mask2 = np.zeros([1])

    def _seed_values(self, X):
        self.imp.fit(X)
        return self.imp.transform(X)


    def _process(self, X, column, model_class):
        # Remove values that are in mask
        # mask = np.array(_get_mask(X,self.missing_values)[:, column].T)
        mask = _get_mask(X,self.missing_values)
        mask_indices = np.where(mask[:,column]==True)
        
        X_temp = np.array(X)
        index = np.where(mask==True)
        X_temp[index[0],index[1]] = 0
        
        #print(mask.shape,mask_indices)
        X_data = np.delete(X_temp, mask_indices, 0)
        # print(X_data)

        # Instantiate the model
        model = model_class()

        # Slice out the column to predict and delete the column.
        y_data = X_data[:, column]
        X_data = np.delete(X_data, column, 1)

        # Split training and test data
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Score the model
        scores = model.score(X_test, y_test)

        # Predict missing vars
        X_predict = np.delete(X_temp, column, 1)
        y = model.predict(X_predict)

        # Replace values in X with their predictions
        predict_indices = np.array(np.where(self.mask2[:,column]==True))
        X[predict_indices,column] = y[predict_indices]

        return X

    def transform(self, X, model_class=LinearRegression, iterations=10):
        # X = np.matrix(X)
        print('len',len(X.T))
        self.mask2 = _get_mask(X, self.missing_values)
        seeded = self._seed_values(X)
        

        for i in range(iterations):
            for c in range(len(X.T)):
                X = self._process(X, c, model_class)

        # Return X matrix with imputed values
        return X