import pandas as pd;
import numpy as np;


class LinearRegressor:

  def __init__(self, Data, regParam):
    self.m, self.n = Data.shape
    self.X =np.array(Data.iloc[:,0]).reshape(self.m, self.n -1)
    self.W0 = np.ones((self.m, 1))
    self.X = np.concatenate((self.W0, self.X), 1)
    self.y = np.array(Data.iloc[:,1]).reshape((self.m,1))
    self.regParam = regParam

  
  def fit(self):
    # We have Normal equation as J(theta) = pinv((X.T @ X) + (regParam * Matrix)) @ X.T @ y
    diagonalValues = np.hstack((np.zeros(1), np.ones(self.n-1)))
    mat = np.zeros((self.n, self.n))
    np.fill_diagonal(mat, diagonalValues)
    self.W = np.linalg.pinv((self.X.T @ self.X) + (self.regParam * mat)) @ self.X.T @ self.y
    return self.W

  def predict(self, X):
    predictions = np.dot(X, self.W)
    return predictions

if __name__ == '__main__':
  data = pd.read_csv('../DummyData/salary_data.csv')
  lr = LinearRegressor(data, .05)
  lr.fit()
  predictions = lr.predict([[1,10.5]])
  print('Predictions: ', predictions)


