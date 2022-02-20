import pandas as pd;
import numpy as np;


class LinearRegressor:

  def __init__(self, Data):
    m, n = Data.shape
    self.X = Data.iloc[:,0]
    self.W0 = np.ones((m, ))
    self.X = np.vstack((self.W0, self.X)).T

    self.y = np.array(Data.iloc[:,1]).reshape((m,1))
  

  def fit(self):
    self.W = (np.linalg.pinv(self.X.T @ self.X)) @ self.X.T @ self.y
    return self.W

  def predict(self, X):
    predictions = np.dot(X, self.W)
    return predictions

if __name__ == '__main__':
  data = pd.read_csv('../DummyData/salary_data.csv')
  lr = LinearRegressor(data)
  lr.fit()
  predictions = lr.predict([[1,10.5]])
  print('Predictions: ', predictions)


