import pandas as pd;
import numpy as np;

class myLinearRegressor:

  def __init__(self,data, learningRate, noOfIterations):
    
    self.alpha = learningRate
    self.noOfIterations = noOfIterations
    self.m = data.shape[0]
    self.n = data.shape[1]
    self.W = np.zeros((self.n,1))
    self.W0 = np.ones((self.m, ))

    self.X = np.array(data.iloc[:,0])
    # Adding W0 as ones and making X of size (m,2)
    self.X = np.vstack((self.W0, self.X)).T

    self.y = np.array(data.iloc[:,1]).reshape((self.m),1)

    self.costList = list()
    self.iterationList = list()
    self.WList = list()

  def updateWeights(self, dw):
    self.W = self.W - self.alpha * dw

  def fit(self):
    for i in range(self.noOfIterations):
      Y_predicted = self.predict()
      
      # Calculating cost on each iteration
      cost = (np.sum(np.square(Y_predicted - self.y)))/(2 * self.m)
      
      #appending cost, iteration, W in respective lists.
      self.costList.append(cost)
      self.iterationList.append(i)
      self.WList.append(self.W)

      # Implementing Gradient Descent.
      # This will give an array of shape(n,1) which is same as the shape of our theta's.
      dw = (np.dot(self.X.T, Y_predicted - self.y))/self.m
      self.updateWeights(dw)
    
    minCost = min(self.costList[1:])
    minIndex = self.costList.index(minCost)
    self.W = self.WList[minIndex]
    
  
  def predict(self):
    # Making prediction here 
    y_predicted = np.dot(self.X, self.W)
    return y_predicted

  def prediction(self, X, W):
    return np.dot(X, W)
    

if __name__ == '__main__':
  data = pd.read_csv('../DummyData/salary_data.csv')
  lr = myLinearRegressor(data,.065, 100)
  lr.fit()
  predictions = lr.predict()
  print(predictions)
