import pandas as pd;
import numpy as np
from pyrsistent import m;
import sklearn.datasets;
import matplotlib.pyplot as plt;

class LogisticRegressor:

  def __init__(self, data, noOfIterations, learningRate):
    self.m, self.n = data.shape

    self.X = np.array(data.iloc[:,0]).reshape(self.m, self.n -1)
    self.W0 = np.ones((self.m, 1))
  
    self.noOfIterations = noOfIterations
    self.learningRate = learningRate
    self.costList = list()
    self.WList = list()
    self.iterationsList = list()

    self.X = np.concatenate((self.W0, self.X), 1)
    self.y = np.array(data.iloc[:,1]).reshape((self.m,1))
    self.W = np.zeros((self.n, 1)) 

    
  def updateWeights(self, dw):
    self.W = self.W - self.learningRate * dw


  def fit(self):
    for i in range(self.noOfIterations):
      y_predicted = self.predict()
      # Calculating the cost on each iteration.
      cost = np.sum(-(self.y * np.log(y_predicted)) - ((1- self.y) * (np.log(1 - y_predicted))))/self.m

      #Calculating derivative of cost
      dw = (np.dot(self.X.T, (y_predicted - self.y)))/self.m

      #Appending the values to respective lists
      self.costList.append(cost)
      self.iterationsList.append(i)
      self.WList.append(self.W)

      #Updating the weights
      self.updateWeights(dw)
      
    minCost = min(self.costList[1:])
    minIndex = self.costList.index(minCost)
    self.W = self.WList[minIndex]


  def predict(self):
    hTheta = np.matmul(self.X, self.W)
    y_predicted = 1/(1 + np.exp(-hTheta))
    return y_predicted
    

if __name__ == '__main__':
  #dataset = sklearn.datasets.load_breast_cancer()
  dataset = pd.read_csv('../DummyData/data.csv')
  #plt.scatter(dataset.iloc[:,0], dataset.iloc[:,1])
  #plt.show()
  lr = LogisticRegressor(dataset, 500, .000000005)
  lr.fit()
  #predictions = [round(x,1) for x in lr.predict().flatten()]
  predictions = lr.predict().flatten();
  print(predictions)
  predictions = [1 if x >= .5 else 0 for x in predictions ]
  print(predictions)
  
  
  
