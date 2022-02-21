import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt
import seaborn as sns

class RidgeRegression:

  def __init__(self,data, learningRate, noOfIterations, regParam):
    
    self.alpha = learningRate
    self.noOfIterations = noOfIterations
    self.regParam = regParam
    self.m, self.n = data.shape

    self.W = np.zeros((self.n, 1))
    self.W0 = np.ones((self.m, 1))

    self.X = np.array(data.iloc[:,0]).reshape((self.m, self.n -1))
    # Adding W0 as ones and making X of size (m,2)
    self.X = np.concatenate((self.W0, self.X), 1)
    self.y = np.array(data.iloc[:,1]).reshape((self.m), 1)

    self.costList = list()
    self.iterationList = list()
    self.WList = list()

  def updateWeights(self, dw):
    self.W = self.W - dw

  def fit(self):
    for i in range(self.noOfIterations):
      Y_predicted = self.predict()
      
      # Calculating cost on each iteration
      cost = (np.sum(np.square(Y_predicted - self.y))/(2 * self.m))  + ((self.regParam/ (2 * self.m)) * np.dot(self.W.T, self.W))
      
      #appending cost, iteration, W in respective lists.
      self.costList.append(cost)
      self.iterationList.append(i)
      self.WList.append(self.W)

      # Implementing Gradient Descent.
      dw = ((self.alpha/self.m) * np.dot(self.X.T, Y_predicted - self.y)) + ((self.regParam/self.m) * (np.sum(self.W)))
      self.updateWeights(dw)
    
    minCost = min(self.costList[1:])
    minIndex = self.costList.index(minCost)
    self.W = self.WList[minIndex]
    
    costs = [x[0][0] for x in list(self.costList)]
    
    plt.figure(figsize=(10,5))
    sns.lineplot(x = self.iterationList, y = costs)
    plt.show()
  
  def predict(self):
    # Making prediction here 
    y_predicted = np.dot(self.X, self.W)
    return y_predicted

  def prediction(self, X, W):
    return np.dot(X, W)
    

if __name__ == '__main__':
  data = pd.read_csv('../DummyData/salary_data.csv')
  rr = RidgeRegression(data,.051, 300, 0.05)
  rr.fit()
  predictions = rr.predict()
  print(predictions)
