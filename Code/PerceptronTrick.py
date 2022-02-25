import pandas as pd;
import numpy as np;
import sklearn.datasets as ds;
import seaborn as sns;
import matplotlib_inline
import matplotlib.pyplot as plt;


class Perceptron:

  def __init__(self, X, target, epochs, learning_rate):
    self.m, self.n = X.shape
    self.X = X
    self.y = np.array(target['Target']).reshape((self.m, 1))
    self.W0 = np.ones((self.m,1))
    self.W = np.zeros((self.n + 1,1))
    self.epochs = epochs
    self.learning_rate = learning_rate

    #print('X: ', self.X.shape)
    #print('y: ', self.y.shape)
    #print('W0: ', self.W0.shape)
    #print('W: ', self.W.shape)

    self.X = np.concatenate((self.W0, self.X), axis=1)
    

  def fit(self):
    self.get_optimum_weights()

  def get_optimum_weights(self):
    for i in range(self.epochs):
      for row, actual_value in zip(self.X, self.y):
        row = row.reshape((3,1))
        predicted = np.dot(row.T, self.W)
        predicted = 1/(1 + np.exp(-predicted))

        if predicted >= .5:
          predicted = 1
        else :
          predicted = 0

        if (predicted != actual_value):
          self.W = self.W + self.learning_rate * (actual_value - predicted) * row


  def predict(self, X, Y):
    predictions = np.dot(X, self.W)
    predictions = [1 if 1/(1 + np.exp(-x)) >= .5 else 0 for x in predictions]
    return predictions

if __name__ == '__main__':
  X, y = ds.make_classification(n_samples=1000, n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class =1, random_state=1)
  data = pd.DataFrame(X, columns=['X1', 'X2'])
  target = pd.DataFrame(y, columns=['Target'])
  #plt.figure(figsize = (12,5))
  #sns.scatterplot(x = 'X1',y = 'X2', data= data, hue=y)
  #plt.show()
  p = Perceptron(data, target, epochs = 100, learning_rate = .1)
  p.fit()
  predictions = p.predict(p.X, p.y)
  #print('Actual Values: ', y)
  #print('Predicted Values: ',list(predictions))
  print('Accuracy: ', 1 - np.mean(predictions != y))

