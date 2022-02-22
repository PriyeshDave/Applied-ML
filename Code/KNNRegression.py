## Build the KNN Regression algorithm in python. 
## The current algorithm assumes the dataset with two features and a target variable. 

from cv2 import mean
import pandas as pd;
import numpy as np;
from collections import Counter
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn.metrics import mean_absolute_error, r2_score;

class KNNRegressor:

  def __init__(self, data, k):
      self.X = np.array(data.iloc[:,:-1])
      self.y = np.array(data.iloc[:,-1])
      self.m, self.n = data.shape
      self.k = k

  def euclidianDistance(self, row1, row2):
    distance = np.sum(np.sqrt(np.square(row1 - row2)))
    return distance

  def fit(self):
    y_predicited = [self.predict(row) for row in self.X]
    return y_predicited

  def predict(self, row):
    distances_from_each_row = [self.euclidianDistance(row, fromRow) for fromRow in self.X]

    #sorted_indexes is the index positions of top k distance values. These index are with respect to the original indexes.
    sorted_indexes = np.argsort(distances_from_each_row)

    # Getting the actual y values at sorted_indexes
    top_predicted = self.y[sorted_indexes][:self.k]

    # Taking the mean top k nearest y values.
    y_predicted = sum(top_predicted)/len(top_predicted)
    return y_predicted


if __name__ == '__main__':
  data = pd.read_csv('../DummyData/KNNClassifierData.csv')

  sns.scatterplot(x = data['X1'], y = data['X2'], hue = data['y'])
  plt.show()

  knn = KNNRegressor(data, 3)
  predictions = knn.fit()
  print('Actual Values: ', knn.y)
  print()
  print('Predictions Values: ', predictions)
  print()
  print('R2 Score: ',r2_score(knn.y, predictions))
  print('MAE: ',mean_absolute_error(knn.y, predictions))
  
