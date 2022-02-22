## Build the KNN Classification algorithm in python. 
## The current algorithm assumes the dataset with two features and a target variable. 

import pandas as pd;
import numpy as np;
from collections import Counter
import seaborn as sns;
import matplotlib.pyplot as plt;

class KNNClassifier:

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
    sorted_indexes = np.argsort(distances_from_each_row)
    top_predicted = self.y[sorted_indexes][:self.k]

    #Now seeing the maximum occurence of values
    predicted = Counter(top_predicted).most_common()[0][0]
    return predicted


if __name__ == '__main__':
  data = pd.read_csv('../DummyData/KNNClassifierData.csv')

  sns.scatterplot(x = data['X1'], y = data['X2'], hue = data['y'])
  plt.show()

  knn = KNNClassifier(data, 3)
  predictions = knn.fit()
  print('Actual Values: ', knn.y)
  print('Predictions Values: ', predictions)
  accuracy = 1 - np.mean(predictions != knn.y)
  print('Accuracy: ', accuracy)
