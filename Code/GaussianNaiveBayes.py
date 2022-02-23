import pandas as pd;
import numpy as np;
import math


class GaussianNB:

  def __init__(self, data):
      self.X = np.array(data.iloc[:,1:])
      self.y = np.array(data.iloc[:,0])
      self.m, self.n = data.shape
      #print('X: ', self.X)
      #print("Y: ", self.y)

      mean_values = data.groupby('Person').mean()
      std_values =  data.groupby('Person').std()

      female_mean_values = mean_values.xs('female')
      male_mean_values = mean_values.xs('male')
      female_std_values = std_values.xs('female')
      male_std_values = std_values.xs('male')

      self.female_height_mean, self.female_weight_mean, self.female_foot_mean = female_mean_values
      self.female_height_std, self.female_weight_std, self.female_foot_std = female_std_values

      self.male_height_mean, self.male_weight_mean, self.male_foot_mean = male_mean_values
      self.male_height_std, self.male_weight_std, self.male_foot_std = male_std_values


  def fit(self, datasetType):
    y_predicted = [self.predict(row, datasetType) for row in self.X]
    return y_predicted


  def predict(self,row, datasetType):
    return self.calculateProbability(row, datasetType)


  def calculateProbability(self, row, datasetType):

    #Calculating respective conditional probabilities for male
    p_height_male = (1/np.sqrt(2 * math.pi * math.pow(self.male_height_std, 2))) * (np.exp(-math.pow(row[0] - self.male_height_mean, 2)/(2 * math.pow(self.male_height_std,2))))

    p_weight_male = (1/np.sqrt(2 * math.pi * math.pow(self.male_weight_std, 2))) * (np.exp(-math.pow(row[1] - self.male_weight_mean, 2)/(2 * math.pow(self.male_weight_std,2))))

    p_footSize_male = (1/np.sqrt(2 * math.pi * math.pow(self.male_foot_std, 2))) * (np.exp(-math.pow(row[2] - self.male_foot_mean, 2)/(2 * math.pow(self.male_foot_std,2))))

    male_probability = .5 * p_height_male * p_weight_male * p_footSize_male


    #Calculating respective conditional probabilities for female
    p_height_female = (1/np.sqrt(2 * math.pi * math.pow(self.female_height_std, 2))) * (np.exp(-math.pow(row[0] - self.female_height_mean, 2)/(2 * math.pow(self.female_height_std,2))))

    p_weight_female = (1/np.sqrt(2 * math.pi * math.pow(self.female_weight_std, 2))) * (np.exp(-math.pow(row[1] - self.female_weight_mean, 2)/(2 * math.pow(self.female_weight_std,2))))

    p_footSize_female = (1/np.sqrt(2 * math.pi * math.pow(self.female_foot_std, 2))) * (np.exp(-math.pow(row[2] - self.female_foot_mean, 2)/(2 * math.pow(self.male_foot_std,2))))

    female_probability = .5 * p_height_female * p_weight_female * p_footSize_female


    # Normalizing probabilities to one
    male_probability_normalised = male_probability / (male_probability + female_probability)
    female_probability_normalised = 1 - male_probability_normalised

    if(datasetType != 'Train'):
      print('Male predicted probability: ',male_probability_normalised)
      print('Female predicted probability: ',female_probability_normalised)

    if(male_probability_normalised > female_probability_normalised):
       return 'male'
    else:
       return 'female'


if __name__ == '__main__':
  data = pd.read_csv('../DummyData/GaussianNaiveBayes.csv')
  gnb = GaussianNB(data)
  train_predicted = gnb.fit('Train')
  print('Actual Training Values: ',gnb.y)
  print('Predicted Training Values: ',train_predicted)
  print()
  print('Predicting for Test Data:',)
  test_prediction = gnb.predict([6,130,8],'Test')
  print('The person predicted with given values is a',test_prediction)
