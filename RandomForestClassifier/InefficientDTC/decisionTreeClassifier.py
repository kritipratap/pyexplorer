import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math


def splitdata_train_test(data, fraction_training):
  np.random.seed(0)
  np.random.shuffle(data)
  
  tr_rows = math.floor(len(data) * fraction_training)
  
  training_set = data[0:tr_rows]
  testing_set = data[tr_rows:len(data)]
  
  return (training_set, testing_set)


def generate_features_targets(data):
  targets = data['class']

  features = np.empty(shape=(len(data), 13))
  features[:, 0] = data['u-g']
  features[:, 1] = data['g-r']
  features[:, 2] = data['r-i']
  features[:, 3] = data['i-z']
  features[:, 4] = data['ecc']
  features[:, 5] = data['m4_u']
  features[:, 6] = data['m4_g']
  features[:, 7] = data['m4_r']
  features[:, 8] = data['m4_i']
  features[:, 9] = data['m4_z']

  features[:, 10] = data['petroR50_u'] / data['petroR90_u']
  features[:, 11] = data['petroR50_r'] / data['petroR90_r']
  features[:, 12] = data['petroR50_z'] / data['petroR90_z']

  return features, targets


def dtc_predict_actual(data):
  training_set, testing_set = splitdata_train_test(data, 0.7)

  features_training, targets_training = generate_features_targets(training_set)
  features_testing, targets_testing = generate_features_targets(testing_set)

  dtc = DecisionTreeClassifier()
  dtc.fit(features_training, targets_training)
  prediction = dtc.predict(features_testing)

  return prediction, targets_testing


if __name__ == '__main__':
  data = np.load('galaxy_catalogue.npy')
    
  predicted_class, actual_class = dtc_predict_actual(data)

  print("Some initial results...\n   predicted,  actual")
  for i in range(10):
    print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))
