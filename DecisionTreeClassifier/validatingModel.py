import numpy as np
from sklearn.tree import DecisionTreeRegressor

def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  
  targets = data['redshift']
  return (features, targets)


def median_diff(predicted, actual):
  med_diff = np.median(np.absolute(predicted - actual))
  return med_diff


def validate_model(model, features, targets):
  split = features.shape[0]//2
  train_features = features[:split]
  test_features = features[split:]
  train_targets = targets[:split]
  test_targets = targets[split:]
  
  model.fit(train_features, train_targets)
  predictions = model.predict(test_features)
  
  # calculate the accuracy
  return median_diff(test_targets, predictions)


if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  dtr = DecisionTreeRegressor()

  # validate the model
  diff = validate_model(dtr, features, targets)
  print('Median difference: {:f}'.format(diff))
