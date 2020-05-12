import numpy as np
from sklearn.model_selection import KFold
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


def cross_validate_model(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)
  mediandiffs = []
  
  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    model.fit(train_features, train_targets)
    predictions = model.predict(test_features)
    mediandiffs.append(median_diff(test_targets, predictions))
    
  return mediandiffs


if __name__ == "__main__":
  data = np.load('./sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  dtr = DecisionTreeRegressor(max_depth=19)
  diffs = cross_validate_model(dtr, features, targets, 10)

  print('Differences: {}'.format(', '.join(['{:.3f}'.format(val) for val in diffs])))
  print('Mean difference: {:.3f}'.format(np.mean(diffs)))
