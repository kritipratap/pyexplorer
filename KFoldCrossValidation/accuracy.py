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


def split_galaxies_qsos(data):
  galaxies = data[data['spec_class'] == b'GALAXY']
  qsos = data[data['spec_class'] == b'QSO']
  return (galaxies, qsos)


def cross_validate_median_diff(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return np.mean(cross_validate_model(dtr, features, targets, 10))
  

if __name__ == "__main__":
    data = np.load('./sdss_galaxy_colors.npy')
    galaxies, qsos= split_galaxies_qsos(data)

    galaxy_med_diff = cross_validate_median_diff(galaxies)
    qso_med_diff = cross_validate_median_diff(qsos)

    print("Median difference for Galaxies: {:.3f}".format(galaxy_med_diff))
    print("Median difference for QSOs: {:.3f}".format(qso_med_diff))
