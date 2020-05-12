import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def get_features_targets(data):
  # complete this function
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


def accuracy_by_treedepth(features, targets, depths):
  split = features.shape[0]//2
  train_features = features[:split]
  test_features = features[split:]
  train_targets = targets[:split]
  test_targets = targets[split:]
  
  mediandiff_training = []
  mediandiff_test = []
  
  for depth in depths:
    dtr = DecisionTreeRegressor(max_depth=depth)
    dtr.fit(train_features, train_targets)
    
    predictions = dtr.predict(train_features)
    accuracy = median_diff(train_targets, predictions)
    mediandiff_training.append(accuracy)
    
    predictions = dtr.predict(test_features)
    accuracy = median_diff(test_targets, predictions)
    mediandiff_test.append(accuracy)   
    
  return (mediandiff_training, mediandiff_test)


if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  tree_depths = [i for i in range(1, 36, 2)]

  train_med_diffs, test_med_diffs = accuracy_by_treedepth(features, targets, tree_depths)
  print("Depth with lowest median difference : {}".format(tree_depths[test_med_diffs.index(min(test_med_diffs))]))
    
  train_plot = plt.plot(tree_depths, train_med_diffs, label='Training set')
  test_plot = plt.plot(tree_depths, test_med_diffs, label='Validation set')
  plt.xlabel("Maximum Tree Depth")
  plt.ylabel("Median of Differences")
  plt.legend()
  plt.show()
