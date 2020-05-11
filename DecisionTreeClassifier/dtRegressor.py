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



data = np.load('sdss_galaxy_colors.npy')
features, targets = get_features_targets(data)
  
dtr = DecisionTreeRegressor()
# train the model
dtr.fit(features, targets)
# make predictions using the same features
predictions = dtr.predict(features)

print(predictions[:4])
