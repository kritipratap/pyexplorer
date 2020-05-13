import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from support_functions import generate_features_targets, plot_confusion_matrix, calculate_accuracy

def rf_predict_actual(data, n_estimators):
  features, targets = generate_features_targets(data)

  rfc = RandomForestClassifier(n_estimators = n_estimators)
  predict = cross_val_predict(rfc, features, targets, cv = 10)

  return predict, data['class']


if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')
  
  # Number of trees
  number_estimators = 50              
  predicted, actual = rf_predict_actual(data, number_estimators)

  accuracy = calculate_accuracy(predicted, actual)
  print("Accuracy score:", accuracy)

  class_labels = list(set(actual))
  model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()
