import numpy as np

def mean_datasets(files):
  count = 0
  for file in files:
    newdata = np.loadtxt(file, delimiter=',')
    if count == 0:
      data = newdata
    else:
     data = data + newdata
    count = count + 1
 
  data = data/count
  data = np.round(data,1)
  
  return data


if __name__ == '__main__':
 
  print(mean_datasets(['data1.csv', 'data2.csv', 'data3.csv']))
  print(mean_datasets(['data4.csv', 'data5.csv', 'data6.csv']))
