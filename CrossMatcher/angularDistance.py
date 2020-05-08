import numpy as np

def angular_dist(a1, d1, a2, d2):
  a1 = np.radians(a1)
  d1 = np.radians(d1)
  a2 = np.radians(a2)
  d2 = np.radians(d2)
  
  p1 = np.sin(abs(d1-d2)/2)**2
  p2 = np.cos(d1)*np.cos(d2)*np.sin(abs(a1-a2)/2)**2
  p3 = 2*np.arcsin(np.sqrt(p1+p2))
  return np.degrees(p3)


if __name__ == '__main__':
  
  print(angular_dist(21.07, 0.1, 21.15, 8.2))
  print(angular_dist(10.3, -3, 24.3, -29))
