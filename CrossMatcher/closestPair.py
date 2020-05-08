import numpy as np

def hms2dec(h, m, s):
  dec = (15 * (h + (m/60) + (s/3600)))
  return dec

def dms2dec(h, m, s):
  if (h >= 0):
    dec = (h + (m/60) + (s/3600))
  else:
    dec = (-1 * ((h*-1) + (m/60) + (s/3600)))
  return dec

def import_bss():
  file = 'bss.dat'
  lines = np.loadtxt(file, usecols=range(1, 7))
  count=1
  result = [ ]
  for line in lines:
    result.append((count, hms2dec(line[0], line[1], line[2]), dms2dec(line[3], line[4], line[5])))
    count += 1
  return result

def angular_dist(a1, d1, a2, d2):
  a1 = np.radians(a1)
  d1 = np.radians(d1)
  a2 = np.radians(a2)
  d2 = np.radians(d2)

  p1 = np.sin(abs(d1-d2)/2)**2
  p2 = np.cos(d1)*np.cos(d2)*np.sin(abs(a1-a2)/2)**2
  p3 = 2*np.arcsin(np.sqrt(p1+p2))
  return np.degrees(p3)

def find_closest(cat, ra, dec):
  distmin = 1000
  minindex = 0
  
  for line in cat:
    dist = angular_dist(ra, dec, line[1], line[2])
    if dist < distmin:
      minindex = line[0]
      distmin = dist

  return (minindex, distmin)


if __name__ == '__main__':
  cat = import_bss()
  
  print(find_closest(cat, 175.3, -32.5))
  print(find_closest(cat, 32.2, 40.7))
