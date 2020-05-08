def hms2dec(h, m, s):
  dec = (15 * (h + (m/60) + (s/3600)))
  return dec

def dms2dec(h, m, s):
  if (h >= 0):
    dec = (h + (m/60) + (s/3600))
  else:
    dec = (-1 * ((h*-1) + (m/60) + (s/3600)))
  return dec


if __name__ == '__main__':
  
  print(hms2dec(23, 12, 6))
  print(dms2dec(22, 57, 18))
  print(dms2dec(-66, 5, 5.1))
