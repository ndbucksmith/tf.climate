import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb

"""
utilites for geotiff files
include building, saving loading lists of windows
longitude latutde to index conversion
toa power,alnedo  by latitude
"""
class latluts:
    lattitude = [-90.0,-80.0,-70.0,-60.0,-50.0,-40.0,-30.0,-20.0,-10.0, 0.0,
                 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    albedo = [0.73, 0.73, 0.4, 0.18, 0.15, 0.12, 0.12, 0.12, 0.11, 0.11,
              0.12, 0.15, 0.17, 0.16, 0.2, 0.30, 0.42, 0.52, 0.58]
    power = [131.6666667,167.1875,197.7083333,233.8541667,280.9375,326.1458333,
             362.3958333,391.7708333,409.375,414.5833333,411.0416667,395,
             366.9791667,331.9791667,287.9166667,241.6666667,206.25,176.1458333,
              140.7291667]

def albbylat(lattitude):
    return np.interp(lattitude, latluts.lattitude, latluts.albedo)

#top of atmosphere power
def toaPower(lat):
  P = np.interp(lat, latluts.lattitude, list(reversed(latluts.power)))
  return P


#deprecated
def get_windims(hei, wid, nocols, norows):
  rw = wid/nocols
  ch = hei/norows
  modrw = wid%nocols
  modch = hei%norows
  assert modrw == 0; assert modch == 0;
  return rw, ch, modrw, modch

#deprecated                     
def build_winlist(NestLat,SestLat, res, nocols, hemirows):
  windows = []   #Window(col_off, row_off, width, height)
  #build NHemi list row by row
  rw,ch, modrw, modch = get_windims(NestLat * res, 360 *res, nocols, hemirows)
  modch = 0; coff=0; roff=0;
  for loct in range(nocols):
    for lact in range(hemirows):
      windows.append(rio.windows.Window(loct*rw, lact*ch, rw, ch))
  eq_off = lact*ch
  rw,ch, modrw, modch = get_windims(SestLat * res, 360 *res, nocols, hemirows)
  for loct in range(nocols):
    for lact in range(hemirows):
      windows.append(rio.windows.Window(loct*rw, eq_off+(lact*ch),\
                      rw, ch))
  return windows

def save_winlist(wins, res, path):
  dc = {}
  dc['ct'] = len(wins)
  dc['res'] = res
  dc['windows'] = wins
  dmp = pickle.dumps(dc)
  with open(path, 'w') as fo:
    fo.write(dmp)

def load_winlist(path):
  with open(path, 'r') as fi:
    dc = pickle.load(fi)
  return dc['windows'], dc['res'], dc['ct']

#barometric pressure in torr (mmHg) by altitude
def bp_byalt(alt):
  alt = max(0, alt)
  return 760 *(1.0/(2.0**(alt/5400.0)))



#deprecate and use rasterio xy()
def llscale(lax, lox, res, startLat):
  res = float(res)
  lat = startLat - (float(lax)/res)
  lon = -180.0 + float(lox)/res
  return lat, lon

#deprecate and use raserio index()
def invllscale(lat, lon, res, startLat):
  lax = int((startLat - lat)* res)
  lox = int((lon+180.0)*res)
  return lax, lox

def arprint(inp):
  ostr = ""
  for ix in range(len(inp)):
    ostr += str(round(inp[ix], 2))
    ostr += "  "
  print ostr[0:-2]  
  return ostr[0:-2]

mon_wts = [31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
def acc12mo_avg(vals):
  wtd_sum = 0.0
  for mx in range(12):
    wtd_sum += (mon_wts[mx] * vals[mx])
  wtd_avg = wtd_sum/365.25
  return wtd_avg
