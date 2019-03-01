import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu
import warnings
warnings.filterwarnings('error')
pst = pdb.set_trace

"""
create pkl files with dicts for each window
version 3 uses dem3 files for elevation.  
There are 24 files to cover eath with 15" res elevation data
this is 2x better reesolution gs elevation and also cover the planet
as in geotif nodata = None
this function is now rewrit to correctly use raterio index() and xy()
in place of my homebrew functions which now be removed

"""
targ = 'epoch3/'


def wd_filteredstats(wd, nodat):
  wdclean = wd
  if nodat < -3.0e+36:
    ndct = (wd < -3.0e+36).sum()
    dct = (wd > -3.0e+36).sum()
    wdclean[wdclean < -3.0e+36] = 0.0    
  else:  
    ndct = (wd == nodat).sum()
    dct = (wd != nodat).sum() 
    wdclean[wdclean == nodat] = 0
  if dct > 0:
    wdmean = wdclean.sum()/dct
  else:
    wdmean  = nodat
  return wdmean

hids = rio.open('wcdat/srad/wc2.0_30s_srad_01.tif')
teds = rio.open('wcdat/tavg/wc2.0_30s_tavg_01.tif')

lcds = rio.open('wcdat/lc/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif')
_st = time.time()
traincts=[]; testcts=[]; badels = []
for tilx  in range(24):
  elds = rio.open('wcdat/dem3/' + gtu.dem3_files[tilx])
  print('opening ' + gtu.dem3_files[tilx])
  upright = elds.xy(0,0); lowleft = elds.xy(10800, 14401)
  print('covering '+ str(upright) + ' to ' + str(lowleft))

  unit_x = 10  #30 second units
  unit_y = 10  #30 sec
  examp_ct = 0; hi_badct= 0; te_badct=0; lc_badct = 0; el_badct=0;
  trainct=0; noelct=0; nohict=0; notect=0; badct=0; 
  teix=[]; trix=[]; testct=0; dc={}; bads = [];
  for d3lax in range(0,14401, 2*unit_y):
    for d3lox in range(0 , 10801, 2*unit_x):
      
      #def xy(self, row, col, offset="center"):
      lon, lat = elds.xy(d3lax, d3lox)
      lrlon, lrlat = elds.xy(d3lax+(2*unit_y), d3lox+(2*unit_x))
      #def index(self, x, y, op=math.floor, precision=None):
      wclax, wclox = hids.index(lon, lat)
      wclrlax, wclrlox =  hids.index(lrlon, lrlat)
      wc_unit_y = wclrlax-wclax; wc_unit_x = wclrlox-wclox
      lclax, lclox = lcds.index(lon, lat)
      lclrlax, lclrlox =  lcds.index(lrlon, lrlat)
      lc_unit_y = lclrlax-lclax; lc_unit_x = lclrlox-lclox
      #Window(col_off, row_off, width, height)
      wcws = rio.windows.Window(wclox, wclax, wc_unit_x, wc_unit_y)
      lcws = rio.windows.Window(lclox, lclax, lc_unit_x, lc_unit_y)
      d3ws = rio.windows.Window(d3lox, d3lax, 2*unit_x, 2*unit_y)      
      el = elds.read(1, window=d3ws)
      hi = hids.read(1, window=wcws)
      te = teds.read(1, window=wcws)
      lc = lcds.read(1, window=lcws)    
      if wd_filteredstats(hi, 65535) != 65535: hi_datact = 10;
      else: hi_datact = 0
      if wd_filteredstats(te, -3.4e+38) > -150: te_datact = 10;
      else: te_datact = 0   
 
      lc_datact =  (lc.shape[0]*lc.shape[1]) -  (lc == 0).sum() - (lc == 255).sum()
      try:
        elev = el.mean()
        el_datact = (el > -500).sum()
      except:
        el_datact = 0
      if hi_datact > 3 and te_datact > 3 and lc_datact > 3 and el_datact > 3:
        if examp_ct % 5000 == 1:
          print(lon, lat, elev, hi_datact, te_datact, lc_datact)
        examp_ct +=1
    
        if np.random.random() < 0.9:
          trix.append([d3lax, d3lox, lon, lat, elev, hi_datact, te_datact, lc_datact, \
                       unit_x, unit_y, lc_unit_x, lc_unit_y]);
          trainct += 1; 
        else:
          teix.append([d3lax, d3lox, lon, lat, elev, hi_datact, te_datact, lc_datact, \
                       unit_x, unit_y, lc_unit_x, lc_unit_y]);
          testct += 1;
      else:
        badct +=1
        if hi_datact < 4: hi_badct +=1
        if te_datact < 4: te_badct +=1      
        if lc_datact < 4: lc_badct +=1
        if el_datact < 4: el_badct +=1;  
  
  dc['trix'] = np.array(trix);
  dc['teix'] = np.array(teix);
  dc['badct'] = badct;
  dc['trainct'] = trainct; dc['testct'] = testct
  print( trainct, testct,'time:', time.time() - _st)
  print(badct, hi_badct, te_badct, lc_badct, el_badct)

    #pdb.set_trace()
  with open( targ+ gtu.dem3_files[tilx][0:-4]  + '.pkl', 'w') as fo:
    dmp = pickle.dumps(dc)
    fo.write(dmp)
  traincts.append(trainct)
  testcts.append(testct)

dc = {}
testcts = np.array(testcts)
traincts = np.array(traincts)
dc['testcts'] = testcts
dc['traincts'] = traincts
with open(targ +'/summary_cts.pkl', 'w') as fo:
    dmp = pickle.dumps(dc)
    fo.write(dmp)
print(dc)
print('total train set size: ' + str(traincts.sum()))


