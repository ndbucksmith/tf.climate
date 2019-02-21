import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu

"""
create pkl files with dicts for each window
"""
targ = 'winstats2'

elds = rio.open('wcdat/ELE.tif')
hids = rio.open('wcdat/GHI.tif')
teds = rio.open('wcdat/TEMP.tif')
_st = time.time()
traincts=[]; testcts=[]; badels = []
for _start in range(0, 105, 5):
  _Unit = 20
  _StartLax = _start * 120
  if _start < 109:
    _EndLax  = (5 + _start) * 120
  else:
    _EndLax  = (4  + _start) * 120    
  trainct=0; noelct=0; nohict=0; notect=0; badct=0
  teix=[]; trix=[]; testct=0; dc={}; bads = [];
  for lax in range(_StartLax, _EndLax, _Unit):
    for lox in range(0 , 360*120, _Unit):
      #pdb.set_trace()
      ws = rio.windows.Window(lox, lax, _Unit, _Unit)
      el = elds.read(1, window=ws)
      hi = hids.read(1, window=ws)
      te = teds.read(1, window=ws)
      datact = 0; badblock = False
      for blax in range(_Unit):
        for blox in range(_Unit):
          if not math.isnan(el[blax][blox]) and \
             not math.isnan(hi[blax][blox]) and \
             not math.isnan(te[blax][blox]):  
               datact += 1          
             
      if np.random.random() < 0.9  and datact != 0 \
         and np.nanmean(el) > -50:
        trix.append([lax, lox, datact]);trainct += 1; 
      elif datact != 0 and np.nanmean(el) > -50:
        teix.append([lax, lox, datact]); testct += 1;
      elif  np.nanmean(el) > -50:
        bads.append([lax, lox, datact]); badct += 1;
      #if lox == 0:
       # print("@", lax, trainct, testct)
  
  dc['trix'] = np.array(trix);
  dc['teix'] = np.array(teix);
  dc['bads'] = np.array(bads); dc['badct'] = badct;
  dc['trainct'] = trainct; dc['testct'] = testct
  print( _start, trainct, testct,'time:', time.time() - _st)
  print(badct)
  print(bads)
    #pdb.set_trace()
  with open( targ+ '/book_' + str(_start) + '.pkl', 'w') as fo:
    dmp = pickle.dumps(dc)
    fo.write(dmp)
  traincts.append(trainct)
  testcts.append(testct)
pdb.set_trace()
dc = {}
testcts = np.array(testcts)
traincts = np.array(traincts)
dc['testcts'] = testcts
dc['traincts'] = traincts
with open(targ +'/summary_cts.pkl', 'w') as fo:
    dmp = pickle.dumps(dc)
    fo.write(dmp)
print(dc)
