import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import tensorflow as tf
import gt_utils as gtu
import gt_model as gtm
import wc_batcher as wcb
import os
pst = pdb.set_trace

"""
collects max and min data for all feature and true data channels across 
all batch files in target directory

copyright 2019 Nelson 'Buck' Smith

"""

params = {}
params['batch_size'] = 400
b_size = params['batch_size']
params['f_width'] = 12

params['learn_rate'] = 0.05
params['init_stddev'] = 0.05
params['take'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
take = params['take']
params['x_size'] = len(params['take'])
params['cell_size'] =  96
params['rxin_size'] = 22  # wcs + h1h + lwi + el
pstr = "trainng with: "
for idx in range(len(params['take'])):
  pstr += wcb.nn_features[take[idx]]
  pstr += ', '
print(pstr)
print('and')
print(wcb.rnn_features)

target = 'wc_v2'
file_ct = len(os.listdir(target)) 
#

maxes = []
for mcx in range(1):

  for tx in range(1526):
    start_t = time.time()
    if False:
      ins, trus = wcb.get_batch(params['batch_size'], True)
    else:
      with open('wc_v2/wcb_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        app = []
        rsqs =  dc['rnn_seqs']
        wc_trus = dc['ec_tru']  #alternative verion of reality, man
        rn_trus =  dc['rnn_trus']
        trus = dc['trus']
    rn_trus = np.array(rn_trus)
    rsqs = np.array(rsqs)
    trus = np.array(trus)
    wc_trus = np.array(wc_trus)
    if tx == 0:
      ins_maxes = np.amax(ins, axis=0)
      ins_mins = np.amin(ins, axis=0)
      rn_maxes = []; rn_mins = []
      rn_trumax = rn_trus.max()
      rn_trumin = rn_trus.min()
      trus_max = trus.max()
      wc_trus_max = wc_trus.max()
      trus_min = trus.min()
      wc_trus_min = wc_trus.min()    
      for rnx in range(4):
        rn_maxes.append(rsqs[:,rnx,:].max())
        rn_mins.append(rsqs[:,rnx,:].min())
    else:
      bmaxes =  np.amax(ins, axis=0)
      bmins = np.amin(ins, axis=0)
      for px in range(len(ins_maxes)):
        if bmaxes[px] > ins_maxes[px]:
          ins_maxes[px] =  bmaxes[px] 
        if bmins[px] <  ins_mins[px]:
          ins_mins[px] =  bmins[px]
      for rnx in range(4):
        if rsqs[:,rnx,:].max() > rn_maxes[rnx]:
          rn_maxes[rnx] = rsqs[:,rnx,:].max()
        if rsqs[:,rnx,:].min() < rn_mins[rnx]:
          rn_mins[rnx] = rsqs[:,rnx,:].min()
        if rn_trumax < rn_trus.max():
          rn_trumax = rn_trus.max()
        if rn_trumin > rn_trus.min():
          rn_trumin = rn_trus.min()
        if wc_trus_max < wc_trus.max():
          wc_trus_max = wc_trus.max()
        if wc_trus_min > wc_trus.min():
          wc_trus_min = wc_trus.min()
        if trus_max < trus.max():
          trus_max = trus.max()
        if trus_min > trus.min():
          trus_min = trus.min()
           
    print(rsqs[:,1,:].max())
                   
    if tx+1 == file_ct :
      gtu.arprint(ins_maxes)
      gtu.arprint(ins_mins)
      gtu.arprint(rn_maxes)
      gtu.arprint(rn_mins)
      gtu.arprint([rn_trumin, rn_trumax])
      gtu.arprint([wc_trus_min, wc_trus_max])
      gtu.arprint([trus_min, trus_max])   
      pst()
