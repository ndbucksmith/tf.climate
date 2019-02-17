import numpy as np
import time
import rasterio as rio
import matplotlib
import matplotlib.pyplot as plt
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




def scat(x, y, z, cmp, name, nrm):
  fi_, ax_ = plt.subplots(1)
  fi_.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.99)
  fi_.suptitle(name)
  ax_.scatter(x, y, s=1, c=z, cmap=cmp, norm=nrm)
  return fi_, ax_

def plotter(plts, name, lbls):
  fi_, ax_ = plt.subplots(1)
  fi_.suptitle(name)
  for px in range(len(plts)):
    ax_.plot(plts[px], label=lbls[px] )
  ax_.legend()  
  return fi_, ax_

def addnote(fig):
  fig.text(0.02, 0.02, str(file_ct) + ' batches of 400 examples', transform=plt.gcf().transFigure)
  fig.text(0.5, 0.02,mdl_path , transform=plt.gcf().transFigure)
  return fig

maxes = []
for mcx in range(1):

  for tx in range(file_ct):
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
      trus_cat = trus
      wc_trus_cat = wc_trus
      
    else:
      trus_cat = np.concatenate((trus_cat, trus), axis=0)
      wc_trus_cat = np.concatenate((wc_trus_cat, wc_trus), axis=0)
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

fig1, axe1 = scat(wc_trus_cat, trus_cat, 'k', cmp=None, name='Global Solar vs worldclim.org average temp', nrm=None)
plt.show()
    
