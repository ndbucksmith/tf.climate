import numpy as np
import time
import rasterio as rio
import matplotlib.pyplot as plt
import pickle
import pdb
import math
import tensorflow as tf
import gt_utils as gtu
import gt_model as gtm
import wc_batcher as wcb
import os
import json
pst = pdb.set_trace

"""
trains rnn climate model using tf bidrectional dynamic rnn.  Using dynamic rnn even though 
our sequence length is static (12 months) becuase tf dynamic rnn is more flexible for data 
marshalling



copyright 2019 Nelson 'Buck' Smith

"""

params = {}
params['sqex'] = False
params['batch_size'] = 400
b_size = params['batch_size']
params['pref_width'] = 32
params['metaf_width'] = 32
params['mdl_path'] = 'mdls/v3Z_no_squex'
params['learn_rate'] = 0.05
params['init_stddev'] = 0.2
params['take'] = [0,1,3,4,9,10,11,12,13,14,15,]
params['rnn_take'] = [0,1,2,3,]; rn_take=params['rnn_take']
take = params['take']
params['yrly_size'] = len(params['take'])  #number of static once per year chanels
params['cell_size'] =  32
params['rxin_size'] = len(take) + len(params['rnn_take'])  # mdl uses this as rmdl.xin_size
pstr = "training with: "

for idx in range(len(params['take'])):
  pstr += wcb.nn_features[take[idx]]
  pstr += ', '
for idx in range(len(rn_take)):
  pstr += wcb.rnn_features[rn_take[idx]]
  pstr += ', '
print(pstr)
target = 'wc_v3' # directory where batch files are

sess = tf.Session()
#pst()
rmdl = gtm.climaRNN(1, sess, params)
init_op = tf.global_variables_initializer()
sess.run(init_op)

zb = wcb.zbatch(params['batch_size'])
  # loop for trying large number of model reinits
  # or for multiple runs thru set of training batches
save_good_model = False; multitrain_history = []
for mcx in range(40):
  train_history = []
  sess.run(init_op)
  file_ct = 15000   #  len(os.listdir('epochZ/NST/train'))
  print file_ct
  params['train_file_ct'] = file_ct
  for tx_ in range(file_ct):
    start_t = time.time()
    #randomization now done on the fly in zone batcher
    ins, rsqs, wc_trus, rn_trus, d3_idx  = zb.zbatch(params['batch_size'], True)
    b_time = time.time()- start_t; # print(b_time);

    #pdb.set_trace()
    feed = rmdl.bld_multiyearfeed(1, ins, rsqs, rn_trus, wc_trus)

    fetch = [rmdl.lossers, rmdl.hypos, rmdl.ts, rmdl.y_trues, rmdl.meta_ts, rmdl.meta_losser ]
 
    errs, ests, step, yt, mts, met_err   = sess.run(fetch, feed)
    errs =np.array(errs)
    #gtu.arprint([mcx, tx_, errs.mean(), errs.max(), errs.min(), met_err])
    
    train_history.append( [errs.mean(), errs.max(), errs.min(), met_err] )
    if tx_ == (file_ct)-1 or tx_ % 500  == 499 or tx_ ==0:
      if tx_ ==0: 
        gtu.arprint([mcx, tx_, errs.mean(), errs.max(), errs.min(), met_err])
      else:
        last500 = np.array(train_history)[-490:,:]
        gtu.arprint([mcx, tx_, 'rnn:', last500[:,0].min(), last500[:,0].mean(), last500[:,0].max()])
        gtu.arprint([mcx, tx_, 'meta:', last500[:,3].min(), last500[:,3].mean(), last500[:,3].max()])
        if last500[:,3].mean() < 0.41  or  last500[:,3].max() < 0.79:
          print('saving a v good meta  model')
          save_good_model = True
          break
        if  last500[:,0].max() < 1.0:
          print('saving a v good rnn  model')
          save_good_model = True
          break
        if tx_ == file_ct -1:
          if not save_good_model:
            break


  
  multitrain_history.append(train_history)
  if save_good_model: break;

if save_good_model:
  rmdl.save(params['mdl_path'] +'/climarnn_', file_ct)
  with open(params['mdl_path'] + '/params.json', 'w') as fo:
    json.dump(params, fo)
  pst()
  with open(params['mdl_path'] + '/train_hist.pkl', 'w') as fo:
    dmp = pickle.dumps(train_history)
    fo.write(dmp)
    
  with open(params['mdl_path'] + '/multitrain_history.pkl', 'w') as fo:
    dmp = pickle.dumps(multitrain_history,)
    fo.write(dmp)    
"""
work around due to  issues with x11 fwd and tmux.  graphing in another .py file


multitrain_history = np.array(multitrain_history)

sum_msg = 'last 500 meta min mean max:' + str(round(last500[:,3].min(),3)) + '    ' + str(round(last500[:,3].mean(),3)) +  \
          '    ' + str(round(last500[:,3].max(),3))
sum_msg2 = 'last 500 rnn min mean max:' + str(round(last500[:,0].min(),3)) + '    ' + str(round(last500[:,0].mean(),3)) + \
           '    ' + str(round(last500[:,0].max(),3))
fig, axe  = plt.subplots(1)
fig.suptitle('error vs.iteration')
fig.text(0.15, 0.75,sum_msg, transform=plt.gcf().transFigure)
fig.text(0.15, 0.7,sum_msg2, transform=plt.gcf().transFigure)
fig.text(0.15, 0.6,params['mdl_path'], transform=plt.gcf().transFigure)
fig.text(0.15, 0.65,str(file_ct) + ' batches', transform=plt.gcf().transFigure)
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
axe.scatter(range(len(train_history)),train_history[:,0], c='k', s=1)
axe.scatter(range(len(train_history)),train_history[:,3], c='g', s=1)
oner = [1.0] * len(train_history)
pointsixer = [0.5]* len(train_history)
axe.plot(oner)
axe.plot(pointsixer)
plt.show()




"""
