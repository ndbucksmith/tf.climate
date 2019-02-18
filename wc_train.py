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
params['batch_size'] = 400
b_size = params['batch_size']
params['pref_width'] = 30
params['metaf_width'] = 31
params['mdl_path'] = 'mdls/nn3031cs50'
params['learn_rate'] = 0.05
params['init_stddev'] = 0.05
params['take'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 19, 20]
params['rnn_take'] = [0,1,2,3]
take = params['take']
params['yrly_size'] = len(params['take'])  #number of static once per year chanels
params['cell_size'] =  50
params['rxin_size'] = len(take) + len(params['rnn_take'])  # mdl uses this as rmdl.xin_size
pstr = "training with: "
for idx in range(len(params['take'])):
  pstr += wcb.nn_features[take[idx]]
  pstr += ', '
print(pstr)
target = 'wc_v2' # directory where batch files are
file_ct = len(os.listdir(target))
params['train_file_ct'] = file_ct
sess = tf.Session()
#pst()
rmdl = gtm.climaRNN(1, sess, params)
init_op = tf.global_variables_initializer()
sess.run(init_op)

  # loop for trying large number of model reinits
  # or for multiple runs thru set of training batches
train_history = []
for mcx in range(3):  
  for tx_ in range(file_ct):
    start_t = time.time()
    if mcx == 0:
      tx = tx_
    else:
      tx = np.random.randint(0, file_ct)
    if False:
      ins, trus = gtb.get_batch(params['batch_size'], True)
    else:
      with open(target + '/wcb_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        app = []
        rsqs =  dc['rnn_seqs']
        wc_trus = dc['ec_tru']  #alternative verion of reality, man
        rn_trus =  dc['rnn_trus']
        trus = dc['trus']
    #pdb.set_trace()
    feed = rmdl.bld_multiyearfeed(1, ins, rsqs, rn_trus, wc_trus)

    fetch = [rmdl.lossers, rmdl.hypos, rmdl.ts, rmdl.y_trues, rmdl.meta_ts, rmdl.meta_losser ]
 
    errs, ests, step, yt, mts, met_err   = sess.run(fetch, feed)
    errs =np.array(errs)
    train_history.append( [errs.mean(), errs.max(), errs.min(), met_err] )
    if tx % 100 == 99:
      pass #pdb.set_trace()
    gtu.arprint([tx, errs.mean(), errs.max(), errs.min(), met_err])

  if errs.mean() < 1.0:
    print('whoop') #pass # pdb.set_trace()

rmdl.save(params['mdl_path'] +'/climarnn_', file_ct)
tvars = tf.trainable_variables()
tvars_vals = sess.run(tvars)
with open(params['mdl_path'] + '/params.json', 'w') as fo:
  json.dump(params, fo)          

if True:
  for var, val in zip(tvars, tvars_vals):
    print(var.name,var.shape)

train_history = np.array(train_history)
fig, axe  = plt.subplots(1)
fig.suptitle('error vs.iteration')
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
axe.scatter(range(len(train_history)),train_history[:,0], c='k', s=1)
axe.scatter(range(len(train_history)),train_history[:,3], c='g', s=1)
oner = [1.0] * len(train_history)
pointsixer = [0.6]* len(train_history)
axe.plot(oner)
axe.plot(pointsixer)
plt.show()
