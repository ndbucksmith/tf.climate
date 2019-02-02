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

"""
trains rnn climate model using tf bidrectional dynamic rnn.  Using dynamic rnn even though 
our sequence length is static (12 months) becuase tf dynamic rnn is more flexible for data 
martialing



copyright 2019 Nelson 'Buck' Smith

"""

params = {}
params['batch_size'] = 400
b_size = params['batch_size']
params['f_width'] = 12

params['learn_rate'] = 0.05
params['init_stddev'] = 0.05
params['take'] = [2,3,4,5,6,7,8,9,10,11,12,13]
take = params['take']
params['x_size'] = len(params['take'])
params['cell_size'] =  53
params['rxin_size'] = 21  # wcs + h1h + lwi + el
pstr = "traing with: "
for idx in range(len(params['take'])):
  pstr += wcb.nn_features[take[idx]]
  pstr += ', '
print(pstr)

sess = tf.Session()

rmdl = gtm.climaRNN(sess, params)

sess.run(tf.global_variables_initializer())

for mcx in range(2):

  for tx in range(999):
    start_t = time.time()
    if False:
      ins, trus = gtb.get_batch(params['batch_size'], True)
    else:
      with open('wc_bs/wcb_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        app = []
        rsqs =  dc['rnn_seqs']
        wc_trus = dc['ec_tru']  #alternative verion of reality, man
        rn_trus =  dc['rnn_trus']
        trus = dc['trus']
    #pdb.set_trace()
    feed = rmdl.bld_feed(ins, rsqs, rn_trus)

    fetch = [rmdl.lossers, rmdl.hypos, rmdl.ts, rmdl.y_trues, ]
 
    errs, ests, step, yt   = sess.run(fetch, feed)
    errs =np.array(errs)
    if tx % 100 == 99:
      pass #pdb.set_trace()
    gtu.arprint([tx, errs.mean(), errs.max(), errs.min(),  ests[6].mean(), ests[6].min(), ests[6].max(), \
                 yt[6].mean(), yt[6].min(), yt[6].max()])

  if errs.mean() < 80.0:
    pass # pdb.set_trace()

rmdl.save('mdl/climarnn_', 550)
 # pdb.set_trace()
  
