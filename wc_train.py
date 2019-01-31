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
trains two model classes in the same session

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
params['cell_size'] = 47
params['rxin_size'] = 21  # wcs + h1h + lwi + el
pstr = "traing with: "
for idx in range(len(params['take'])):
  pstr += wcb.nn_features[take[idx]]
  pstr += ', '
print(pstr)

sess = tf.Session()

rmdl = gtm.climaRNN(sess, params)
sess.run(tf.global_variables_initializer())

for mcx in range(500):

  for tx in range(80):
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
    gtu.arprint([tx, errs.mean(), errs.max(), errs.min(),  ests[0].mean(), ests[0].min(), ests[0].max()])

  if errs.mean() < 80.0:
    pdb.set_trace()

  
 # pdb.set_trace()
  
