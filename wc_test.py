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
pst = pdb.set_trace

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
params['take'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18]
take = params['take']
params['x_size'] = len(params['take'])
params['cell_size'] = 128
params['rxin_size'] = 22  # wcs + h1h + lwi + el
pstr = "channels:: "
for idx in range(len(params['take'])):
  pstr += wcb.nn_features[take[idx]]
  pstr += ', '
print(pstr)

sess = tf.Session()

rmdl = gtm.climaRNN(1, sess, params, bTrain=False)
init_op = tf.global_variables_initializer()
sess.run(init_op)

#tf.reset_default_graph()

rmdl.restore('mdls/climarnn_999.ckpt')
tvars = tf.trainable_variables()
tvars_vals = sess.run(tvars)

if True:
  for var, val in zip(tvars, tvars_vals):
    print(var.name,var.shape)
print(val)
#pdb.set_trace()

for mcx in range(1):

  for tx in range(5):
    fig, axes = plt.subplots(4)
    fig.suptitle('Error by month for >30N, >0, >30S, >60S')
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
    feed = rmdl.bld_multiyearfeed(1, ins, rsqs, rn_trus)
    #feed = rmdl.bld_feed(ins, rsqs, rn_trus)

    fetch = [rmdl.lossers, rmdl.hypos,  rmdl.y_trues, ]
 
    sq_errs, ests, yt   = sess.run(fetch, feed)
    sq_errs =np.array(sq_errs)
    ests = np.array(ests)
    yt = np.array(yt)
    errs =np.array(ests - yt)
    print('errs mean, max, min, std, plus sq_errs mean, max min')
    gtu.arprint([tx, errs.mean(), errs.max(), errs.min(), errs.std(),\
                  sq_errs.mean(), sq_errs.max(), sq_errs.min()])
    for bx in range(b_size):
      if ins[bx,1] > 30.0:
        axes[0].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
      elif ins[bx,1] > 0.0:
        axes[1].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
      elif ins[bx,1] > -30.0:
        axes[2].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
      else:
        axes[3].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
    print('location plus top 5 features for worst errors')
    for mx in range(12):
      badbx = np.argmax(sq_errs[mx])
      gtu.arprint(ins[badbx,0:7]+ errs[mx,badbx])
      badbx = np.argmin(sq_errs[mx])
      gtu.arprint(ins[badbx,0:7]+ errs[mx,badbx])

    
    gtu.arprint(feed[rmdl.xin][0,0:4,18])
    feed[rmdl.xin][:,:,1] = feed[rmdl.xin][:,:,1] + 1.0
    feed[rmdl.xin][:,:,18] = feed[rmdl.xin][:,:,18] + (1.0*80)
    gtu.arprint(feed[rmdl.xin][0,0:4,18])    
 
    d_sq_errs, d_ests, d_yt   = sess.run(fetch, feed)
    dTdP = d_ests - ests
    print('vis power sensitivity mean, max, min, stdev, count negative')
    gtu.arprint([dTdP.mean(), dTdP.max(), dTdP.min(), dTdP.std(), (dTdP < 0.0).sum()])
    pst()    
    plt.show()
    pst()

 


  
 # pdb.set_trace()
  
