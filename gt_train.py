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
import gt_batcher as gtb

fl_fmtr = lambda x: round(x,1) % x
np.set_printoptions(formatter={'float_kind':fl_fmtr})

params = {}
params['batch_size'] = 400
b_size = params['batch_size']
params['f_width'] = 17
params['x_size'] = 11
params['learn_rate'] = 0.001
params['init_stddev'] = 0.2
params['take'] = [2,4,5]
take = [2,4,5]

params['x_size'] = len(params['take'])
pstr = "traing with: "
for idx in range(len(params['take'])):
  pstr += gtb.feat_list[take[idx]]
  pstr += ', '
print(pstr)
pdb.set_trace()
  
def stupidmodel(pwr, el):
  temp = (pwr*0.2) -25 -(el * 0.0064)
  return temp

mdl = gtm.ClimaModel(params)
smdl = gtm.stupidModel(mdl.sess, params)

for mcx in range(1000):

  for tx in range(515):
    start_t = time.time()
    if False:
      ins, trus = gtb.get_batch(params['batch_size'], True)
    else:
      with open('batches/b_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        trus = dc['trus'] 
    feed = mdl.bld_feed(ins, trus)
    smfeed = smdl.bld_feed(ins, trus) 
    fetch = [mdl.losser, mdl.h, mdl.ts, mdl.y_true]
    smfetch = [smdl.losser, smdl.h, smdl.ts, smdl.y_true]
    errs, ests, step, yt  = mdl.sess.run(fetch, feed)
    smerrs, smests, smstep, smyt  = mdl.sess.run(smfetch, smfeed)
    print(tx, errs, smerrs)
  if errs < smerrs:
    pdb.set_trace()
  else:  
    mdl.reup()  # reinit and try again
  
for ix in range(params['batch_size']):
  #pdb.set_trace()
  smt = stupidmodel(ins[ix][2],ins[ix][5])
  
  gtu.arprint([ests[ix], yt[ix], smests[ix]])

  if ix  % 20 == 19:          
    pdb.set_trace()
