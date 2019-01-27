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

"""
trains two model classes in the same session

"""

params = {}
params['batch_size'] = 400
b_size = params['batch_size']
params['f_width'] = 17
params['x_size'] = 11
params['learn_rate'] = 0.001
params['init_stddev'] = 0.2
params['take'] = [2,3,4,5,6,7,8,9,10,11]
take = params['take']

params['x_size'] = len(params['take'])
pstr = "traing with: "
for idx in range(len(params['take'])):
  pstr += gtb.feat_list[take[idx]]
  pstr += ', '
print(pstr)
#pdb.set_trace()
  


mdl = gtm.ClimaModel(params)
#stupidmodel- a 4 parameter model
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
    fetch = [mdl.losser, mdl.h, mdl.ts, mdl.y_true, mdl.wtd_losser]
    smfetch = [smdl.losser, smdl.h, smdl.ts, smdl.y_true]  #smdl.ts,
    errs, ests, step, yt, wtderrs  = mdl.sess.run(fetch, feed)
    smerrs, smests, smstep, smyt  = mdl.sess.run(smfetch, smfeed) #smstep,
    #(514, 109.974335, 30.97795, 102.13406, 14.407771, -10.13775, 29.30625)
#(514, 120.75759, 30.98626, 94.057785, 14.407771, -10.13775, 29.30625)

    print(tx, errs, smerrs, wtderrs, ests.mean(), ests.min(), ests.max())
    print(ins[:,2].max(), ins[:,3].max(), ins[:,4].max(), ins[:,5].max(),\
             ins[:,6].max(), ins[:,7].max(), ins[:,8].max(),\
             ins[:,9].max(), ins[:,10].max(), ins[:,11].max())
    #pdb.set_trace()
  if errs < smerrs:
    pdb.set_trace()
  else:  
    mdl.reup()  # reinit and try again

    fet = [smdl.psens, smdl.esens, smdl.toasens, smdl.smb]
    ps, es, toas, smbias = mdl.sess.run(fet)
    print(ps, es, toas, smbias)
    pdb.set_trace()
  
for ix in range(params['batch_size']):
  #pdb.set_trace()
  smt = stupidmodel(ins[ix][2],ins[ix][5])
  
  gtu.arprint([ests[ix], yt[ix], smests[ix]])

  if ix  % 20 == 19:          
    pdb.set_trace()
