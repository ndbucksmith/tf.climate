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
params['f_width'] = 11

params['learn_rate'] = 0.3
params['init_stddev'] = 0.5
params['take'] = [2,4,5,12, 13] #7,8,9,10,11,12,13]
take = params['take']
params['x_size'] = len(params['take'])
pstr = "traing with: "
for idx in range(len(params['take'])):
  pstr += gtb.feat_list[take[idx]]
  pstr += ', '
print(pstr)
#pdb.set_trace()
  
#climaModel is simple nn
mdl = gtm.ClimaModel(params)
#stupidmodel- a 4 parameter model
smdl = gtm.artisanalModel(mdl.sess, params)

for mcx in range(1):

  for tx in range(539):
    start_t = time.time()
    if False:
      ins, trus = gtb.get_batch(params['batch_size'], True)
    else:
      with open('batches/b_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        app = []
        for kx in range(b_size):
          app.append([ins[kx][2]/ins[kx][5], gtu.bp_byalt(ins[kx][2])])
        #pdb.set_trace()
        ins = np.append(ins, app, axis=1)
        trus = dc['trus'] 
    feed = mdl.bld_feed(ins, trus)
    smfeed = smdl.bld_feed(ins, trus) 
    fetch = [mdl.losser, mdl.h, mdl.ts, mdl.y_true, mdl.wtd_losser]
    smfetch = [smdl.losser, smdl.h, smdl.ts, smdl.y_true]  #smdl.ts,
    errs, ests, step, yt, wtderrs  = mdl.sess.run(fetch, feed)
    smerrs, smests, smstep, smyt  = mdl.sess.run(smfetch, smfeed) #smstep,
 #(514, 110.60154, 30.97823, 101.80145, 14.603874, 12.420467, 17.50097)
    if tx % 100 == 99:
      pass #pdb.set_trace()
    print(tx, errs, smerrs, wtderrs, ests.mean(), ests.min(), ests.max())
    #print(ins[:,2].max(), ins[:,3].max(), ins[:,4].max(), ins[:,5].max(),\
    #         ins[:,6].max(), ins[:,7].max(), ins[:,8].max(),\
    #         ins[:,9].max(), ins[:,10].max(), ins[:,11].max())
    #pdb.set_trace()
  if errs < smerrs:
    pdb.set_trace()
  else:  
    pass #mdl.reup()  # reinit and try again

  fet = [smdl.psens, smdl.esens, smdl.toasens, smdl.smb]
  ps, es, toas, smbias = mdl.sess.run(fet)
  print(ps.tolist(), es.tolist(), toas.tolist(), smbias.tolist())
 # pdb.set_trace()
  
for ix in range(params['batch_size']):
  #pdb.set_trace()
  gtu.arprint([ests[ix], yt[ix], smests[ix]])

  if ix  % 20 == 19:          
    pdb.set_trace()
