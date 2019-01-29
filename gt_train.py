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
params['f_width'] = 12

params['learn_rate'] = 0.0000003
params['init_stddev'] = 0.05
params['take'] = [2,3,4,5,6,7,8,9,10,11,12,13]
take = params['take']
params['x_size'] = len(params['take'])
pstr = "traing with: "
for idx in range(len(params['take'])):
  pstr += gtb.feat_list[take[idx]]
  pstr += ', '
print(pstr)

sess = tf.Session()
#stupidmodel- a 4 parameter model
smdl = gtm.artisanalModel(sess, params)
#climaModel is simple nn
mdl = gtm.ClimaModel(params, sess, smdl.arth)
sess.run(tf.global_variables_initializer())

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
          app.append([ins[kx][2]/ins[kx][4], gtu.bp_byalt(ins[kx][2])])
        #pdb.set_trace()
        ins = np.append(ins, app, axis=1)
        trus = dc['trus']

    feed = mdl.bld_feed(ins, trus)
    smfeed = smdl.bld_feed(ins, trus) 
    fetch = [mdl.losser, mdl.h, mdl.ts, mdl.y_true, mdl.wtd_losser]
    smfetch = [smdl.losser, smdl.arth, smdl.ts, smdl.smy_true, smdl.hp, smdl.tp, smdl.he]  #smdl.ts,
    #pdb.set_trace()
    smerrs, smests, smstep, smyt, smhp, smtp, smep  = smdl.sess.run(smfetch, smfeed)
    errs, ests, step, yt, wtderrs  = mdl.sess.run(fetch, feed)

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

  fet = [smdl.psens, smdl.esens, smdl.toasens, smdl.smb, smdl.bpsens]
  ps, es, toas, smbias, bpsen = mdl.sess.run(fet)
  print(ps.tolist(), es.tolist(), toas.tolist(), smbias.tolist(), bpsen.tolist())
 # pdb.set_trace()
  
for ix in range(params['batch_size']):
  #pdb.set_trace()
  gtu.arprint([ests[ix], yt[ix], smests[ix], smhp[ix], smep[ix], smtp[ix], ins[ix][2], ins[ix][4],  ins[ix][5]])

  if ix  % 20 == 19:          
    pdb.set_trace()
