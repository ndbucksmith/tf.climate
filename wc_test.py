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
pst = pdb.set_trace

"""
tests with batches from segregated test data set
prints many graphs
graph code needs refactoring

copyright 2019 Nelson 'Buck' Smith

"""

params = {}
params['batch_size'] = 400
b_size = params['batch_size']
params['f_width'] = 12

params['learn_rate'] = 0.05
params['init_stddev'] = 0.05
params['take'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19]
take = params['take']
params['x_size'] = len(params['take'])
params['cell_size'] = 64
params['rxin_size'] = 23  # wcs + h1h + lwi + el
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

rmdl.restore('mdls/climarnn_1157.ckpt')
tvars = tf.trainable_variables()
tvars_vals = sess.run(tvars)

if True:
  for var, val in zip(tvars, tvars_vals):
    print(var.name,var.shape)
print(val)
#pdb.set_trace()
test_map = []
left = 0.05
bottom = 0.05 
width = 0.9
height = 0.9

for mcx in range(1):

  for tx in range(10):

    if False:
      ins, trus = gtb.get_batch(params['batch_size'], True)
    else:
      with open('wc_v2test/wcb_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        app = []
        rsqs =  dc['rnn_seqs']
        wc_trus = dc['ec_tru']  #alternative verion of reality, man
        rn_trus =  dc['rnn_trus']
        trus = dc['trus']
    feed = rmdl.bld_multiyearfeed(1, ins, rsqs, rn_trus, trus)
    #feed = rmdl.bld_feed(ins, rsqs, rn_trus)

    fetch = [rmdl.lossers, rmdl.hypos,  rmdl.y_trues, rmdl.meta_h, rmdl.meta_loss, \
                rmdl.meta_yt]


    ploterrs = False
    if ploterrs:
      fig, axes = plt.subplots(4, sharex=True)
      fig.suptitle('Error by month for tropic and temperate bands')
      fig.subplots_adjust(top=0.95, bottom=0.01, left=0.1, right=0.99)
      axes[0].set_title('30N to 60N', fontsize=8)
      axes[0].xaxis.set_visible(False)
      axes[1].set_title('Eq to 30N', fontsize=8)
      axes[2].set_title('30S to 3Eq', fontsize=8)
      axes[3].set_title('~52S to 30S', fontsize=8)

      
    sq_errs, ests, yt, meta_ests, meta_sqerrs, meta_yt   = sess.run(fetch, feed)
    sq_errs =np.array(sq_errs)
    ests = np.array(ests)
    yt = np.array(yt)
    errs =np.array(ests - yt)
    meta_errs = np.array(meta_ests - meta_yt)
    overall_errs = np.mean(errs, axis=0)
    print('errs mean, max, min, std, plus overall_errs mean, max min')
    gtu.arprint([tx, errs.mean(), errs.max(), errs.min(), errs.std(),\
                  overall_errs.mean(), overall_errs.max(), overall_errs.min(), sq_errs.mean()])
    print('meta model errs mean, min, max')
    gtu.arprint([meta_errs.mean(), meta_errs.max(), meta_errs.min(), meta_sqerrs.mean()])
    

    
    for bx in range(b_size):
      if ins[bx,1] > 30.0 and ploterrs:
        axes[0].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
      elif ins[bx,1] > 0.0 and ploterrs:
        axes[1].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
      elif ins[bx,1] > -30.0 and ploterrs:
        axes[2].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
      elif ploterrs:
        axes[3].plot([1,2,3,4,5,6,7,8,9,10,11,12], errs[:,bx])
    print('location plus top 5 features for worst errors')
   # pst()
    for mx in range(12):
      badbx = np.argmax(errs[mx])
 #     gtu.arprint(ins[badbx,0:7]+ errs[mx,badbx])
      badbx = np.argmin(sq_errs[mx])
#      gtu.arprint(ins[badbx,0:7]+ errs[mx,badbx])

    #add some wiggle and check the jiggle.  
    watts = 3.7

    feed[rmdl.xin][:,:,1] = feed[rmdl.xin][:,:,1] + watts
   # feed[rmdl.xin][:,:,2] = feed[rmdl.xin][:,:,2] + watts # toa solar
    feed[rmdl.xin][:,:,7] = feed[rmdl.xin][:,:,7] + (watts*75.0)
    feed[rmdl.xin][:,:,19] = feed[rmdl.xin][:,:,19] + (watts*75.0)
   
    # get temperature estimates with extra power added in
    d_sq_errs, d_ests, d_yt, dmeta_h, dmeta_err, d_meta_yt   = sess.run(fetch, feed)
    dTdP = (d_ests - ests)/watts  # in unit degreeC per watt per meter square
    meta_dTdP = (dmeta_h - meta_ests)/watts
    print('power sensitivity - mean, max, min, stdev, count negative')
    gtu.arprint([dTdP.mean(), dTdP.max(), dTdP.min(), dTdP.std(), (dTdP < 0.0).sum()])
    gtu.arprint([meta_dTdP.mean(), meta_dTdP.max(), meta_dTdP.min(), meta_dTdP.std(), (meta_dTdP < 0.0).sum()])
# ____________ make the test map    
    for bx in range(b_size):
      test_map.append(np.concatenate( (ins[bx,0:7],errs[:,bx,0],overall_errs[bx], \
                                       dTdP[:,bx,0], [dTdP[:,bx,0].mean()],meta_errs[bx],  \
                                       meta_dTdP[bx], [bx], [tx]) ))   
    #plt.show()



test_map = np.array(test_map)
print('test map dataset shape, max and min sensitivty')
print(test_map.shape, test_map[:,32].max(), test_map[:,32].min())
sen_histo = np.histogram(test_map[:,34], [-1.0,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4] )
print(sen_histo)
fig21, axe21 = plt.subplots(1)
fig21.suptitle('sensitivity distribution')
axe21.bar(range(10), sen_histo[0], align='center', )
         # tick=['-0.5','-0.35','-0.25','-0.15','-0.05','0.05','0.15','0.25','0.35','0.7'])
axe21.set_xticklabels(['-0.5','-0.35','-0.25','-0.05','0.05','0.15','0.25','0.25','0.35','0.45','0.7'])



normalize = matplotlib.colors.Normalize(vmin=test_map[:,32].min(), vmax=test_map[:,32].max())
fig2, axe2 = plt.subplots(1)
fig2.suptitle('sensitivity map from test data')
fig2.subplots_adjust(top=0.95, bottom=0.01, left=0.1, right=0.99)
axe2.scatter(test_map[:,0], test_map[:,1], c = test_map[:,32], s=1, cmap='RdBu')
cax, _ = matplotlib.colorbar.make_axes(axe2)
cbar = matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap='RdBu')

normalize3 = matplotlib.colors.Normalize(vmin=test_map[:,33].min(), vmax=test_map[:,33].max())
fig3, axe3 = plt.subplots(1)
fig3.suptitle('overall error map from test data')
fig3.subplots_adjust(top=0.95, bottom=0.01, left=0.1, right=0.99)
axe3.scatter(test_map[:,0], test_map[:,1], c = test_map[:,33], s=1, cmap='RdBu')
cax3, _3 = matplotlib.colorbar.make_axes(axe3)
cba3 = matplotlib.colorbar.ColorbarBase(cax3, norm=normalize3, cmap='RdBu')


#normalize3 = matplotlib.colors.Normalize(vmin=test_map[:,17].min(), vmax=test_map[:,17].max())
fig4, axe4 = plt.subplots(1)
fig4.suptitle('sensitivity v elevation')
fig4.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
axe4.scatter(test_map[:,4], test_map[:,32], s=1, c= test_map[:,32], cmap='RdBu')

#normalize3 = matplotlib.colors.Normalize(vmin=test_map[:,17].min(), vmax=test_map[:,17].max())
fig5, axe5 = plt.subplots(1)
fig5.suptitle('sensitivity v latitude')
fig5.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
axe5.scatter(test_map[:,1], test_map[:,32], s=1, c= test_map[:,32], cmap='RdBu')

if False:
  fig6, axe6 = plt.subplots(1)
  fig6.suptitle('sensitivity v surface solar power')
  fig6.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
  axe6.scatter(test_map[:,2], test_map[:,32], s=1, c= test_map[:,32], cmap='RdBu')



if False:
  fig7, axe7 = plt.subplots(1)
  fig7.suptitle('sensitivity v toa solar')
  fig7.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
  axe7.scatter(test_map[:,3], test_map[:,32], s=1, c= test_map[:,32], camp='RdBu')


#normalize3 = matplotlib.colors.Normalize(vmin=test_map[:,17].min(), vmax=test_map[:,17].max())
fig8, axe8 = plt.subplots(1)
fig8.suptitle('error v latitude')
fig8.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
axe8.scatter(test_map[:,1], test_map[:,33], s=1, c= test_map[:,33], cmap='RdBu')

fig9, axe9 = plt.subplots(1)
fig9.suptitle('error v elevation')
fig9.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
axe9.scatter(test_map[:,4], test_map[:,33], s=1, c=test_map[:,33], cmap='RdBu')

plt.show()

  
 # pdb.set_trace()
  
