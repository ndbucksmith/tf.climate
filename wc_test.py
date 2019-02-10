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
import os
import json
pst = pdb.set_trace

"""
tests with batches from segregated test data set
prints many graphs
graph code needs refactoring

copyright 2019 Nelson 'Buck' Smith

"""


target = 'wc_v2test'
file_ct = len(os.listdir(target))
mdl_path = 'mdls/take2_10'
if True:
  with open('mdls/take2_10'+'/params.json', 'r') as fi:
    params = json.load(fi)
    take = params['take']
    b_size = params['batch_size']
else:
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
wc_start = len(params['take'])
gs_radx = 0  #index used for wiggle jiggle sensitivity
wc_radx = 6
assert wcb.nn_features[take[gs_radx]]  == 'vis_down'
assert wcb.nn_features[take[wc_radx]]  == 'wc_srad'

def tabler(ctxt, name, colhdrs):
  fi, ax = plt.subplots(1)
  fi.suptitle(name)
  fi.subplots_adjust(top=0.95, bottom=0.01, left=0.1, right=0.99)
  tbl = ax.table(cellText=ctxt, colLabels=colhdrs, loc='center')
  ax.axis('tight')
  ax.axis('off')
  #ax[0].xticks([])
  #ax[0].yticks([])
  return fi, ax

def mapper(x,y,z, cmp, name, nrm):
  fi, ax = plt.subplots(1)
  fi.suptitle(name)  #'sensitivity map from test data 1 deg C mse'
  fi.subplots_adjust(top=0.95, bottom=0.01, left=0.1, right=0.99)
  ax.scatter(x, y, c=z, s=1, cmap=cmp, norm=nrm)
  cax, _ = matplotlib.colorbar.make_axes(ax)
  cbar = matplotlib.colorbar.ColorbarBase(cax, norm=nrm,  cmap=cmp)
  return fi, ax

def scat(x, y, z, cmp, name, nrm):
  fi_, ax_ = plt.subplots(1)
  fi_.suptitle(name)
  fi_.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
  ax_.scatter(x, y, s=1, c=z, cmap=cmp, norm=nrm)
  return fi_, ax_

def plotter(plts, name, lbls):
  fi_, ax_ = plt.subplots(1)
  fi_.suptitle(name)
  for px in range(len(plts)):
    ax_.plot(plts[px], label=lbls[px] )
  ax_.legend()  
  return fi_, ax_

def addnote(fig):
  fig.text(0.02, 0.02, str(file_ct) + ' batches of 400 examples', transform=plt.gcf().transFigure)
  return fig

def get_worst_errs(errs, sq_errs):
  for mx in range(12):
    badbx = np.argmax(errs[mx])
 #  gtu.arprint(ins[badbx,0:7]+ errs[mx,badbx])
    badbx = np.argmin(sq_errs[mx])
#   gtu.arprint(ins[badbx,0:7]+ errs[mx,badbx])

sess = tf.Session()
rmdl = gtm.climaRNN(1, sess, params, bTrain=False)
init_op = tf.global_variables_initializer()
sess.run(init_op)
#tf.reset_default_graph()
rmdl.restore(mdl_path + '/climarnn_1657.ckpt')
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
mm_errs = []
for mcx in range(1):

  for tx in range(file_ct):

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
    feed = rmdl.bld_multiyearfeed(1, ins, rsqs, rn_trus, wc_trus)
    #feed = rmdl.bld_feed(ins, rsqs, rn_trus)

    fetch = [rmdl.lossers, rmdl.hypos,  rmdl.y_trues, rmdl.meta_h, rmdl.meta_loss, \
                rmdl.meta_yt, rmdl.dodi]
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

      
    sq_errs, ests, yt, meta_ests, meta_sqerrs, meta_yt, dodis   = sess.run(fetch, feed)
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
    mme = [meta_errs.mean(), meta_errs.max(), meta_errs.min(), meta_sqerrs.mean()]
    gtu.arprint(mme)
    mm_errs.append(mme)
    print('dT/ dInputs;')
    dodis = np.array(dodis[0])
    
    print(dodis[:,:,gs_radx].mean(), dodis[:,:,wc_radx].mean(), dodis[:,:,wc_start].mean(), )
    f90, ax0 = plotter([dodis[0,:,gs_radx],dodis[0,:,wc_radx],dodis[0,:,wc_start]], name='sensi' + str(tx),  \
                         lbls=['vis','srad','21mosr'])
    #plt.show()
    #XSpst()

    
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


    #add some wiggle and check the jiggle.  
    watts = 3.7  # add to global solar srad as watts and wc 12 month and monthly in their wacky units
    feed[rmdl.xin][:,:,gs_radx] = feed[rmdl.xin][:,:,gs_radx] + watts
    feed[rmdl.xin][:,:,wc_radx] = feed[rmdl.xin][:,:,wc_radx] +  (watts*75.0)
    feed[rmdl.xin][:,:,wc_start] = feed[rmdl.xin][:,:,wc_start] + (watts*75.0)
    # fetch temperature estimates with extra power added in from tf session run
    d_sq_errs, d_ests, d_yt, dmeta_h, dmeta_err, d_meta_yt,d_dodis = sess.run(fetch, feed)
    # calculate sensitivity tosurface  solar power
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
print(test_map.shape, test_map[:,34].max(), test_map[:,34].min())
sen_histo = np.histogram(test_map[:,34] )
print('histo')
print(zip(sen_histo[0], sen_histo[1][0:-1]))

fig21, axe21 = plt.subplots(1)
fig21.suptitle('sensitivity distribution')
axe21.bar(range(10), sen_histo[0], align='center', )
ticklbls = []
for ix in range(len(sen_histo[1][0:-1])):
  ticklbls.append( str(round(sen_histo[1][ix+1], 3)))

axe21.set_xticklabels(ticklbls)

# [-1.0,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]

sen_normalize = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
err_normalize = matplotlib.colors.Normalize(vmin=-3.0, vmax=3.0)


chds = ['mean', 'max', 'min','mse']
mm_errs = np.transpose(np.array(mm_errs))

ft1, at1 = plotter(mm_errs, 'multi model errors', lbls=['mean', 'max', 'min', 'mse'] )
ft1 = addnote(ft1)


f3, a3 = mapper(test_map[:,0], test_map[:,1], test_map[:,33], 'coolwarm', \
                'overall error map from test data 1 deg mse', \
                err_normalize)

f2, a2 = mapper(test_map[:,0], test_map[:,1], test_map[:,34], 'coolwarm', \
              'sensitivity map from test data 1 deg C mse', \
                sen_normalize)

f4, a4 = scat(test_map[:,4], test_map[:,34], test_map[:,34], 'RdBu', \
              'sensitivity v elevation', sen_normalize)

f5, a5 = scat(test_map[:,1], test_map[:,34], test_map[:,34], 'RdBu', \
              'sensitivity v latutude', sen_normalize)

if False:
  f6, a6 =  scat(test_map[:,2], test_map[:,32], test_map[:,32], 'RdBu', \
              'sensitivity v surf sol', sen_normalize)
if True:
  f7, a7 =  scat(test_map[:,32], test_map[:,33], 'k', None, \
                 'sensitivity v error', None)

f8, a8 = scat(test_map[:,1], test_map[:,33], test_map[:,33], 'RdBu', \
              'error v latutude', err_normalize)

f9, a9 = scat(test_map[:,4], test_map[:,33], test_map[:,33], 'RdBu', \
              'error v elevation', err_normalize)

plt.show()

  
 # pdb.set_trace()
  
