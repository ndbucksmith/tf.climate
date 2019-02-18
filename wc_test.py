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
from gt_utils import mapper,tabler, plotter, scat, addnote
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
mdl_path = 'mdls/nn3031cs60walb'
if True:
  with open(mdl_path +'/params.json', 'r') as fi:
    params = json.load(fi)
take = params['take']
b_size = params['batch_size']
x_size = params['rxin_size']
params['file_ct'] = file_ct
params['mdl_path'] = mdl_path

imap_lbls = []
pstr = "channels:: "
for idx in take:
  pstr += wcb.nn_features[idx]
  pstr += ', '
  imap_lbls.append(wcb.nn_features[idx])
for ix in params['rnn_take']:
  imap_lbls.append(wcb.rnn_features[ix]) 
print(pstr)
params['ch_labels'] = imap_lbls

wc_start = len(params['take'])
gs_radx = take.index(2)  #index used for wiggle jiggle sensitivity
wc_radx = take.index(8)
alb_ix =  wc_start-1

assert wcb.nn_features[take[gs_radx]]  == 'gsra'
assert wcb.nn_features[take[wc_radx]]  == 'sra_'
assert wcb.nn_features[take[alb_ix]]  == 'alb'


infl_map_hh = []; infl_map_h =[ ];
infl_map_cc = []; infl_map_c = [];
infl_ct_hh = [0]*x_size; infl_ct_h = [0]*x_size;
infl_ct_cc = [0]*x_size; infl_ct_c = [0]*x_size;
def influence_map(infl_sort, grads):
  mm_ix =[0, 1, 2, x_size-3, x_size-2, x_size-1]
  for bx in range(b_size):
    for mx in range(12):
      for ix in mm_ix:
       gradix = infl_sort[bx,mx,ix]
       if ix == 0:
         infl_ct_cc[gradix] += 1
       elif ix==2 or ix == 1:
         infl_ct_c[gradix] += 1
       elif ix ==  x_size-1:
         infl_ct_hh[gradix] += 1
       elif ix == x_size - 2: 
         infl_ct_h[gradix] += 1

def examp_plot(bx, name):
  f_ex, a_ex = plt.subplots(4,1)
  f_ex.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99)
  colhs_ = ['lon','lat', 'RNN Temps', 'wcTempTru', 'gsTempTru','err']
  tabl_  = [[round(ins[bx,0],1), round(ins[bx,1],1), round(meta_ests[bx],1), round(meta_yt[bx],1), round(trus[bx],1), round(meta_errs[bx],3)],
             np.ma.round(ins[bx,2:8],2),np.ma.round(ins[bx,8:14],2), np.ma.round(ins[bx,14:20],2) ]
  a_ex[0].table(cellText=tabl_, colLabels=colhs_, loc='center')
  a_ex[1].plot(rsqs[bx,0,:])
  a_ex[2].plot(rsqs[bx,2,:])
  a_ex[3].plot(rsqs[bx,1,:])
  return f_ex, a_ex

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
mm_errs = [];
for mcx in range(1):

  for tx in range(file_ct):

    if False:
      ins, trus = gtb.get_batch(params['batch_size'], True)
    else:
      with open('wc_v2test/wcb_' + str(tx) + '.pkl', 'r') as fi:
        dc = pickle.load(fi)
        ins = dc['ins']
        app = []
        rsqs =  np.array(dc['rnn_seqs'])
        wc_trus = dc['ec_tru']  #alternative verion of reality, man
        rn_trus =  dc['rnn_trus']
        trus = dc['trus']

#_______feed, fetch, and run
    feed = rmdl.bld_multiyearfeed(1, ins, rsqs, rn_trus, wc_trus)
    fetch = [rmdl.lossers, rmdl.hypos,  rmdl.y_trues, rmdl.meta_h, rmdl.meta_loss, \
                rmdl.meta_yt, rmdl.do_di, rmdl.do_din]     
    sq_errs, ests, yt, meta_ests, meta_sqerrs, meta_yt, dodis, do_dins   = sess.run(fetch, feed)

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
    mme.append(0.5); mme.append(1.0);
    mm_errs.append(mme)
    do_dins = np.array(do_dins[0]) 
    dodis = np.array(dodis[0])
#______________build influencer map
    influence_sort = np.argsort(do_dins, axis=2)
    influence_map(influence_sort, do_dins)
  
    surfPower_grad = (dodis[:,:,gs_radx].sum(axis=1) + (75.0 * dodis[:,:,wc_start].sum(axis=1))  \
                       + (75.0 *  dodis[:,:,wc_radx].sum(axis=1)))
   
  
 #   f90, ax0 = plotter([dodis[0,:,gs_radx],dodis[0,:,wc_radx],dodis[0,:,wc_start]], name='sensi' + str(tx),  \
 #                        lbls=['vis','srad','12mosr'])
 

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

    #add some wiggle and check the jiggle.  
    watts = 3.7  # add to global solar srad as watts and wc 12 month and monthly in their wacky units
    feed[rmdl.xin][:,:,gs_radx] = feed[rmdl.xin][:,:,gs_radx] + watts
    feed[rmdl.xin][:,:,wc_radx] = feed[rmdl.xin][:,:,wc_radx] +  (watts*75.0)
    feed[rmdl.xin][:,:,wc_start] = feed[rmdl.xin][:,:,wc_start] + (watts*75.0)
    # fetch temperature estimates with extra power added in from tf session run
    d_sq_errs, d_ests, d_yt, dmeta_h, dmeta_err, d_meta_yt,d_dodis, d_dodins = sess.run(fetch, feed)
    # calculate sensitivity tosurface  solar power
    dTdP = (d_ests - ests)/watts  # in unit degreeC per watt per meter square
    meta_dTdP = (dmeta_h - meta_ests)/watts

    meta_dTdP = meta_dTdP /  (1-np.reshape(feed[rmdl.xin][:,0,alb_ix], (400,1)))
    print('power sensitivity - mean, max, min, stdev, count negative')
    gtu.arprint([dTdP.mean(), dTdP.max(), dTdP.min(), dTdP.std(), (dTdP < 0.0).sum()])
    gtu.arprint([meta_dTdP.mean(), meta_dTdP.max(), meta_dTdP.min(), meta_dTdP.std(), (meta_dTdP < 0.0).sum()])
    gtu.arprint([surfPower_grad.mean(),surfPower_grad.max(),surfPower_grad.min(),surfPower_grad.std(),(surfPower_grad < 0.0).sum()])

# ____________ make the test map    
    for bx in range(b_size):
      test_map.append(np.concatenate( (ins[bx,0:7],errs[:,bx,0],overall_errs[bx], \
                                       dTdP[:,bx,0], [dTdP[:,bx,0].mean()],meta_errs[bx],  \
                                       meta_dTdP[bx],meta_yt[bx], meta_ests[bx], [bx], [tx]) ))


  #  f_expH, a_exHp = examp_plot(meta_errs.argmax(), 'model hot error')
  #  f_expC, a_expC = examp_plot(meta_errs.argmin(), 'model cold error')
  #  bgx = 0; # find example with good accuracy
  #  while (meta_errs[bgx] > 0.2) or (meta_errs[bgx] < -0.2):
  #    bgx+=1
  #  f_exp, a_exp = examp_plot(bgx, 'model pretty good')  
 #   plt.show()
    if False:  # graph to compare manaul sens calc and tf.grads
      f1sens, a1sens  =  scat(meta_dTdP, surfPower_grad ,'k' , None, \
                              'Manual Sens calc with  delta=' + str(watts) + ' watts vs. tf.gradients', None)
      a1sens.xaxis.set_label_text('manual')
      a1sens.yaxis.set_label_text('tf.gradient')
      f1sens.text(0.2, 0.8, 'units are degree C/wm2', transform=plt.gcf().transFigure)
      plt.show()


test_map = np.array(test_map)
print('test map dataset shape, max and min sensitivty, count negative')
print(test_map.shape, test_map[:,34].max(), test_map[:,34].min(), (test_map[:,34] < 0.0).sum())
sen_histo = np.histogram(test_map[:,34], bins=20)
print('histo')
print(zip(sen_histo[0], sen_histo[1][0:-1]))


print('influence count  dT/dfeat ')
print(pstr)
print(infl_ct_cc)
print(infl_ct_c)
print(infl_ct_h)
print(infl_ct_hh)


fig_inf_ct, axe_inf_ct = plt.subplots(2,1)
fig_inf_ct.suptitle('tf.climate meta-model number 1 and 2 gradients, hot and cold count by normalized feature')
cowi = 0.4
axe_inf_ct[0].bar(range(x_size), infl_ct_cc, color='b', align='edge', width=cowi)
axe_inf_ct[0].bar(range(x_size), infl_ct_hh, color='r', align='edge', width=-cowi)
axe_inf_ct[1].bar(range(x_size), infl_ct_c, color='b', align='edge', width=cowi)
axe_inf_ct[1].bar(range(x_size), infl_ct_h, color='r', align='edge', width=-cowi)
axe_inf_ct[1].xaxis.set_ticks(range(x_size))
axe_inf_ct[1].xaxis.set_ticklabels(imap_lbls)
axe_inf_ct[0].xaxis.set_ticks(range(x_size))
axe_inf_ct[0].xaxis.set_ticklabels(imap_lbls)
fig_inf_ct = addnote(fig_inf_ct, params)

fig21, axe21 = plt.subplots(1)
fig21.suptitle('albedo corrected sensitivity distribution')
pst()
axe21.bar(range(len(sen_histo[0])), sen_histo[0], align='center', )
ticklbls = []
for ix in range(len(sen_histo[1][0:-1])):
  ticklbls.append( str(round(sen_histo[1][ix+1], 3)))
axe21.xaxis.set_ticks(range(len(ticklbls)))
axe21.xaxis.set_ticklabels(ticklbls)
fig21 = addnote(fig21, params)


sen_normalize = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3)
err_normalize = matplotlib.colors.Normalize(vmin=-3.0, vmax=3.0)


chds = ['mean', 'max', 'min','mse','0.5','1.0']
mm_errs = np.transpose(np.array(mm_errs))

ft1, at1 = plotter(mm_errs, 'dual train RNN+NN meta model errors', lbls=chds )
ft1 = addnote(ft1, params)


f3, a3 = mapper(test_map[:,0], test_map[:,1], test_map[:,33], 'coolwarm', \
                'overall error map from test data w 0.5 deg C mse', \
                err_normalize)
f3 = addnote(f3,params)

f2, a2 = mapper(test_map[:,0], test_map[:,1], test_map[:,34], 'coolwarm', \
              'albedo corrected sensitivity map', \
                sen_normalize)
f2 = addnote(f2, params)

if True:
  f4, a4 = scat(test_map[:,4], test_map[:,34], 'k', None, \
                'sensitivity v elevation', None)

  f5, a5 = scat(test_map[:,1], test_map[:,34],'k', None, \
                'sensitivity v latutude', None)

if False:
  f6, a6 =  scat(test_map[:,2], test_map[:,32], test_map[:,32], 'RdBu', \
              'sensitivity v surf sol', sen_normalize)
if True:
  f7, a7 =  scat(test_map[:,32], test_map[:,33], 'k', None, \
                 'sensitivity v error', None)

if False:
  f8, a8 = scat(test_map[:,1], test_map[:,33], test_map[:,33], 'RdBu', \
              'error v latutude', err_normalize)

  f9, a9 = scat(test_map[:,4], test_map[:,33], test_map[:,33], 'RdBu', \
              'error v elevation', err_normalize)

ffreeze, axfreeze = scat(test_map[:,35], test_map[:,34], test_map[:,34], 'RdBu', \
              'sensitivity v Trutemp', sen_normalize)

fyt, axtt = scat(test_map[:,35], test_map[:,36], 'k', None, \
                 'model est v Trutemp', None)

plt.show()

  
 # pdb.set_trace()
  
