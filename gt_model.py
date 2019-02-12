import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import tensorflow as tf
import gt_utils as gtu
import gt_batcher as gtb
import wc_batcher as wcb
from tensorflow.keras import models
from tensorflow.keras import layers
pst = pdb.set_trace

"""
tensorflow models to predict temperature as a function of solar power, toa power, elevation
 and eventually many other parameters


copyright 2019 Nelson 'Buck' Smith
"""
metaTrain = True

class artisanalModel():
  def __init__(self,sess, params, bTrain=True):
    self.batch_size = params['batch_size']
    self.params = params
    self.sess =  sess
    self.pwr = tf.placeholder(tf.float32, (None), name='pwr')
    self.elev = tf.placeholder(tf.float32, (None), name='elev')
    self.toap  = tf.placeholder(tf.float32, (None), name='toap_ratio')
    self.bp  = tf.placeholder(tf.float32, (None), name='baro')
    self.smy_true = tf.placeholder(tf.float32, (None), name='smy_true')
    psens = tf.get_variable('psens', None, tf.float32,tf.random_normal([1],mean=0.25, stddev=0.00001))
    esens = tf.get_variable('esens', None, tf.float32,tf.random_normal([1],mean=-0.0064, stddev=0.0000001))
    toasens = tf.get_variable('toasens', None, tf.float32,tf.random_normal([1],mean=-0.2, stddev=0.0000001))
    smb =  tf.get_variable('smbias', None, tf.float32,tf.random_normal([1],mean=-23.6, stddev=0.01))
    bpsens =  tf.get_variable('bpsens', None, tf.float32,tf.random_normal([1],mean=0.00, stddev=0.01))
    self.hp = tf.add(tf.multiply(self.pwr, psens),smb)
    self.ratio = tf.divide(self.pwr, self.toap)
    self.tp = tf.multiply(self.ratio, toasens)
    self.htp = tf.add(self.hp, self.tp)
    self.hbp = tf.multiply(self.bp, bpsens)
    self.he =  tf.multiply(self.elev, esens)
    self.he_hb =tf.add(self.he, self.hbp)
    self.arth = tf.add(self.he_hb,self.htp)
    self.loss = tf.square(self.smy_true - self.arth)
    self.losser = tf.reduce_mean(self.loss)
    self.psens = psens; self.esens = esens; self.toasens = toasens; self.smb = smb; self.bpsens=bpsens;
    if bTrain:
      global_step = tf.Variable(0, trainable=False)
      start_learnrate = 0.3 # params['learn_rate']                          
      self.learning_rate = tf.train.exponential_decay(start_learnrate, global_step, 300000, 0.9, staircase=True)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.ts = optimizer.minimize(self.loss, global_step=global_step)
    else:
      self.ts = None
   # self.sess.run(tf.global_variables_initializer())

   
    
  def bld_feed(self, Ins, Trues):
    fd = {}
    fd[self.pwr] = Ins[:,2]
    fd[self.elev] = Ins[:,5]
    fd[self.toap] = Ins[:,2]/Ins[:,4]
    fd[self.bp] = Ins[:,13]
    fd[self.smy_true] = Trues
    return fd

class climaRNN():
  def __init__(self, _years, sess, params, bTrain=True):
    cell_size = params['cell_size']; self.cell_size = cell_size 
    xin_size = params['rxin_size']; self.xin_size = xin_size
    b_size = params['batch_size']
    y_size = 1
    self.params = params
    self.sess = sess
  #  self.cell_fw =  tf.contrib.rnn.GRUBlockCellV2(num_units=self.cell_size, name='fwd_cell') 
 #   self.cell_bw =  tf.contrib.rnn.GRUBlockCellV2(num_units=self.cell_size, name='bwd_cell')
    self.fw_init =  tf.get_variable('fwd_init', None, tf.float32, tf.zeros(cell_size))
    self.bw_init =  tf.get_variable('bwd_init', None, tf.float32, tf.zeros(cell_size))
    rnn_init_fwd = []; rnn_init_bwd = [];
    for ix in range(b_size):
      rnn_init_fwd.append(self.fw_init)
      rnn_init_bwd.append(self.bw_init)
#  tf.Tensor &apos;GRUCellZeroState/zeros:0&apos; shape=(400, 64) dtype=float32 
    rnn_init_fwd = tf.stack(rnn_init_fwd)
    rnn_init_bwd = tf.stack(rnn_init_bwd)
    self.cell_fw =  tf.nn.rnn_cell.GRUCell(num_units=cell_size, name='fwd_cell') 
    self.cell_bw =  tf.nn.rnn_cell.GRUCell(num_units=cell_size, name='bwd_cell')  
    self.xin = tf.placeholder(tf.float32, (None, 12*_years,  xin_size), name='rnn_xin')
    self.norms =  tf.placeholder(tf.float32, (xin_size), name='rnx_norms')
    self.xnorms = tf.divide(self.xin, self.norms)
    rnn_w1, rnn_b1 = weightSet('rnn_l1', [xin_size, params['pref_width']])
    rnn_l1 = []
    for mx in range(12):
      rnn_l1.append(tf.nn.tanh(tf.add(tf.matmul(self.xnorms[:,mx], rnn_w1), rnn_b1)))
    self.rnn_l1 = tf.transpose(tf.stack(rnn_l1), perm=[1,0,2])
    (fwouts, bwouts),(fwstate, bwstate) = tf.nn.bidirectional_dynamic_rnn( \
                  self.cell_fw,
                  self.cell_bw,
                  self.rnn_l1,
                  initial_state_fw=rnn_init_fwd,
                  initial_state_bw=rnn_init_bwd,                                                     
                  dtype=tf.float32)
    #default is both states (fw and bw) init to zeros
 
    rnn_wy = tf.get_variable('rnn_wy', None, tf.float32, tf.random_normal(
                               [(cell_size*2) + xin_size, y_size],\
                               stddev=0.01))
    rnn_by = tf.get_variable('rnn_by', None, tf.float32, tf.zeros(y_size))
    hypos = []; losses = []; y_trues = []; lossers = []; h_norms = []
    self.temp_norms = tf.constant(40.0, dtype=tf.float32, shape=[1])
    outs = tf.concat((fwouts, bwouts), axis=2)
    for ix in range(12*_years):
      y_trues.append(tf.placeholder(tf.float32, (None,  y_size), name='rnn_yt'+str(ix)))
      hypos.append(tf.add(rnn_by, tf.matmul(  \
                          tf.concat((outs[:,ix,:], self.xnorms[:,ix,:]), axis=1), \
                          rnn_wy), name='h_'+str(ix)))
      h_norms.append(tf.divide(hypos[ix], self.temp_norms))
      losses.append(tf.square(y_trues[ix] - hypos[ix], name='losses'))
      lossers.append(tf.reduce_mean(losses[ix], name='lossers'))
    self.hypos = hypos; self.losses = losses; self.lossers = lossers;
    self.y_trues = y_trues
    loss = tf.add_n(losses); self.loss = loss;
    losser = tf.add_n(lossers); self.losser = losser;
    if bTrain:
      global_step = tf.Variable(0, trainable=False)
      start_learnrate = params['learn_rate']                          
      self.learning_rate = tf.train.exponential_decay(start_learnrate, global_step, 300000, 0.9, staircase=True)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.rnn_varlist = tf.trainable_variables()
      self.ts = optimizer.minimize(tf.stack(losses), var_list=self.rnn_varlist, global_step=global_step)
    else:
      self.ts = None;
    if metaTrain:
      meta_in = tf.concat((tf.transpose(tf.stack(h_norms))[0], self.xnorms[:,ix,0:self.xin_size]), axis=1)
     # pst()
      meta_w1, meta_b1 = weightSet('meta_l1', [12+self.xin_size, params['metaf_width']])
      meta_o1 = tf.nn.tanh(tf.add(tf.matmul(meta_in, meta_w1), meta_b1))
      meta_w2, meta_b2 = weightSet('meta_l2', [params['metaf_width'], 1])
      self.meta_h  = tf.add(meta_b2,  tf.matmul(meta_o1, meta_w2))
      self.meta_yt = tf.placeholder(tf.float32, (None,  y_size), name='meta_yt')
      self.meta_loss = tf.square(self.meta_h-self.meta_yt, name='meta_loss')
      self.meta_losser = tf.reduce_mean(self.meta_loss, name='meta_losser')
      self.do_din = tf.gradients(self.meta_h, self.xnorms, name='gradients_io_norm')
      self.do_di = tf.gradients(self.meta_h, self.xin, name='gradients_io')
    if metaTrain and bTrain:
      meta_globstep = tf.Variable(0, trainable=False)
      meta_learn =  tf.train.exponential_decay(start_learnrate, meta_globstep, 300000, 0.9, staircase=True)
      meta_optim = tf.train.AdamOptimizer(meta_learn)
      self.meta_ts  = meta_optim.minimize(self.meta_loss, var_list=[meta_w1, meta_b1, meta_w2, meta_b2], \
                                          global_step=meta_globstep)

                       
    #pdb.set_trace()
    self.vars_to_save=(v.name for v in tf.trainable_variables())
    self.vl = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.sv1 = tf.train.Saver()
 

#deprecated
  def bld_feed(self, ins, rins, rn_trus):
    fd ={}
    static_ins = ins[:,1:]
    static_norms = wcb.nn_norms[1:]
    static_12mo = []
    rn_trus = np.reshape(np.array(rn_trus), (-1,12,1))
    #pdb.set_trace()
    for mx in range(12):
      static_12mo.append(static_ins)
      fd[self.y_trues[mx]] = rn_trus[:,mx]

    rins_swap = np.swapaxes(rins, 1, 2)
    stat_swap = np.swapaxes(static_12mo, 0, 1)
    full_rinset = np.concatenate((stat_swap, rins_swap), axis=2)
    full_normset = np.concatenate((static_norms, wcb.rnn_norms))
    fd[self.xin] = full_rinset
    fd[self.norms] = full_normset
    #pdb.set_trace()
    return fd

  def bld_multiyearfeed(self, yrs, ins, rins, rn_trus, me_trus):
    fd ={}
    #static_ins = ins[:,1:] 
    static_ins =np.take(ins, self.params['take'], axis=1)  #shape batch, 20 odd
    static_norms = np.take(wcb.nn_norms, self.params['take'])  #shape 20 od
    static_multiyr = []
    rn_trus = np.reshape(np.array(rn_trus), (-1,12,1))
    #pdb.set_trace()
    for mx in range(12*yrs):
      static_multiyr.append(static_ins)  #shape mo, batch, 20 odd
      fd[self.y_trues[mx]] = rn_trus[:,mx % 12]
   # rins are  batch, feature, month  swap to batch, month, feature 
    rins_swap = np.swapaxes(rins, 1, 2)
    rins_multiyr = rins_swap
    for yx in range(yrs-1):
        rins_multiyr = np.concatenate((rins_multiyr, rins_swap), axis=1)
    static_multiyr = np.array(static_multiyr)
    stat_swap = np.swapaxes(static_multiyr, 0, 1) # swap to bat,mo,feat
    full_rinset = np.concatenate((stat_swap, rins_multiyr), axis=2)
    full_normset = np.concatenate((static_norms, wcb.rnn_norms))
    fd[self.xin] = full_rinset
    fd[self.norms] = full_normset
    fd[self.meta_yt] = np.reshape(np.array(me_trus), (-1,1))
    #pdb.set_trace()
    return fd


  def save(self, path, tx):
    #pdb.set_trace()
    tvar = tf.trainable_variables()
    tvar_vals = self.sess.run(tvar)
    for var, val in zip(tvar, tvar_vals):
      print(var.name, var.shape)
    print(val)
    ses = self.sess
    save_path = self.sv1.save(ses, path+str(tx)+'.ckpt', )

  def restore(self, path):
   # sv2 = tf.train.import_meta_graph(path + '.meta')
    vl = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) #
    ses = self.sess
    self.sv1.restore(ses, path)
    tvar = tf.trainable_variables()
    tvar_vals = self.sess.run(tvar)
    for var, val in zip(tvar, tvar_vals):
      print(var.name, var.shape)
    print(val)

def weightSet(name, shape):
  wt = tf.get_variable(name+'_wt', None, tf.float32, tf.random_normal(shape, stddev=0.01))
  bi = tf.get_variable(name+'_bias', None, tf.float32, tf.zeros(shape[-1]))
  return wt, bi

class  climaLayer():
  def __init__(self, name, x, width):
    
    in_size = x.shape[-1].value
    lay_wt = tf.get_variable('wt_' + name, [in_size, width], tf.float32,  \
                              tf.random_normal([in_size,width], stddev=0.01))
    lay_bi = tf.get_variable('bi_'+name, None, tf.float32, tf.zeros(width))
    lay_op =  (x * W) + b
    lay_act = tf.relu(lay_op, name='lay_out_'+name)
    return lay_out

    
"""
print(fwouts)
Tensor("bidirectional_rnn/fw/fw/transpose_1:0", shape=(?, 12, 32), dtype=float32)

print(bwouts)
Tensor("ReverseV2:0", shape=(?, 12, 32), dtype=float32)

print(hypos)
[<tf.Tensor 'h_0:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_1:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_2:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_3:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_4:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_5:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_6:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_7:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_8:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_9:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_10:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'h_11:0' shape=(?, 1) dtype=float32>]


"""
