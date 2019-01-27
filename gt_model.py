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

"""
tensorflow models to predict temperature as a function of solar power, toa power, elevation
 and eventually many other parameters

"""



class ClimaModel:
  def __init__(self, params, bTrain=True):
    x_size = params['x_size']
    y_size = 1
    f_width = params['f_width']
    istd = params['init_stddev']
    self.activation = tf.nn.relu  #tf.tanh  #tf.nn.relu #tf.sigmoid
    self.filters = {}
    self.batch_size = params['batch_size']
    self.params = params
    self.sess = tf.Session()
    self.x = tf.placeholder(tf.float32, (None, x_size), name='x')
    self.norms = tf.placeholder(tf.float32, (x_size), name='x_norms')
    self.xn = tf.divide(self.x, self.norms)
    self.y_true = tf.placeholder(tf.float32, (None), name='y_true')
    self.xf1 = self.filter(self.xn, f_width=f_width, name='xf1')
    wy1 = tf.get_variable('wy', None, tf.float32, tf.random_normal([f_width, y_size], stddev=istd))
    by1 = tf.get_variable('by', None, tf.float32, tf.random_normal([y_size], stddev=0.01))
    self.h = tf.add(tf.matmul(self.xf1, wy1), by1)
    meantemp = tf.fill(tf.shape(self.y_true), 14.6)
    lossfactor = tf.fill(tf.shape(self.y_true), 250.0)
    min_wt = tf.fill(tf.shape(self.y_true), 0.1)
    #pdb.set_trace()
    self.losswts = tf.add(tf.divide(tf.square(self.y_true - meantemp), lossfactor), min_wt)
    self.loss = tf.square(self.y_true - self.h)
    self.wtd_loss = tf.multiply(self.loss, self.losswts)
    self.losser = tf.reduce_mean(self.loss)
    self.wtd_losser =  tf.reduce_mean(self.wtd_loss)
    if bTrain:
      global_step = tf.Variable(0, trainable=False)
      start_learnrate = params['learn_rate']                          
      self.learning_rate = tf.train.exponential_decay(start_learnrate, global_step, 300000, 0.9, staircase=True)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.ts = optimizer.minimize(self.wtd_loss, global_step=global_step)
    else:
      self.ts = None
    #pdb.set_trace()
    vars_to_save=(v.name for v in tf.trainable_variables())
    self.saver=tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)
    self.sess.run(tf.global_variables_initializer())    
                            
  def losser_val(es, ts):
    squs = []
    for ix in range(len(es)):
      squs.append((es[ix] -ts[ix])**2)
    squs = np.array(squs)
    return squs, squs.mean()

  def reup(self):
    self.sess.run(tf.global_variables_initializer())      
                  
    
  def filter(self, x, f_width=-1, y_width=-1, residual=False, name=''):
    if name in self.filters:
      var_reuse = True
    else:
      var_reuse = False
      self.filters[name] = name
    istd = self.params['init_stddev']
    with tf.variable_scope(name+'filter', reuse=var_reuse):
      in_size = int(x.get_shape()[-1])
      if f_width == -1:
          f_width = int(x.get_shape()[-1])
          
      w1 = tf.get_variable(name+'_filter_w1', None, tf.float32, tf.random_normal([in_size, f_width], stddev=istd))
      b1 = tf.get_variable(name+'_filter_b1', None, tf.float32, tf.random_normal([f_width], stddev=0.01))
      f1 = tf.add(b1, tf.matmul(x, w1))
      out1 = self.activation(f1, name=name+'out1')
                           
      w2 = tf.get_variable(name+'_filter_w2', None, tf.float32, tf.random_normal([f_width, f_width], stddev=istd))
      b2 = tf.get_variable(name+'_filter_b2', None, tf.float32, tf.random_normal([f_width], stddev=0.01))
      f2 = tf.add(b2, tf.matmul(out1, w2))
      out2 = self.activation(f2, name=name+'out2')

      if residual == False:
        f_output = out2
      else:      
        wr1 = tf.get_variable(name+'_filter_wr1', None, tf.float32, tf.random_normal([f_width, f_width], stddev=istd))
        br1 = tf.get_variable(name+'_filter_br1', None, tf.float32, tf.random_normal([f_width], stddev=0.01))
        fr1 = tf.add(br1, tf.matmul(out2, wr1))
        outr1 = self.activation(f1, name=name+'outr1')
                           
        wr2 = tf.get_variable(name+'_filter_wr2', None, tf.float32, tf.random_normal([f_width, f_width], stddev=istd))
        br2 = tf.get_variable(name+'_filter_br2', None, tf.float32, tf.random_normal([f_width], stddev=0.01))
        fr2 = tf.add(br2, tf.matmul(outr1, wr2))
        outr2 = self.activation(fr2, name=name+'out2')
        f_output = tf.add(out2, outr2)

    return f_output


  def bld_feed(self, Ins, Trues):
    fd = {}
    fd[self.x] = np.take(Ins, self.params['take'], axis=1)
    fd[self.norms] = np.take(gtb.feat_norms, self.params['take'], axis=0)       
    #pdb.set_trace()
    assert self.params['x_size'] == fd[self.x].shape[-1]
    fd[self.y_true] = Trues
    return fd

  def save(self, path, tx):
    save_path = self.saver(self.sess, path+str(tx)+'ckpt', write_meta_graph=False)

  def restore(self, path):
    self.save.restore(self.sess, path)

class stupidModel():
  def __init__(self, sess, params, bTrain=True):
    self.batch_size = params['batch_size']
    self.params = params
    self.sess = sess
    self.pwr = tf.placeholder(tf.float32, (None), name='pwr')
    self.elev = tf.placeholder(tf.float32, (None), name='elev')
    self.toap  = tf.placeholder(tf.float32, (None), name='toap_ratio')
    self.y_true = tf.placeholder(tf.float32, (None), name='y_true')
    psens = tf.get_variable('psens', None, tf.float32,tf.random_normal([1],mean=0.205, stddev=0.00001))
    esens = tf.get_variable('esens', None, tf.float32,tf.random_normal([1],mean=-0.0064, stddev=0.0000001))
    toasens = tf.get_variable('toasens', None, tf.float32,tf.random_normal([1],mean=0.005, stddev=0.0000001))
    smb =  tf.get_variable('smbias', None, tf.float32,tf.random_normal([1],mean=-23.6, stddev=0.01))
    self.hp = tf.add(tf.multiply(self.pwr, psens),smb)
    self.tp = tf.multiply(self.toap, toasens)
    self.htp = tf.add(self.hp, self.tp)
    self.h = tf.add(tf.multiply(self.elev, esens),self.htp)
    self.loss = tf.square(self.y_true - self.h)
    self.losser = tf.reduce_mean(self.loss)
    self.psens = psens; self.esens = esens; self.toasens = toasens; self.smb = smb;
    if bTrain:
      global_step = tf.Variable(0, trainable=False)
      start_learnrate = 0.005 #params['learn_rate']                          
      self.learning_rate = tf.train.exponential_decay(start_learnrate, global_step, 300000, 0.9, staircase=True)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.ts = optimizer.minimize(self.loss, global_step=global_step)
    else:
      self.ts = None
    sess.run(tf.global_variables_initializer())

   
    
  def bld_feed(self, Ins, Trues):
    fd = {}
    fd[self.pwr] = Ins[:,2]
    fd[self.elev] = Ins[:,4]
    fd[self.toap] = Ins[:,4]/Ins[:,5]
    fd[self.y_true] = Trues
    return fd