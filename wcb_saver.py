import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu
import wc_batcher as wtb

"""
Saves pickled batches using wc batcher



copyright 2019 Nelson 'Buck' Smith
"""


for tx in range(16, 1000):
  start_t = time.time()
  ins, trus, rnn_seqs, wc_trues, rnn_trus = wtb.get_batch(400, True)
  dc = {}
  dc['ins'] = ins
  dc['trus'] = trus
  dc['rnn_seqs'] = rnn_seqs
  dc['ec_tru'] = wc_trues #alternative verion of reality, man
  dc['rnn_trus'] = rnn_trus
  #print('fx',tx)
  with open('wc_bs/wcb_' + str(tx) + '.pkl', 'w') as fo:
    dmp = pickle.dumps(dc)
    fo.write(dmp)

