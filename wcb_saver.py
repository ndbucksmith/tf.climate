import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu
import wc_batcher as wcb
import os

"""
Saves pickled batches using wc batcher



copyright 2019 Nelson 'Buck' Smith
"""
target = "wc_v3"
start_file = len(os.listdir(target))
bTrain = True

for tx in range(start_file, 3000):
    start_t = time.time()
    ins, rnn_seqs, wc_trues, rnn_trus, d3_idx = wcb.get_batch(400, bTrain)
    dc = {}
    dc["bTrain"] = bTrain
    dc["ins"] = ins
    dc["d3_idx"] = d3_idx
    dc["rnn_seqs"] = rnn_seqs
    dc["ec_tru"] = wc_trues  # alternative verion of reality, man
    dc["rnn_trus"] = rnn_trus
    # print('fx',tx)
    with open(target + "/wcb_" + str(tx) + ".pkl", "w") as fo:
        dmp = pickle.dumps(dc)
        fo.write(dmp)
