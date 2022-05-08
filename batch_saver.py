import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
# import pdb
import math


import gt_utils as gtu

import gt_batcher as gtb

float_formatter = lambda x: "%.1f" % x
np.set_printoptions(formatter={"float_kind": float_formatter})

for tx in range(14, 1000):
    start_t = time.time()
    ins, trus = gtb.get_batch(400, True)
    dc = {}
    dc["ins"] = ins
    dc["trus"] = trus
    with open("batches/b_" + str(tx) + ".pkl", "w") as fo:
        dmp = pickle.dumps(dc)
        fo.write(dmp)
        print(ins[0])
        print(trus[0])
