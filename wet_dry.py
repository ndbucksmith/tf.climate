import numpy as np
import time
import rasterio as rio
import matplotlib
import matplotlib.pyplot as plt
import pickle
# import pdb
import math
import tensorflow as tf
import gt_utils as gtu
import gt_model as gtm
import wc_batcher as wcb
import os
import json

pst = pdb.set_trace


def plotter(plts, name, lbls):
    fi_, ax_ = plt.subplots(1)
    fi_.suptitle(name)
    for px in range(len(plts)):
        ax_.plot(plts[px], label=lbls[px])
    ax_.legend()
    return fi_, ax_


wet = 0.0
dry = 1.0
ir_back_co2_mean = 0.05
ir_back_h20_mean = 1 - ir_back_co2_mean
ir_back_h20_dry = ir_back_h20_mean - (ir_back_co2_mean * 0.3)
ir_back_h20_wet = ir_back_h20_mean + (ir_back_co2_mean * 0.3)
cloud_refl_mean = 0.23
cloud_refl_wet = cloud_refl_mean + (cloud_refl_mean * 0.3)
cloud_refl_dry = cloud_refl_mean - (cloud_refl_mean * 0.3)
vis_wet = 1.0 - ir_back_h20_wet - ir_back_co2_mean
vis_dry = 1.0 - ir_back_h20_dry - ir_back_co2_mean

plotz = [
    [ir_back_h20_dry, ir_back_h20_wet],
    [cloud_refl_dry, cloud_refl_wet],
    [vis_dry, vis_wet],
]
# pst()
fi, ax = plotter(plotz, "wet-dry", ["h20", "cloud", "vis"])
plt.show()
