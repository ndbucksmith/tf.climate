import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu

"""
read geotiff files and generate a batch of examples
"""

feat_list = [
    "long",
    "lati",
    "vis_down",
    "vis_dstd",
    "toa_pwr",
    "elev",
    "elev_std",
    "zels",
    "gtzels",
    "ltzls",
    "sh1h",
    "nh1h",
    "pwr_ratio",
    "barr_press",
]
feat_list_width = len(feat_list)
feat_norms = [
    1.0,
    1.0,
    310.0,
    30.0,
    415.0,
    7000.0,
    1000.0,
    40.0,
    400.0,
    400.0,
    1.0,
    1.0,
    1.0,
    760.0,
]


elds = rio.open("wcdat/ELE.tif")
hids = rio.open("wcdat/GHI.tif")
teds = rio.open("wcdat/TEMP.tif")
_st = time.time()
with open("winstats/summary_cts.pkl", "r") as fi:
    _dc = pickle.load(fi)
testcts = _dc["testcts"]
traincts = _dc["traincts"]
train_sums = []
test_sums = []
assert len(traincts) == len(testcts)
# builds lists to index window
for wx in range(len(traincts)):
    if wx == 0:
        train_sums.append(traincts[wx])
        test_sums.append(testcts[wx])
    else:
        train_sums.append(train_sums[wx - 1] + traincts[wx])
        test_sums.append(test_sums[wx - 1] + testcts[wx])
train_sums = np.array(train_sums)
test_sums = np.array(test_sums)
train_total = traincts.sum()
test_total = testcts.sum()


def get_windpt(ptx, cts):
    wx = 0
    while cts[wx] < ptx:
        wx += 1
    pickpt = ptx - cts[wx]
    return wx, pickpt


def get_example_index(wx, pickpt, bTrain):
    fn = "winstats/book_" + str(wx * 5) + ".pkl"
    # print(fn)
    with open(fn, "r") as fi:
        windict = pickle.load(fi)
    if bTrain:
        wlax, wlox, datact = windict["trix"][pickpt]
    else:
        wlax, wlox, datact = windict["teix"][pickpt]
    windo = rio.windows.Window(wlox, wlax, 20, 20)
    return windo, wlax, wlox, datact


# print(train_sums)
# print(train_total)


def bld_eu_examp(ptix, bTrain):  # an example in eng units
    # print(ptix)
    wx, pickpt = get_windpt(ptix, train_sums)
    # print(wx, pickpt)
    wnd, lax, lox, datct = get_example_index(wx, pickpt, bTrain)
    # print(lax, lox, datct)
    el = elds.read(1, window=wnd)
    te = teds.read(1, window=wnd)
    hi = hids.read(1, window=wnd)
    lat, lon = gtu.llscale(lax, lox, 120, 60.0)
    vis_down = np.nanmean(hi) * 1000.0 / 24
    toa_pwr = gtu.toaPower(lat)
    assert toa_pwr > vis_down
    temp = np.nanmean(te)
    elev = np.nanmean(el)
    elev_std = np.nanstd(el)
    vis_dstd = np.nanstd(hi) * 1000.0 / 24
    gtzs = (el > 0.0).sum()
    ltzs = (el < 0.0).sum()
    zs = (el == 0.0).sum()
    pwr_ratio = vis_down / toa_pwr
    ins = [
        lon,
        lat,
        vis_down,
        vis_dstd,
        toa_pwr,
        elev,
        elev_std,
        zs,
        gtzs,
        ltzs,
        pwr_ratio,
    ]
    # print(gtzs,ltzs, zs)
    # pdb.set_trace()
    if elev < -100:
        batch_good = False
    else:
        batch_good = True
    if lat > 0:
        ins.append(0.0)
        ins.append(1.0)
    else:
        ins.append(1.0)
        ins.append(0.0)
    return np.array(ins), temp, batch_good


def get_batch(size, bTrain):
    ins_bat = []
    trus_bat = []
    for ix in range(size):
        b_g = False
        while not b_g:
            ptix = np.random.randint(0, train_total)
            ins, temp, b_g = bld_eu_examp(ptix, bTrain)
        ins_bat.append(ins)
        trus_bat.append(temp)

    return np.array(ins_bat), np.array(trus_bat)
