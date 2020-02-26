import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu
import os

pst = pdb.set_trace

"""
read geotiff files and generate a batch of examples

gtiff profile data
dataset name, feature, dtype, no data
wc precipitation int16  no data  -32768
wc temp tavg     float32  nodat -3.4e+38
wc srad          uint16  nodata 65536
wc wind         float32  nodata -3.4e+38
glo sol hi      float64  nodata nan
glo temp     float64  nodata nan
glo elevation   int16  nodata -32768
lc   land       unit8   no data 256
lc legend 0 nodata? 10 to 200 land, 210 water, 220 snow ice, 255 nodata?
dem3  version3 elevation int16 nodat: None

Version 2 batcher corrects errors in prec scaling and bad data filters 
and adds annual wind value to nn_features

Version 3 batcher corrects errors in err logs, and index into random examples 
and adds desert as a land type and slopes of the example land  to nn_feature
Also add vapr pressure to rnn features

copyright 2019 Nelson 'Buck' Smith

Annual feature  names entered in three rows 
row1 ct 9 slice 0:9, 
row2 ct 5 slice 9:14, 
row3 ct 9 slice 14:23,
Total annual features = 9+5+9 = 23
Feature names capped at 4 characters to make charts display legibly.
Here is more detail description of the features
lon,  lat  coordinates of sample tile
toa_  top of atmosphere annual average radiation in Trenberth units (24/7 w/m2)
elev - average elevation in the tile
barp - barometric pressure mmHg a pure function of elev
s/t - annual surface solar as a fraction of toa. This number goes lowers as cloud refelction and greenhouse absoption increase
pre_ annual average precipitation from 12 monthly in mm
sra_ annual surface solar from 12 monthly in kJ/m2 day
pre_ annual average wind from 12 monthly
land, wat, ice, dsrt  - Fraction of tile that land, water, ice, bare
alb albedo calculated from previous 4 fields
sh, nh identifies hemisphere
rast - std dev of surface radiation in tile
elst - std dev of elevation in the tile
sslp - north south slope of tile with sun facing positive, pole facing negative
eslp - eastern slope of tile
zs, gtzs, ltzs - fraction of land at sea level and below and above

Monthly features are
suraface visble rad in kJm-2/day
precipitation in mm
toa visble flux in watts / m-2
wind  = km/hr?
vapor rpessure in psi?  



"""
eZtarget = "epochZ"
wc_radCon = 86.4  # kJm-2/day > watts m-2
nn_features = [
    "lon",
    "lat",
    "toa_",
    "elev",
    "barp",
    "s/t_",
    "pre_",
    "sra_",
    "win_",
    "land",
    "wat",
    "ice",
    "dsrt",
    "alb",
    "sh",
    "nh",
    "rast",
    "elst",
    "sun",
    "cori",
    "zs",
    "gtzs",
    "ltzs",
]
nn_feat_len = len(nn_features)
nn_norms = [
    180.0,
    90.0,
    415.0,
    7000.0,
    760.0,
    wc_radCon,
    600.0,
    25000.0,
    11.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2000.0,
    4000.0,
    150.0,
    150.0,
    400.0,
    400.0,
    400.0,
]
assert len(nn_norms) == nn_feat_len
rnn_features = ["srad", "prec", "toa", "wind", "scra", "s/t", "vap"]
rnn_norms = [32768.0, 2400.0, 500.0, 32768.0, 1.0, 11.0, 5]
rnn_feat_len = 7

# to build wc data file names
def str2(mx):
    if mx < 9:
        return str(0) + str(mx + 1)
    else:
        return str(mx + 1)


# record bad data errors during batching
def blog(msg, lat, lon, val=0.001):
    with open("berrlog.txt", "a") as fo:
        fm = msg + str(lat) + "  " + str(lon) + "  " + str(val)
        print(fm)
        fo.write(fm + "\r")


# version 3 with zones
eZones = ["NP", "NST", "NTR", "STR", "SST", "SP"]
eZdist = [0.0, 0.35, 0.2, 0.25, 0.2, 0.0]  # default zone ditribution
eZtrain_dirs = []
eZtest_dirs = []
eZtrain_filis = []
eZtest_filis = []
for zx in range(len(eZones)):
    zo = eZones[zx]
    eZtrain_dirs.append(eZtarget + "/" + zo + "/" + "train/")
    eZtest_dirs.append(eZtarget + "/" + zo + "/" + "test/")
    eZtrain_filis.append(os.listdir(eZtrain_dirs[zx]))
    eZtest_filis.append(os.listdir(eZtest_dirs[zx]))

# create all data sources as globals
elds = rio.open("wcdat/ELE.tif")
hids = rio.open("wcdat/GHI.tif")
teds = rio.open("wcdat/TEMP.tif")
sradds = []
tavgds = []
precds = []
windds = []
vapds = []
fn_prefix = "/wc2.0_30s_"
file_dirs = ["srad", "prec", "wind", "tavg", "vapr"]
for mx in range(12):
    sradds.append(rio.open("wcdat/srad" + fn_prefix + "srad_" + str2(mx) + ".tif"))
    tavgds.append(rio.open("wcdat/tavg" + fn_prefix + "tavg_" + str2(mx) + ".tif"))
    precds.append(rio.open("wcdat/prec" + fn_prefix + "prec_" + str2(mx) + ".tif"))
    windds.append(rio.open("wcdat/wind" + fn_prefix + "wind_" + str2(mx) + ".tif"))
    vapds.append(rio.open("wcdat/vapr" + fn_prefix + "vapr_" + str2(mx) + ".tif"))
lcds = rio.open("wcdat/lc/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif")

dem3ds = []
for tilx in range(24):
    dem3ds.append(rio.open("wcdat/dem3/" + gtu.dem3_files[tilx]))
    # print(dem3ds[tilx].profile['height'],dem3ds[tilx].profile['width'], )

"""
# open list of points withj dem3  data 20 klick squares with data
#with open('epoch3/summary_cts.pkl', 'r') as fi:
#  _dc = pickle.load(fi)
$testcts = _dc['testcts']
traincts = _dc['traincts'] 
train_sums =[]; test_sums = [];
assert len(traincts) == len(testcts)
#builds lists to index window
for wx in range(len(traincts)):
  if wx==0:
    train_sums.append(traincts[wx])
    test_sums.append(testcts[wx])
  else:
    train_sums.append(train_sums[wx-1] +  traincts[wx])
    test_sums.append(test_sums[wx-1] +  testcts[wx])
train_sums = np.array(train_sums)
test_sums = np.array(test_sums)
train_total = traincts.sum()
test_total = testcts.sum()
#print(train_sums)
print('total 20 klick squares for training:' + str(train_total))
print('total 20 klick squares for test:' + str(test_total))
"""

# version 3 use index from dem3, get coresponding windows into
# g tifs with other coordinate systems
def get_windsetdep(lax, lox, _30sec_x, _30sec_y, d3x):
    lon, lat = dem3ds[d3x].xy(lax, lox)
    lrlon, lrlat = dem3ds[d3x].xy(lax + (2 * _30sec_x), lox + (2 * _30sec_y))
    wclax, wclox = sradds[0].index(lon, lat)
    wclrlax, wclrlon = sradds[0].index(lrlon, lrlat)
    lclax, lclox = lcds.index(lon, lat)
    lclrlax, lclrlox = lcds.index(lrlon, lrlat)

    assert lclrlax - lclax == lclrlox - lclox
    lcunit = lclrlax - lclax
    print("d3", lox, lax, (2 * _30sec_x), (2 * _30sec_y))
    d3wnd = rio.windows.Window(lox, lax, (2 * _30sec_x), (2 * _30sec_y))
    print("wc", wclox, wclax, _30sec_x, _30sec_y)
    wcwnd = rio.windows.Window(wclox, wclax, _30sec_x, _30sec_y)
    print("lc", lclox, lclax, lcunit, lcunit)
    lcwnd = rio.windows.Window(lclox, lclax, lcunit, lcunit)
    return d3wnd, wcwnd, lcwnd


def wd_filteredstats(wd, nodat):
    wdclean = wd
    if nodat < -3.0e36:
        ndct = (wd < -3.0e36).sum()
        dct = (wd > -3.0e36).sum()
        wdclean[wdclean < -3.0e36] = 0.0

    else:
        ndct = (wd == nodat).sum()
        dct = (wd != nodat).sum()
        wdclean[wdclean == nodat] = 0
    if dct > 0:
        wdmean = float(wdclean.sum()) / float(dct)
        wd_std = wdclean.std()
    else:
        wdmean = nodat
        wd_std = nodat
    return wdmean, wd_std


# returns fraction of the area that is land, water, ice, desert
def lc_histo(lc):
    histo = np.histogram(lc, [0, 5, 199, 205, 215, 225, 256])
    recct = float(lc.size - histo[0][0] - histo[0][5])
    # pdb.set_trace()
    if recct == 0.0:
        return -1, -1, -1, -1
    else:
        land = float(histo[0][1]) / recct
        water = float(histo[0][3]) / recct
        ice = float(histo[0][4]) / recct
        desert = float(histo[0][2]) / recct
        assert abs(land + water + ice + desert - 1.0) < 0.01
        print(land, water, ice, desert)
        return land, water, ice, desert


def get_windpt(ptx, cts):  # deprecate
    wx = 0
    while cts[wx] - 1 < ptx:  # was not pciking zero and crashing one past last index
        wx += 1
    if wx == 0:  # this is fix for version 2 bug
        pickpt = ptx
    else:
        pickpt = ptx - cts[wx - 1]
    return wx, pickpt


def get_example_indexdep(wx, pickpt, bTrain):  # deprecate
    fn = "epoch3/" + gtu.dem3_files[wx]
    fn = fn[0:-4] + ".pkl"
    print(fn)
    with open(fn, "r") as fi:
        windict = pickle.load(fi)
    try:
        if bTrain:
            erow = windict["trix"][pickpt]
        else:
            erow = windict["teix"][pickpt]
    except:  # hackish
        if bTrain:
            erow = windict["trix"][pickpt - 1]
        else:
            erow = windict["teix"][pickpt - 1]
    return erow


def elev_slope(el, lati):
    mid, east = el.shape[1] / 2, el.shape[1]
    try:
        mid_slope, mid_b = np.polyfit(range(el.shape[0]), el[:, mid], 1)  # N+. S-
    except:
        pst()
    east_slope, east_b = np.polyfit(range(el.shape[0]), el[:, east - 1], 1)
    west_slope, west_b = np.polyfit(range(el.shape[0]), el[:, 0], 1)
    if lati >= 0.0:  # change ploarity to sun facing = +
        mid_slope = -mid_slope
        east_slope = -east_slope
        west_slope = -west_slope
    sun_slope = (mid_slope + east_slope + west_slope) / 3.0
    nsmid, south = el.shape[0] / 2, el.shape[0]
    nsmid_slope, mid_b = np.polyfit(range(el.shape[1]), el[:, nsmid], 1)  # W+. E-
    south_slope, east_b = np.polyfit(range(el.shape[1]), el[:, south - 1], 1)
    north_slope, west_b = np.polyfit(range(el.shape[1]), el[:, 1], 1)
    if lati >= 0.0:  # change ploarity to coriolis facing = +
        nsmid_slope = -nsmid_slope
        south_slope = -south_slope
        north_slope = -north_slope
    cori_slope = (nsmid_slope + south_slope + north_slope) / 3.0
    return sun_slope, cori_slope


# build full example with yearly and monthy data
# some qc checks are ugly but necessary
def bld_eu_examp(
    wnd, wcwnd, lcwnd, d3lax, d3lox, lon, lat, wx, bTrain
):  # an example in eng units
    ex_good = True
    temp_12mo = []
    srad_12mo = []
    prec_12mo = []
    wind_12mo = []
    vap_12mo = []
    temp_12std = []
    srad_12std = []
    prec_12std = []
    wind_12std = []
    vap_12std = []
    pwra = []
    csra = []
    # print(lax, lox, datct)
    el = dem3ds[wx].read(1, window=wnd)
    if el.shape[0] < 6 or el.shape[1] < 6:
        ex_good = False
        blog("bad el shape ", lat, lon)
        sun_slope, ew_slope = -1.0, -1.0
    else:
        sun_slope, ew_slope = elev_slope(el, lat)
    elev = np.nanmean(el)
    elev_std = np.nanstd(el)
    if elev < -100:
        ex_good = False
        blog("bad elev: ", lat, lon)
    lc = lcds.read(1, window=lcwnd)
    try:
        land, water, ice, desert = lc_histo(lc)
    except:
        pst()
        land, water, ice, desert = lc_histo(lc)
    new_alb = (0.1 * water) + (0.18 * land) + (0.3 * desert) + (0.7 * ice)
    if land == -1:
        ex_good = False
        blog("bad lwi: ", lat, lon)
    barop = gtu.bp_byalt(elev)
    if lat > 0:
        sh1h = 0.0
        nh1h = 1.0
    else:
        sh1h = 1.0
        nh1h = 0.0
    toa_12mo = np.array(gtu.toa_series(lat))
    for mx in range(12):
        vmean, vstd = wd_filteredstats(sradds[mx].read(1, window=wcwnd), 65535)
        srad_12mo.append(vmean)
        srad_12std.append(vstd)
        vmean, vstd = wd_filteredstats(precds[mx].read(1, window=wcwnd), -32768)
        prec_12mo.append(vmean)
        prec_12std.append(vstd)
        vmean, vstd = wd_filteredstats(tavgds[mx].read(1, window=wcwnd), -3.4e38)
        temp_12mo.append(vmean)
        temp_12std.append(vstd)
        vmean, vstd = wd_filteredstats(windds[mx].read(1, window=wcwnd), -3.4e38)
        wind_12mo.append(vmean)
        wind_12std.append(vstd)
        vmean, vstd = wd_filteredstats(vapds[mx].read(1, window=wcwnd), -3.4e38)
        vap_12mo.append(vmean)
        vap_12std.append(vstd)
        if srad_12mo[mx] == 65535:
            ex_good = False
            blog("bad srad ", lat, lon)
        if prec_12mo[mx] == -32768:
            ex_good = False
            blog("bad prec ", lat, lon)
        if temp_12mo[mx] == -3.4e38:
            ex_good = False
            blog("bad temp ", lat, lon)
        csra.append(srad_12mo[mx] * (1 - new_alb))
        if csra[mx] != 0.0:
            pwra.append(srad_12mo[mx] / (86.4 * csra[mx]))
        else:
            pwra.append(0.0)

    temp_12mo = np.array(temp_12mo)
    srad_12mo = np.array(srad_12mo)
    prec_12mo = np.array(prec_12mo)
    wind_12mo = np.array(wind_12mo)
    vap_12mo = np.array(vap_12mo)
    rnn_seq = [srad_12mo, prec_12mo, toa_12mo, wind_12mo, csra, pwra, vap_12mo]
    toa_pwr = gtu.toaPower(lat)
    wc_temp = gtu.acc12mo_avg(temp_12mo)
    if wc_temp < -80 or wc_temp > 80.0:
        ex_good = False
        blog("bad wc_temp: ", lat, lon, wc_temp)
    wc_prec = gtu.acc12mo_avg(prec_12mo)
    if wc_prec < 0 or wc_prec > 3000:
        ex_good = False
        blog("bad wc_prec: ", lat, lon, wc_prec)
    wc_srad = gtu.acc12mo_avg(srad_12mo)
    if wc_srad < 0.0 or wc_srad > (600.0 * wc_radCon):
        ex_good = False
        blog("bad wc_srad: ", lat, lon, wc_srad)
    wc_wind = gtu.acc12mo_avg(wind_12mo)
    if wc_wind < 0.0 or wc_wind > 100.0:
        ex_good = False
        blog("bad wc_wind: ", lat, lon, wc_wind)
    # pdb.set_trace()

    vis_dstd = np.array(srad_12std).mean()
    gtzs = (el > 0.0).sum()
    ltzs = (el < 0.0).sum()
    zs = (el == 0.0).sum()
    pwr_ratio = wc_srad / gtu.acc12mo_avg(toa_12mo)

    gtu.arprint(
        [
            lat,
            lon,
            wc_temp,
            temp_12mo.min(),
            temp_12mo.max(),
            sun_slope,
            ew_slope,
            new_alb,
        ]
    )
    # print(ex_good)
    ins = [
        lon,
        lat,
        toa_pwr,
        elev,
        barop,
        pwr_ratio,
        wc_prec,
        wc_srad,
        wc_wind,
        land,
        water,
        ice,
        desert,
        new_alb,
        sh1h,
        nh1h,
        vis_dstd,
        elev_std,
        sun_slope,
        ew_slope,
        zs,
        gtzs,
        ltzs,
    ]
    return np.array(ins), ex_good, rnn_seq, temp_12mo, wc_temp, wx
    # wx is index to one 24 d3 geotif files


def get_batch(size, bTrain):
    ins_bat = []
    d3_idx = []
    rnn_seqs = []
    wc_trs = []
    rnn_trus = []
    for ix in range(size):
        b_g = False
        # print('bx',ix)
        while not b_g:  # reject examples with bad data
            if bTrain:
                ptix = np.random.randint(0, train_total)
            else:
                ptix = np.random.randint(0, test_total)
            ins, b_g, r_s, t_12, wc_t, d3i = bld_eu_examp(ptix, 20, bTrain)
        ins_bat.append(ins)
        # trus_bat.append(temp)
        rnn_seqs.append(r_s)
        wc_trs.append(wc_t)
        rnn_trus.append(t_12)
        d3_idx.append(d3i)
    return np.array(ins_bat), rnn_seqs, wc_trs, rnn_trus, d3_idx


class zbatch:
    def __init__(self, size, zd=eZdist):
        self.zd = np.array(zd)
        z_breaks = []
        assert self.zd.sum() == 1.0
        for zval in zd:
            z_breaks.append(int(zval * size))
        # fix roundign errs
        if size - np.array(z_breaks).sum() == 1:
            z_breaks[np.argmax(np.array(z_breaks))] += 1
        elif size - np.array(z_breaks).sum() == -1:
            z_breaks[np.argmax(np.array(z_breaks))] -= 1
        elif size - np.array(z_breaks).sum() != 0:
            raise "Zone distrubtion error"
        self.z_breaks = z_breaks

    def zbatch(self, size, bTrain):
        if bTrain:
            zdirs = eZtrain_dirs
            zfilis = eZtrain_filis
        else:
            zdirs = eZtest_dirs
            zfilis = eZtest_filis
        ins_bat = []
        d3_idx = []
        rnn_seqs = []
        wc_trs = []
        rnn_trus = []
        batch_ = []
        for zx in range(len(zdirs)):
            for zbx in range(self.z_breaks[zx]):
                fix = np.random.randint(0, len(zfilis[zx]))
                batch_.append([zx, fix, zdirs[zx], zfilis[zx][fix]])
        np.random.shuffle(batch_)
        for bx in range(size):
            zx = batch_[bx][0]
            fix = batch_[bx][1]
            with open(zdirs[zx] + zfilis[zx][fix], "r") as fi:
                exD = pickle.load(fi)
            ins = exD["ins"]
            r_s = exD["r_s"]
            t_12 = exD["t_12"]
            coord = exD["coords"]
            wc_t = exD["wc_t"]
            ins_bat.append(ins)
            rnn_seqs.append(r_s)
            wc_trs.append(wc_t)
            rnn_trus.append(t_12)
            d3_idx.append(coord[-1])
        return np.array(ins_bat), rnn_seqs, wc_trs, rnn_trus, d3_idx


def count_epochZ():
    dict = {}
    train_ct = 0
    test_ct = 0
    for zo in eZones:
        dict[zo + "_train"] = len(os.listdir(eZtarget + "/" + zo + "/train"))
        train_ct += len(os.listdir(eZtarget + "/" + zo + "/train"))
        dict[zo + "_test"] = len(os.listdir(eZtarget + "/" + zo + "/test"))
        test_ct += len(os.listdir(eZtarget + "/" + zo + "/test"))
    dict["train_ct"] = train_ct
    dict["test _ct"] = test_ct
    return dict


def sPole_batch(size, bTrain):
    ins_bat = []
    trus_bat = []
    rnn_seqs = []
    wc_trs = []
    rnn_trus = []
    ins, temp, b_g, r_s, t_12, wc_t = southPole()
    for ix in range(size):
        ins_bat.append(ins)
        trus_bat.append(temp)
        rnn_seqs.append(r_s)
        wc_trs.append(wc_t)
        rnn_trus.append(t_12)
    return np.array(ins_bat), np.array(trus_bat), rnn_seqs, wc_trs, rnn_trus


# https://www.timeanddate.com/weather/antarctica/south-pole/climate
def southPole():
    pwr_ratio = 0.8
    lat = 89.5
    lon = 10.0
    temp_12mo = np.array(
        [
            -14.0,
            -37.0,
            -57.0,
            -67.0,
            -66.0,
            -64.0,
            -67.0,
            -66.0,
            -71.0,
            -59.0,
            -33.0,
            -16.0,
        ]
    )
    wc_temp = gtu.acc12mo_avg(temp_12mo)
    temp = wc_temp
    ex_good = True
    toa_12 = np.array(gtu.toa_series(-89.999))
    toa_pwr = gtu.acc12mo_avg(toa_12)
    srad12 = toa_12 * pwr_ratio * 75.0
    prec12 = (
        np.array(
            [0.36, 0.23, 0.17, 0.17, 0.22, 0.21, 0.17, 0.17, 0.14, 0.11, 0.09, 0.23]
        )
        * 25.4
    )  # inches > mm
    wind12 = (
        np.array([11, 12, 13, 13, 13, 15, 14, 13, 13, 12, 11, 10]) * 0.1
    )  # this is mph?? does nto seem right
    elev = 5000 / 3.1
    elev_std = 50.0
    vis_dstd = 5.0
    gtzs = 400
    ltzs = 0
    zs = 0
    vis_down = gtu.acc12mo_avg(toa_12) * pwr_ratio
    wc_srad = vis_down * 75.0
    wc_prec = gtu.acc12mo_avg(prec12)
    wc_wind = gtu.acc12mo_avg(wind12)
    land = 0.0
    water = 0.0
    ice = 1.0
    sh1h = 1.0
    nh1h = 0.0
    barop = gtu.bp_byalt(elev)
    rnn_seq = [srad12, prec12, toa_12, wind12]
    ins = [
        lon,
        lat,
        vis_down,
        toa_pwr,
        elev,
        barop,
        pwr_ratio,
        wc_prec,
        wc_srad,
        land,
        water,
        ice,
        sh1h,
        nh1h,
        vis_dstd,
        elev_std,
        zs,
        gtzs,
        ltzs,
        wc_wind,
    ]
    return np.array(ins), temp, ex_good, rnn_seq, temp_12mo, wc_temp


# pst()
# get_batch(400, True)
# pst()

# ins, gt_trues, r_sqs, wctrs, rnn_trus = get_batch(1004, True)
# pdb.set_trace()


# maturn edinburgh  kuwait, ponca city, sydney, port au france
# some of these locations have no data for some params,
# get_batch(0 + build_eu_example() refect windows with no data or bad data
def data_validation_test():
    pickpts = [
        [-63.18, 9.8],
        [-3.2, 55.5],
        [47.98, 29.5],
        [-97.1, 36.7],
        [151.2, -33.9],
        [70.25, -49.35],
        [0.0, 0.0],
    ]
    GISStemps = [27.5, 8.5, 28.0, 15.0, 18.0, 5.0, 27]
    for ix in range(len(pickpts)):
        lax, lox = elds.index(pickpts[ix][0], pickpts[ix][1])
        wnd, wcwnd, lcwnd = get_windset(lax, lox, 2)
        gloTemp = np.nanmean(teds.read(1, window=wnd))
        gloSRAD = np.nanmean(hids.read(1, window=wnd)) * 1000.0 / 24.0
        lc = lcds.read(1, window=lcwnd)
        print("temps for ", pickpts[ix])

        srad_months = []
        prec_months = []
        temp_months = []
        for mx in range(12):
            srad_months.append(
                wd_filteredstats(sradds[mx].read(1, window=wcwnd), 65535)
            )
            prec_months.append(
                wd_filteredstats(precds[mx].read(1, window=wcwnd), -32768)
            )
            temp_months.append(
                wd_filteredstats(tavgds[mx].read(1, window=wcwnd), -3.4e38)
            )
        temp_months = np.array(temp_months)
        print(GISStemps[ix], gloTemp, temp_months.mean())
        print("radiations")
        # pdb.set_trace()
        srad_months = np.array(srad_months)
        toa_months = gtu.toa_series(pickpts[ix][1])
        sr12 = gtu.acc12mo_avg(srad_months)
        sr_cal = sr12 / gloSRAD
        gtu.arprint([gloSRAD, sr12, sr_cal])
        gtu.arprint(srad_months / sr_cal)
        gtu.arprint(toa_months)
        prec_months = np.array(prec_months)
        print("precip", prec_months.mean())
        print(lc)
        print("lc", lc_histo(lc))
        pdb.set_trace()
