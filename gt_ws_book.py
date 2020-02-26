import numpy as np
import time
import rasterio as rio
import matplotlib as plt
import pickle
import pdb
import math
import gt_utils as gtu
import wc_batcher as wcb
import warnings

warnings.filterwarnings("error")
pst = pdb.set_trace

"""
create pkl files with dicts for each window
version 3 uses dem3 files for elevation.  
There are 24 files to cover eath with 15" res elevation data
this is 2x better reesolution gs elevation and also cover the planet
as in geotif nodata = None
this function is now rewrit to correctly use raterio index() and xy()
in place of my homebrew functions which now be removed

"""
targ = "epochZ_/"


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
        wdmean = wdclean.sum() / dct
    else:
        wdmean = nodat
    return wdmean


unit_x = 10
unit_y = 10  # 30 second units
u15x = 2 * unit_x
u15y = 2 * unit_y
# 15 second units
hids = rio.open("wcdat/srad/wc2.0_30s_srad_01.tif")
teds = rio.open("wcdat/tavg/wc2.0_30s_tavg_01.tif")
lcds = rio.open("wcdat/lc/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif")
_st = time.time()
traincts = []
testcts = []
badels = []
N_polar_ct = 0, 0
S_polar_ct = 0, 0
N_temp_ct = 0, 0
S_temp_ct = 0.0
N_tropic_ct = 0, 0
S_tropic_ct = 0, 0
N_tropic_bds = 0, 0, 0, 0, 0, 0
N_pole_bds = 0, 0, 0, 0, 0, 0
S_tropic_bds = 0, 0, 0, 0, 0, 0
S_pole_bds = 0, 0, 0, 0, 0, 0
Zones_gt_bound = [60.0, 30.0, 0.0, -30, -60.0, -90.0]
Zones_lt_bound = [90.0, 60.0, 30.0, 0.0, -30.0, -60.0]
rg00 = range(0, 7200, u15x)
rg01 = range(7200, 10800 - u15x, u15x)
rg10 = range(0, 3600, u15x)
rg11 = range(3600, 10800 - u15x, u15x)
d3_zoners = [[rg00, rg01], [rg10, rg11], [rg10, rg11], [rg00, rg01]]
d3_zones = [["NP", "NST"], ["NST", "NTR"], ["STR", "SST"], ["SST", "SP"]]
zoneTr_cts = {"NP": 0, "NST": 0, "NTR": 0, "STR": 0, "SST": 0, "SP": 0}
zoneTe_cts = {"NP": 0, "NST": 0, "NTR": 0, "STR": 0, "SST": 0, "SP": 0}
tile_ranges = [(0, 6), (6, 12), (12, 18), (18, 24)]
lo_range = range(0, 14400 - u15y, u15y)


def zone_chk(d3lax, zoneset):
    top = d3lax in zoneset[0]
    bottom = d3lax in zoneset[1]
    assert top != bottom
    return top, bottom


for d3row in range(4):
    for tilx in range(tile_ranges[d3row][0], tile_ranges[d3row][1]):
        elds = rio.open("wcdat/dem3/" + gtu.dem3_files[tilx])
        print("opening " + gtu.dem3_files[tilx])
        upright = elds.xy(0, 0)
        lowleft = elds.xy(10800, 14401)
        print("covering " + str(upright) + " to " + str(lowleft))

        examp_ct = 0
        hi_badct = 0
        te_badct = 0
        lc_badct = 0
        el_badct = 0
        trainct = 0
        noelct = 0
        nohict = 0
        notect = 0
        badct = 0
        teix = []
        trix = []
        testct = 0
        dc = {}
        bads = []
        for d3lax in range(0, 10800 - u15x, u15x):
            for d3lox in lo_range:

                # def xy(self, row, col, offset="center"):
                lon, lat = elds.xy(d3lax, d3lox)
                lrlon, lrlat = elds.xy(d3lax + u15x, d3lox + u15y)
                # def index(self, x, y, op=math.floor, precision=None):
                wclax, wclox = hids.index(lon, lat)
                wclrlax, wclrlox = hids.index(lrlon, lrlat)
                wc_unit_y = wclrlax - wclax
                wc_unit_x = wclrlox - wclox
                lclax, lclox = lcds.index(lon, lat)
                lclrlax, lclrlox = lcds.index(lrlon, lrlat)
                lc_unit_y = lclrlax - lclax
                lc_unit_x = lclrlox - lclox
                # Window(col_off, row_off, width, height)
                wcws = rio.windows.Window(wclox, wclax, wc_unit_x, wc_unit_y)
                lcws = rio.windows.Window(lclox, lclax, lc_unit_x, lc_unit_y)
                d3ws = rio.windows.Window(d3lox, d3lax, 2 * unit_x, 2 * unit_y)
                el = elds.read(1, window=d3ws)
                hi = hids.read(1, window=wcws)
                te = teds.read(1, window=wcws)
                lc = lcds.read(1, window=lcws)
                if wd_filteredstats(hi, 65535) != 65535:
                    hi_datact = 10
                else:
                    hi_datact = 0
                if wd_filteredstats(te, -3.4e38) > -150:
                    te_datact = 10
                else:
                    te_datact = 0
                lc_datact = (
                    (lc.shape[0] * lc.shape[1]) - (lc == 0).sum() - (lc == 255).sum()
                )
                try:
                    elev = el.mean()
                    el_datact = (el > -500).sum()
                except:
                    el_datact = 0
                if hi_datact > 3 and te_datact > 3 and lc_datact > 3 and el_datact > 3:
                    to, bo = zone_chk(d3lax, d3_zoners[d3row])
                    if to:
                        zone = d3_zones[d3row][0]
                    else:
                        zone = d3_zones[d3row][1]

                    # if examp_ct % 5000 == 1:
                    print(lon, lat, elev, hi_datact, te_datact, lc_datact)
                    examp_ct += 1

                    if np.random.random() < 0.9:
                        trix.append(
                            [
                                d3lax,
                                d3lox,
                                lon,
                                lat,
                                elev,
                                hi_datact,
                                te_datact,
                                lc_datact,
                                unit_x,
                                unit_y,
                                lc_unit_x,
                                lc_unit_y,
                            ]
                        )
                        trainct += 1
                        bTrain = True
                        ins, b_g, r_s, t_12, wc_t, d3i = wcb.bld_eu_examp(
                            d3ws, wcws, lcws, d3lax, d3lox, lon, lat, tilx, True
                        )
                    else:
                        teix.append(
                            [
                                d3lax,
                                d3lox,
                                lon,
                                lat,
                                elev,
                                hi_datact,
                                te_datact,
                                lc_datact,
                                unit_x,
                                unit_y,
                                lc_unit_x,
                                lc_unit_y,
                            ]
                        )
                        testct += 1
                        bTrain = False
                        ins, b_g, r_s, t_12, wc_t, d3i = wcb.bld_eu_examp(
                            d3ws, wcws, lcws, d3lax, d3lox, lon, lat, tilx, False
                        )

                    def fmlat(co):
                        if co > 0.0:
                            return "_N" + str(int(round(co, 2) * 100))
                        else:
                            return "_S" + str(int(round(co, 2) * 100))[1:]

                    def fmlon(co):
                        if co > 0.0:
                            return "_E" + str(int(round(co, 2) * 100))
                        else:
                            return "_W" + str(int(round(co, 2) * 100))[1:]

                    if b_g:
                        if bTrain:
                            zoneTr_cts[zone] += 1
                            fpath = (
                                targ
                                + zone
                                + "/train/ex_"
                                + str(zoneTr_cts[zone])
                                + fmlon(lon)
                                + fmlat(lat)
                                + ".pkl"
                            )
                        else:
                            zoneTe_cts[zone] += 1
                            fpath = (
                                targ
                                + zone
                                + "/test/ex_"
                                + str(zoneTe_cts[zone])
                                + fmlon(lon)
                                + fmlat(lat)
                                + ".pkl"
                            )
                        exD = {
                            "coords": [d3lax, d3lox, lon, lat, tilx],
                            "ins": ins,
                            "r_s": r_s,
                            "t_12": t_12,
                            "wc_t": wc_t,
                        }
                        with open(fpath, "w") as fo:
                            dmp = pickle.dumps(exD)
                            fo.write(dmp)

                else:
                    badct += 1
                    if hi_datact < 4:
                        hi_badct += 1
                    if te_datact < 4:
                        te_badct += 1
                    if lc_datact < 4:
                        lc_badct += 1
                    if el_datact < 4:
                        el_badct += 1

        dc["trix"] = np.array(trix)
        dc["teix"] = np.array(teix)
        dc["badct"] = badct
        dc["trainct"] = trainct
        dc["testct"] = testct
        dc["zTr_cts"] = zoneTr_cts
        dc["zTe_cts"] = zoneTe_cts
        print(trainct, testct, "time:", time.time() - _st)
        print(badct, hi_badct, te_badct, lc_badct, el_badct)
        print(zoneTr_cts)
        with open(targ + gtu.dem3_files[tilx][0:-4] + ".pkl", "w") as fo:
            dmp = pickle.dumps(dc)
            fo.write(dmp)
        traincts.append(trainct)
        testcts.append(testct)

dc = {}
testcts = np.array(testcts)
traincts = np.array(traincts)
dc["testcts"] = testcts
dc["traincts"] = traincts
dc["testcts"] = testcts
dc["traincts"] = traincts
with open(targ + "/summary_cts.pkl", "w") as fo:
    dmp = pickle.dumps(dc)
    fo.write(dmp)
print(dc)
print("total train set size: " + str(traincts.sum()))
