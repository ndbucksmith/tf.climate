import numpy as np
import time
import datetime
import rasterio as rio
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pdb
import math
import os

pst = pdb.set_trace

"""
utilites for geotiff files
include building, saving loading lists of windows
longitude latutde to index conversion (deprecated use rasterio)
toa power,alnedo  by latitude
toa power by lattitude and month

copyright 2019 Nelson "Buck" Smith
"""


class latluts:
    lattitude = [
        -90.0,
        -80.0,
        -70.0,
        -60.0,
        -50.0,
        -40.0,
        -30.0,
        -20.0,
        -10.0,
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
    ]
    albedo = [
        0.73,
        0.73,
        0.4,
        0.18,
        0.15,
        0.12,
        0.12,
        0.12,
        0.11,
        0.11,
        0.12,
        0.15,
        0.17,
        0.16,
        0.2,
        0.30,
        0.42,
        0.52,
        0.58,
    ]
    power = [
        131.6666667,
        167.1875,
        197.7083333,
        233.8541667,
        280.9375,
        326.1458333,
        362.3958333,
        391.7708333,
        409.375,
        414.5833333,
        411.0416667,
        395,
        366.9791667,
        331.9791667,
        287.9166667,
        241.6666667,
        206.25,
        176.1458333,
        140.7291667,
    ]


mon_wts = [31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]

dem3_files = os.listdir("wcdat/dem3")
dem3_files.sort()
dem3_rows = [dem3_files[0:6], dem3_files[6:12], dem3_files[12:18], dem3_files[18:24]]

global_precip_annual_volume = 505000  # km3
global_precip_annual_mass = 505000.0e12  # kg
global_precip_turnover = 7  # days
total_water_vol_atmosphere = 12900
total_water_mass_atmosphere = 12900e12
heatcap_ice = 2.1  # kJ/kg/ deg C
heatcap_water = 4.2
heatcap_water_vapor = 1.996


def dem3_fix(lon, lat):
    rowx = int((90.0 - lat) / 45.0)
    colx = int((lon + 180.0) / 60.0)
    fix = (6 * rowx) + colx
    return fix


def acc12mo_avg(vals):
    wtd_sum = 0.0
    for mx in range(12):
        wtd_sum += mon_wts[mx] * vals[mx]
    wtd_avg = wtd_sum / 365.25
    return wtd_avg


def albbylat(lattitude):
    return np.interp(lattitude, latluts.lattitude, latluts.albedo)


# top of atmosphere power
def toaPower(lat):
    P = np.interp(lat, latluts.lattitude, list(reversed(latluts.power)))
    return P


# print array with rounded values for easy compare
def arprint(inp):
    ostr = ""
    for ix in range(len(inp)):
        try:
            ostr += str(round(inp[ix], 2))
        except:
            ostr += inp[ix]
        ostr += "  "
    print ostr[0:-2]
    return ostr[0:-2]


def save_winlist(wins, res, path):
    dc = {}
    dc["ct"] = len(wins)
    dc["res"] = res
    dc["windows"] = wins
    dmp = pickle.dumps(dc)
    with open(path, "w") as fo:
        fo.write(dmp)


def load_winlist(path):
    with open(path, "r") as fi:
        dc = pickle.load(fi)
    return dc["windows"], dc["res"], dc["ct"]


# barometric pressure in torr (mmHg) by altitude
def bp_byalt(alt):
    alt = max(0, alt)
    return 760 * (1.0 / (2.0 ** (alt / 5400.0)))


# toa power factors for 24 hours
def Fp_on_first(mx, lat):
    st = datetime.datetime(2018, 3, 21, 0)
    day_hrs = []
    Fps = []
    for hr in range(24):
        day_hrs.append(
            (datetime.datetime(2019, mx, 1, hr,) - st).total_seconds() / 3600
        )
        Fps.append(Fp(day_hrs[hr], lat))
    Fps = np.array(Fps)
    return day_hrs, Fps, Fps.sum() * 1350 / 24.0


# get the power factor as function of time lat and declination = 23
def Fp(t, lat):
    theta = lat * 3.14159 / 180.0
    delta = 23.0 * 3.14159 / 180.0
    _c = math.cos
    _s = math.sin
    _W = 2 * 3.14159 / (365.24 * 24)
    _w = 2 * 3.14159 / (24.0 + (4.0 / 3600.0))
    f1 = _c(_W * t) * _c(theta) * _c(_w * t)
    f2 = _c(delta) * _c(theta) * _s(_w * t)
    f3 = _s(delta) * _s(theta)
    f4 = _s(_W * t)
    _Fp = max(0.0, (f1 + (f4 * (f2 + f3))))
    return _Fp


# get 12 month series of top of atmo power
def toa_series(lat):
    firsts = []
    mos = []
    for mx in range(1, 13):
        dhs, fps, dayP = Fp_on_first(mx, lat)
        firsts.append(dayP)
    for mx in range(12):
        mos.append((firsts[mx] + firsts[(mx + 1) % 12]) / 2.0)
    return mos


# compare monthlt power series to og lookup
def toa_series_val():
    for lat in range(-60, 70, 10):
        lat = float(lat)
        toalut = toaPower(lat)
        toaser = acc12mo_avg(toa_series(lat))
        arprint([toalut, toaser])


def tabler(ctxt, name, colhdrs):
    fi, ax = plt.subplots(1)
    fi.suptitle(name)
    fi.subplots_adjust(top=0.95, bottom=0.01, left=0.1, right=0.99)
    tbl = ax.table(cellText=ctxt, colLabels=colhdrs, loc="center")
    ax.axis("tight")
    ax.axis("off")
    # ax[0].xticks([])
    # ax[0].yticks([])
    return fi, ax


def scat(x, y, z, cmp, name, nrm):
    fi_, ax_ = plt.subplots(1)
    fi_.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.99)
    fi_.suptitle(name)
    ax_.scatter(x, y, s=1, c=z, cmap=cmp, norm=nrm)
    return fi_, ax_


def plotter(plts, name, lbls):
    fi_, ax_ = plt.subplots(1)
    fi_.suptitle(name)
    for px in range(len(plts)):
        ax_.plot(plts[px], label=lbls[px])
    ax_.legend()
    return fi_, ax_


def addnote(fig, params):
    fig.text(
        0.02,
        0.02,
        str(params["file_ct"]) + " batches of 400 examples",
        transform=plt.gcf().transFigure,
    )
    fig.text(0.5, 0.02, params["mdl_path"], transform=plt.gcf().transFigure)
    return fig


def mapper(x, y, z, cmp, name, nrm):
    fi, ax = plt.subplots(1)
    fi.suptitle(name)  #'sensitivity map from test data 1 deg C mse'
    fi.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.99)
    ax.scatter(x, y, c=z, s=1, cmap=cmp, norm=nrm)
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, norm=nrm, cmap=cmp)
    return fi, ax


# -----------------old code some of which may still be used in win_stat_book.py
# deprecate and use rasterio xy()
def llscale(lax, lox, res, startLat):
    res = float(res)
    lat = startLat - (float(lax) / res)
    lon = -180.0 + float(lox) / res
    return lat, lon


# deprecate and use raserio index()
def invllscale(lat, lon, res, startLat):
    lax = int((startLat - lat) * res)
    lox = int((lon + 180.0) * res)
    return lax, lox


# deprecated windows now built handled  in batcher using rasterio
def get_windims(hei, wid, nocols, norows):
    rw = wid / nocols
    ch = hei / norows
    modrw = wid % nocols
    modch = hei % norows
    assert modrw == 0
    assert modch == 0
    return rw, ch, modrw, modch


# deprecated
def build_winlist(NestLat, SestLat, res, nocols, hemirows):
    windows = []  # Window(col_off, row_off, width, height)
    # build NHemi list row by row
    rw, ch, modrw, modch = get_windims(NestLat * res, 360 * res, nocols, hemirows)
    modch = 0
    coff = 0
    roff = 0
    for loct in range(nocols):
        for lact in range(hemirows):
            windows.append(rio.windows.Window(loct * rw, lact * ch, rw, ch))
    eq_off = lact * ch
    rw, ch, modrw, modch = get_windims(SestLat * res, 360 * res, nocols, hemirows)
    for loct in range(nocols):
        for lact in range(hemirows):
            windows.append(rio.windows.Window(loct * rw, eq_off + (lact * ch), rw, ch))
    return windows
