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

Version 2 batcher corrects errors in prec scaling and bad data filters 
and adds annual wind value to nn_features

copyright 2019 Nelson 'Buck' Smith

"""
# three rows - 9 elements 0:8 # 3 elements 9:11  #8 elemetns 12:19
nn_features = ['lon', 'lat', 'vis_down', 'toa_pwr', 'elev', 'barop', 'pwr_ratio', 'wc_prec', 'wc_srad', \
             'land', 'water', 'ice', \
               'sh1h', 'nh1h', 'vis_dstd', 'elev_std', 'zs', 'gtzs', 'ltzs', 'wc_wind'] 
nn_feat_len = len(nn_features)
nn_norms = [1.0, 90.0, 310.0,  415.0, 7000.0, 760.0, 1.0, 600.0, 25000.0, \
              1.0, 1.0, 1.0,  \
              1.0, 1.0, 50.0, 4000.0, 400.0, 400.0, 400.0, 10.0]
assert len(nn_norms) == nn_feat_len
rnn_features =['srad','prec','toa','wind']
rnn_norms = [32768.0, 2400.0, 500.0, 11.0]
rnn_feat_len = 4

#to build wc data file names
def str2(mx):
  if mx < 9:
    return str(0) + str(mx+1)
  else:
    return str(mx+1)

#record bad data errors during batching
def blog(msg, lat, lon, val=0.001):
  with open('berrlog.txt', 'a') as fo:
    fm = msg + str(lat) + '  ' + str(lon) + '  ' + str(val)
    print(fm)
    fo.write(fm + '\r')

# create all data sources as globals
elds = rio.open('wcdat/ELE.tif')
hids = rio.open('wcdat/GHI.tif')
teds = rio.open('wcdat/TEMP.tif')  
sradds = []; tavgds = []; precds = []; windds = [];
fn_prefix = '/wc2.0_30s_'
file_dirs = ['srad', 'prec', 'wind', 'tavg']
for mx in range(12):
  sradds.append(rio.open('wcdat/srad' + fn_prefix + 'srad_' +  str2(mx) +'.tif'))
  tavgds.append(rio.open('wcdat/tavg' + fn_prefix + 'tavg_' +  str2(mx) +'.tif'))
  precds.append(rio.open('wcdat/prec' + fn_prefix + 'prec_' +  str2(mx) +'.tif'))
  windds.append(rio.open('wcdat/wind' + fn_prefix + 'wind_' +  str2(mx) +'.tif'))
lcds = rio.open('wcdat/lc/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif')


# open list of points withj global solar data 20 klick squares with data
with open('winstats/summary_cts.pkl', 'r') as fi:
  _dc = pickle.load(fi)
testcts = _dc['testcts']
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

#use index from global solar atlas, get coresponding windows into
# g tifs with other coordinate systems
def get_windset(lax, lox, _unit):
  lon, lat = elds.xy(lax, lox)
  lrlon, lrlat = elds.xy(lax+_unit, lox+_unit)
  wclax, wclox = sradds[0].index(lon, lat)
  wclrlax, wclrlon = sradds[0].index(lrlon, lrlat)
  lclax, lclox = lcds.index(lon, lat)
  lclrlax, lclrlox = lcds.index(lrlon, lrlat)
  assert lclrlax - lclax == lclrlox - lclox
  lcunit =  lclrlax - lclax
  wnd =  rio.windows.Window(lox, lax, _unit, _unit)
  wcwnd =  rio.windows.Window(wclox, wclax, _unit, _unit)
  lcwnd =  rio.windows.Window(lclox, lclax, lcunit , lcunit)
  return  wnd, wcwnd, lcwnd

def wd_filteredstats(wd, nodat):
  wdclean = wd
  if nodat < -3.0e+36:
    ndct = (wd < -3.0e+36).sum()
    dct = (wd > -3.0e+36).sum()
    wdclean[wdclean < -3.0e+36] = 0.0    
  else:  
    ndct = (wd == nodat).sum()
    dct = (wd != nodat).sum() 
    wdclean[wdclean == nodat] = 0
  if dct > 0:
    wdmean = wdclean.sum()/dct
  else:
    wdmean  = nodat
  return wdmean

#returns fraction of the area that is land, water, ice
def lc_histo(lc):
  histo = np.histogram(lc, [0, 5, 205, 215, 225, 256])
  recct = float(lc.size -histo[0][0] - histo[0][4])
  #pdb.set_trace()
  if recct == 0.0:
    return -1, -1, -1
  else:
    land = float(histo[0][1])/recct
    water = float(histo[0][2])/recct
    ice = float(histo[0][3])/recct
    assert abs(land + water + ice - 1.0) < 0.01
    return land, water, ice                     


#maturn edinburgh  kuwait, ponca city, sydney, port au france
# some of these locations have no data for some params,
#get_batch(0 + build_eu_example() refect windows with no data or bad data
def data_validation_test():
  pickpts = [[-63.18, 9.8],[-3.2, 55.5],[47.98, 29.5], [-97.1, 36.7], \
                [151.2, -33.9], [70.25, -49.35], [0.0, 0.0]]
  GISStemps = [27.5, 8.5, 28.0, 15.0, 18.0, 5.0, 27]
  for ix in range(len(pickpts)):
    lax, lox = elds.index(pickpts[ix][0], pickpts[ix][1])
    wnd, wcwnd, lcwnd = get_windset(lax, lox, 2)       
    gloTemp = np.nanmean(teds.read(1, window=wnd))
    gloSRAD = np.nanmean(hids.read(1, window=wnd)) * 1000.0/24.0
    lc = lcds.read(1, window=lcwnd)
    print('temps for ', pickpts[ix])

    srad_months = []; prec_months = []; temp_months = []
    for mx in range(12):
      srad_months.append(wd_filteredstats(sradds[mx].read(1, window=wcwnd), 65535))
      prec_months.append(wd_filteredstats(precds[mx].read(1, window=wcwnd), -32768))
      temp_months.append(wd_filteredstats(tavgds[mx].read(1, window=wcwnd), -3.4e+38))
    temp_months = np.array(temp_months)
    print(GISStemps[ix], gloTemp, temp_months.mean())
    print('radiations')
    #pdb.set_trace()
    srad_months = np.array(srad_months)
    toa_months = gtu.toa_series( pickpts[ix][1])
    sr12 = gtu.acc12mo_avg(srad_months)
    sr_cal = sr12/gloSRAD
    gtu.arprint([gloSRAD, sr12, sr_cal])
    gtu.arprint(srad_months/sr_cal)
    gtu.arprint(toa_months)
    prec_months = np.array(prec_months)
    print('precip', prec_months.mean())
    print(lc)
    print('lc',  lc_histo(lc)  )                 
    pdb.set_trace()   

#data_validation_test()

def get_windpt(ptx, cts):
  wx = 0 
  while cts[wx] < ptx:
    wx +=1
  pickpt = ptx - cts[wx]
  return wx, pickpt

def get_example_index(wx, pickpt, bTrain):
  fn = 'winstats/book_' + str(wx*5) +  '.pkl'
  #print(fn)
  with open(fn, 'r') as fi:
    windict = pickle.load(fi)
  if bTrain:
    wlax, wlox, datact = windict['trix'][pickpt]
  else:
    wlax, wlox, datact = windict['teix'][pickpt]
  windo = rio.windows.Window(wlox, wlax, 20, 20)  
  return windo, wlax, wlox, datact


# build full example with yearly and monthy data
# some qc checks are ugly but necessary
def bld_eu_examp(ptix, _unit, bTrain): #an example in eng units
  ex_good = True
  if bTrain:
    wx, pickpt = get_windpt(ptix, train_sums)
  else:
    wx, pickpt = get_windpt(ptix, test_sums)
  #print(wx, pickpt)
  wnd, lax, lox, datct  = get_example_index(wx, pickpt, bTrain)
  if _unit == 10:
    corner = np.random.randint(0,4)
    if corner == 1:
      lox += 10
    elif corner == 2:
      lox += 10; lax += 10;
    elif corner == 3:
      lax += 10
  wnd, wcwnd, lcwnd = get_windset(lax, lox, _unit) 
  lon, lat = elds.xy(lax, lox)
  temp_12mo = []; srad_12mo = []; prec_12mo = []; wind_12mo = []
  #print(lax, lox, datct)
  el = elds.read(1, window=wnd)
  te = teds.read(1, window=wnd)
  hi = hids.read(1, window=wnd)
  for mx in range(12):
      srad_12mo.append(wd_filteredstats(sradds[mx].read(1, window=wcwnd), 65535))
      prec_12mo.append(wd_filteredstats(precds[mx].read(1, window=wcwnd), -32768))
      temp_12mo.append(wd_filteredstats(tavgds[mx].read(1, window=wcwnd), -3.4e+38))
      wind_12mo.append(wd_filteredstats(windds[mx].read(1, window=wcwnd), -3.4e+38))
      if srad_12mo[mx] == 65535: ex_good = False; blog('bad srad ', lat, lon);
      if prec_12mo[mx] == -32768: ex_good = False; blog('bad prec ', lat, lon);
      if temp_12mo[mx] == -3.4e38:  ex_good = False; blog('bad temp ', lat, lon);
  toa_12mo = np.array(gtu.toa_series(lat))    
  temp_12mo = np.array(temp_12mo)
  srad_12mo = np.array(srad_12mo)
  prec_12mo = np.array(prec_12mo)
  wind_12mo = np.array(wind_12mo)
  rnn_seq = [srad_12mo, prec_12mo, toa_12mo, wind_12mo]
  vis_down = np.nanmean(hi) * 1000.0 / 24
  if vis_down < -1.0 or vis_down > 500.0: ex_good = False;  blog('bad gsa_vis_down: ', lat, lon, vis_down)
  toa_pwr = gtu.toaPower(lat)
  assert toa_pwr > vis_down
  temp  = np.nanmean(te)
  if temp < -80  or temp > 80.0: ex_good = False;  blog('bad gsa_temp: ', lat, lon, temp)
  wc_temp = gtu.acc12mo_avg(temp_12mo)
  if wc_temp < - 80  or wc_temp > 80.0: ex_good =False;  blog('bad wc_temp: ', lat, lon, wc_temp)
  wc_prec = gtu.acc12mo_avg(prec_12mo)
  if wc_prec < 0   or wc_prec > 3000:  ex_good =False;  blog('bad wc_prec: ', lat, lon, wc_prec)
  wc_srad = gtu.acc12mo_avg(srad_12mo)
  if wc_srad< 0.0  or wc_srad > (600.0*85.0):  ex_good =False;  blog('bad wc_srad: ', lat, lon, wc_srad)
  wc_wind = gtu.acc12mo_avg(wind_12mo)
  if wc_wind< 0.0  or wc_wind > 100.0:  ex_good =False;  blog('bad wc_srad: ', lat, lon, wc_wind)
  #pdb.set_trace()
  elev = np.nanmean(el)
  elev_std = np.nanstd(el)
  vis_dstd = np.nanstd(hi) * 1000.0 / 24
  gtzs = (el > 0.0).sum()
  ltzs = (el < 0.0).sum()
  zs = (el == 0.0).sum()
  pwr_ratio = vis_down / toa_pwr
  if elev < -100:
    ex_good = False; blog('bad elev: ', lat, lon)
  lc = lcds.read(1, window=lcwnd)
  land, water, ice = lc_histo(lc)
  if land == -1: ex_good = False;  blog('bad lwi: ', lat, lon)
  barop = gtu.bp_byalt(elev)
  if lat > 0:
    sh1h = 0.0; nh1h = 1.0
  else:  
    sh1h = 1.0; nh1h = 0.0
  gtu.arprint([lat, lon, temp, wc_temp, temp_12mo.min(), temp_12mo.max(), wind_12mo.max()])
  #print(ex_good)
  ins = [lon, lat, vis_down, toa_pwr, elev, barop, pwr_ratio, wc_prec, wc_srad, land, water, ice, \
         sh1h, nh1h, vis_dstd, elev_std, zs, gtzs, ltzs, wc_wind]  
  return np.array(ins), temp, ex_good, rnn_seq, temp_12mo, wc_temp


def get_batch(size, bTrain):
  ins_bat = []; trus_bat = []; rnn_seqs=[];
  wc_trs=[]; rnn_trus = [];   
  for ix in range(size):
    b_g = False
    #print('bx',ix)
    while not b_g:  # reject examples with bad data
      if bTrain:
        ptix = np.random.randint(0,train_total)
      else:
        ptix = np.random.randint(0,test_total)      
      ins, temp, b_g, r_s, t_12, wc_t = bld_eu_examp(ptix, 20, bTrain)
    ins_bat.append(ins)
    trus_bat.append(temp)
    rnn_seqs.append(r_s)
    wc_trs.append(wc_t)
    rnn_trus.append(t_12)  
  return np.array(ins_bat), np.array(trus_bat), rnn_seqs, wc_trs, rnn_trus  



#ins, gt_trues, r_sqs, wctrs, rnn_trus = get_batch(1004, True)
#pdb.set_trace()
