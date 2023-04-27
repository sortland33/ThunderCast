#Author: Stephanie M. Ortland
import os, sys
from glob import glob
import torch
import psutil
import pathlib
from netCDF4 import Dataset
import torch.nn as nn
from datetime import datetime,timedelta
from time import time
from torch.utils.data import DataLoader
import errno
import pickle
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from satpy import Scene
import argparse
from pyproj import Proj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyresample as pr
import pygrib as pg
from metpy.plots import ctables
import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy.ma as ma
from scipy.ndimage import zoom
import random
import dask.array as da

#Define functions
def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise

#Determine dataset stats for normalizing the dataset in the deep learning model
if __name__ == "__main__":
    train_file = '/ships19/grain/convective_init/data_lists/training/train_files.txt'
    with open(train_file, 'r') as data_files:
        training_files = data_files.read().splitlines()

    print('Number of Files:', len(training_files))
    vars = {'goes16_half_km':['C02'], 'goes16_one_km':['C05', 'solar_zenith', 'solar_azimuth'], 'goes16_two_km':['C13', 'C11', 'C14', 'C15'], 'target':{'MRMS_one_km':['reflectivity_60min_neg10C_max']}} #in format group:[variable(s)]

    inputs = []
    for k in vars.keys():
        if k != 'target':
            inputs.append(vars[k])
    all_percents = []
    mean_C02 = []
    mean_C05 = []
    mean_C11 = []
    mean_C13 = []
    mean_C14 = []
    mean_C15 = []
    mean_sz = []
    mean_sa = []
    std_C13 = []
    std_C02 = []
    std_C05 = []
    std_C11 = []
    std_C14 = []
    std_C15 = []
    std_sz = []
    std_sa = []
    print('Determining avg percent, channel means, and channel stds....')
    for file in training_files:
        ds = Dataset(file, 'r')
        percent = ds.getncattr('percent_60max_neg10C_greater30dBZ')
        all_percents.append(percent)
        for i, inps in enumerate(inputs):
            if len(inps) > 1:
                for inp in inps:
                    if (inp in vars['goes16_half_km']):
                        group = ds.groups['goes16_half_km']
                        if inp == 'C02':
                            m = da.nanmean(da.from_array(group['C02']))
                            s = da.nanstd(da.from_array(group['C02']))
                            m, s, = da.compute(m,s)
                            mean_C02.append(m)
                            std_C02.append(s)
                    if (inp in vars['goes16_one_km']):
                        group = ds.groups['goes16_one_km']
                        if inp == 'C05':
                            m = da.nanmean(da.from_array(group['C05']))
                            s = da.nanstd(da.from_array(group['C05']))
                            m, s = da.compute(m, s)
                            mean_C05.append(m)
                            std_C05.append(s)
                        if inp == 'solar_azimuth':
                            m = da.nanmean(da.from_array(group['solar_azimuth']))
                            s = da.nanstd(da.from_array(group['solar_azimuth']))
                            m, s = da.compute(m, s)
                            mean_sa.append(m)
                            std_sa.append(s)
                        if inp == 'solar_zenith':
                            m = da.nanmean(da.from_array(group['solar_zenith']))
                            s = da.nanstd(da.from_array(group['solar_zenith']))
                            m, s = da.compute(m, s)
                            mean_sz.append(m)
                            std_sz.append(s)
                    if (inp in vars['goes16_two_km']):
                        group = ds.groups['goes16_two_km']
                        if inp == 'C11':
                            m = da.nanmean(da.from_array(group['C11']))
                            s = da.nanstd(da.from_array(group['C11']))
                            m, s, = da.compute(m,s)
                            mean_C11.append(m)
                            std_C11.append(s)
                        if inp == 'C13':
                            m = da.nanmean(da.from_array(group['C13']))
                            s = da.nanstd(da.from_array(group['C13']))
                            m, s, = da.compute(m,s)
                            mean_C13.append(m)
                            std_C13.append(s)
                        if inp == 'C14':
                            m = da.nanmean(da.from_array(group['C14']))
                            s = da.nanstd(da.from_array(group['C14']))
                            m, s, = da.compute(m,s)
                            mean_C14.append(m)
                            std_C14.append(s)
                        if inp == 'C15':
                            m = da.nanmean(da.from_array(group['C15']))
                            s = da.nanstd(da.from_array(group['C15']))
                            m, s, = da.compute(m,s)
                            mean_C15.append(m)
                            std_C15.append(s)
            else:
                inp = inps[0]
                if (inp in vars['goes16_half_km']):
                    group = ds.groups['goes16_half_km']
                    if inp == 'C02':
                        m = da.nanmean(da.from_array(group['C02']))
                        s = da.nanstd(da.from_array(group['C02']))
                        m, s = da.compute(m, s)
                        mean_C02.append(m)
                        std_C02.append(s)
                if (inp in vars['goes16_one_km']):
                    group = ds.groups['goes16_one_km']
                    if inp == 'C05':
                        m = da.nanmean(da.from_array(group['C05']))
                        s = da.nanstd(da.from_array(group['C05']))
                        m, s = da.compute(m, s)
                        mean_C05.append(m)
                        std_C05.append(s)
                    if inp == 'solar_azimuth':
                        m = da.nanmean(da.from_array(group['solar_azimuth']))
                        s = da.nanstd(da.from_array(group['solar_azimuth']))
                        m, s = da.compute(m, s)
                        mean_sa.append(m)
                        std_sa.append(s)
                    if inp == 'solar_zenith':
                        m = da.nanmean(da.from_array(group['solar_zenith']))
                        s = da.nanstd(da.from_array(group['solar_zenith']))
                        m, s = da.compute(m, s)
                        mean_sz.append(m)
                        std_sz.append(s)
                if (inp in vars['goes16_two_km']):
                    group = ds.groups['goes16_two_km']
                    if inp == 'C11':
                        m = da.nanmean(da.from_array(group['C11']))
                        s = da.nanstd(da.from_array(group['C11']))
                        m, s, = da.compute(m, s)
                        mean_C11.append(m)
                        std_C11.append(s)
                    if inp == 'C13':
                        m = da.nanmean(da.from_array(group['C13']))
                        s = da.nanstd(da.from_array(group['C13']))
                        m, s, = da.compute(m, s)
                        mean_C13.append(m)
                        std_C13.append(s)
                    if inp == 'C14':
                        m = da.nanmean(da.from_array(group['C14']))
                        s = da.nanstd(da.from_array(group['C14']))
                        m, s, = da.compute(m, s)
                        mean_C14.append(m)
                        std_C14.append(s)
                    if inp == 'C15':
                        m = da.nanmean(da.from_array(group['C15']))
                        s = da.nanstd(da.from_array(group['C15']))
                        m, s, = da.compute(m, s)
                        mean_C15.append(m)
                        std_C15.append(s)
        ds.close()

    print('Final Calculations:')
    all_percents = da.from_array(all_percents)
    mn_perc = da.nanmean(all_percents)
    min_perc = da.min(all_percents)
    max_perc = da.max(all_percents)
    #med = da.median(all_percents).compute()
    mn_c02 = da.nanmean(da.from_array(mean_C02))
    std_c02 = da.nanmean(da.from_array(std_C02))
    mn_c05 = da.nanmean(da.from_array(mean_C05))
    std_c05 = da.nanmean(da.from_array(std_C05))
    mn_c13 = da.nanmean(da.from_array(mean_C13))
    std_c13 = da.nanmean(da.from_array(std_C13))
    mn_c14 = da.nanmean(da.from_array(mean_C14))
    std_c14 = da.nanmean(da.from_array(std_C14))
    mn_c15 = da.nanmean(da.from_array(mean_C15))
    std_c15 = da.nanmean(da.from_array(std_C15))
    mn_c11 = da.nanmean(da.from_array(mean_C11))
    std_c11 = da.nanmean(da.from_array(std_C11))
    mn_sa = da.nanmean(da.from_array(mean_sa))
    std_sa = da.nanmean(da.from_array(std_sa))
    mn_sz = da.nanmean(da.from_array(mean_sz))
    std_sz = da.nanmean(da.from_array(std_sz))
    mn_perc, min_perc, max_perc = da.compute(mn_perc, min_perc, max_perc)
    mn_c02, std_c02, mn_c05, std_c05, mn_c13, std_c13, mn_c14, std_c14, mn_c15, std_c15, mn_c11, std_c11, mn_sa, std_sa, mn_sz, std_sz = da.compute(mn_c02, std_c02, mn_c05, std_c05, mn_c13, std_c13, mn_c14, std_c14, mn_c15, std_c15, mn_c11, std_c11, mn_sa, std_sa, mn_sz, std_sz)

    print('mean perc', mn_perc)
    print('min perc', min_perc)
    print('max perc', max_perc)
    #print('median perc', med)
    print('mean C02', mn_c02)
    print('std C02', std_c02)
    print('mean C05', mn_c05)
    print('std C05', std_c05)
    print('mean C13', mn_c13)
    print('std C13', std_c13)
    print('mean C11', mn_c11)
    print('std C11', std_c11)
    print('mean C14', mn_c14)
    print('std C14', std_c14)
    print('mean C15', mn_c15)
    print('std C15', std_c15)
    print('mean sa', mn_sa)
    print('std sa', std_sa)
    print('mean sz', mn_sz)
    print('std sz', std_sz)
    print('Process Complete')

    # print('Process complete. Stats:')
    # print('mean perc', np.mean(all_percents))
    # print('min perc', np.min(all_percents))
    # print('max perc', np.max(all_percents))
    # print('median perc', np.median(all_percents))
    # print('mean C02', np.nanmean(mean_C02))
    # print('std C02', np.nanmean(std_C02))
    # print('mean C05', np.nanmean(mean_C05))
    # print('std C05', np.nanmean(std_C05))
    # print('mean C13', np.nanmean(mean_C13))
    # print('std C13', np.nanmean(std_C13))
    # print('mean C11', np.nanmean(mean_C11))
    # print('std C11', np.nanmean(std_C11))
    # print('mean C14', np.nanmean(mean_C14))
    # print('std C14', np.nanmean(std_C14))
    # print('mean C15', np.nanmean(mean_C15))
    # print('std C15', np.nanmean(std_C15))