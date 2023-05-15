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
from torchmetrics import Precision
from torchmetrics import AUROC
from torchmetrics import Recall
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
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy.ma as ma
from scipy.ndimage import zoom
import pyart
import matplotlib

if __name__ == "__main__":

    training_file_path = '/ships19/grain/convective_init/data_lists/training/y2019_train_files.txt'
    validation_file_path = '/ships19/grain/convective_init/data_lists/validation/y2019_val_files.txt'
    with open(training_file_path, 'r') as data_files:
        training_files = data_files.read().splitlines()
    with open(validation_file_path, 'r') as data_files:
        validation_files = data_files.read().splitlines()

    training_files = training_files[:100]
    for file in training_files:
        basefile = os.path.basename(file)
        parts = basefile.split('_')
        dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        lat = int(parts[-2])/100
        lon = int(parts[-3])/100
        timestamp = str(dt)
        title_str = timestamp + ' Lat: ' + str(lat) + ' Lon: ' + str(lon)
        name_parts = timestamp.split(' ')
        figure_path = '/ships19/grain/convective_init/data_testing/' + name_parts[0] + '_' + name_parts[1] + '_' + str(lat) + '_' + str(lon) + '.png'
        print(timestamp)

        root = Dataset(file, 'r+')
        group_2km = root.groups['goes16_two_km']
        #x_2km = group_2km['x']
        #y_2km = group_2km['y']
        ch13 = group_2km['C13']

        group_1km = root.groups['goes16_one_km']
        x_1km = group_1km['x']
        y_1km = group_1km['y']
        ch05 = group_1km['C05']

        group_halfkm = root.groups['goes16_half_km']
        x_halfkm = group_halfkm['x']
        y_halfkm = group_halfkm['y']
        ch02 = group_halfkm['C02']

        group_MRMS = root.groups['MRMS_one_km']
        #x_MRMS = group_MRMS['x']
        #y_MRMS = group_MRMS['y']
        radar = group_MRMS['reflectivity_60min_neg10C_max']

        #xx_2km, yy_2km = np.meshgrid(x_2km, y_2km)
        xx_1km, yy_1km = np.meshgrid(x_1km, y_1km)
        xx_halfkm, yy_halfkm = np.meshgrid(x_halfkm, y_halfkm)
        #xx_MRMS, yy_MRMS = np.meshgrid(x_MRMS, y_MRMS)

        CVD_cmap_all = matplotlib.cm.get_cmap('pyart_HomeyerRainbow')
        colors1 = CVD_cmap_all(np.linspace(0, 0.25, 98))
        colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 158))
        cmap_colors = np.vstack((colors1, colors2))
        CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)

        fig, axs = plt.subplots(2,2, figsize=(12, 8))
        img = axs[0,0].pcolormesh(xx_1km, yy_1km, zoom(ch13, 2, order=0), vmin=200, vmax=300, cmap=plt.get_cmap('Greys'))
        fig.colorbar(img, ax=axs[0,0])
        axs[0,0].set_title('C13')
        img2 = axs[0,1].pcolormesh(xx_1km, yy_1km, ch05, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
        fig.colorbar(img2, ax=axs[0,1])
        axs[0,1].set_title('C05')
        img3 = axs[1,0].pcolormesh(xx_halfkm, yy_halfkm, ch02, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
        fig.colorbar(img3, ax=axs[1,0])
        axs[1,0].set_title('C02')
        img4 = axs[1,1].pcolormesh(xx_1km, yy_1km, ma.masked_less(radar, 10), cmap=CVD_cmap,
                                   norm=Normalize(-1, 80),
                                   linewidth=0, antialiased=True, alpha=0.7)
        fig.colorbar(img4, ax=axs[1,1])
        axs[1,1].set_title('Max -10C Radar for +1 hour')
        fig.suptitle(title_str)
        plt.savefig(figure_path, bbox_inches='tight', dpi=200)
        plt.close('all')