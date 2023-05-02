#Author: Stephanie M. Ortland
import os, sys
from glob import glob
import torch
import torch.nn as nn
import psutil
import pathlib
from netCDF4 import Dataset
from datetime import datetime,timedelta
from time import time
import numpy as np
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
import errno

os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

#This file will create figures to display the number of sample regions in each month and the number of samples
# per region for training, validation, and testing.

def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise
    
if __name__ == "__main__":
    training_file_path = '/ships19/grain/convective_init/data_lists/training/train_files.txt'
    validation_file_path = '/ships19/grain/convective_init/data_lists/validation/val_files.txt'
    testing_file_path = '/ships19/grain/convective_init/data_lists/testing/test_files.txt'

    file_paths = {'Training (2019)':training_file_path, 'Validation (2020)':validation_file_path, 'Testing (2021)':testing_file_path}
    datasets = {}
    plot_dir = '/ships19/grain/convective_init/stats/dataset_all/'
    month_filepath = plot_dir + 'samples_per_month.png'
    region_filepath = plot_dir + 'samples_per_region.png'

    plt.style.use('tableau-colorblind10')
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    n = len(file_paths.keys())
    w = 0.3
    for i, key in enumerate(file_paths.keys()):
        file_path = file_paths[key]
        info = {'months':{'01':0, '02':0, '03':0, '04':0, '05':0, '06':0, '07':0, '08':0, '09':0, '10':0, '11':0, '12':0},
                'regions':{'northwest':0, 'west':0, 'west_north_central':0, 'southwest':0, 'east_north_central':0, 'south':0, 'southeast':0, 'central':0,'northeast':0, 'oconus':0}}
        with open(file_path, 'r') as data_files:
            files = data_files.read().splitlines()
        print('Number of files:', len(files))
        #files = files[:10]
        for file in files:
            parts = file.split('/')
            month = parts[-5]
            region = parts[-3]
            if 'oconus' in region:
                region = 'oconus'
            info['months'][month] = info['months'][month] + 1
            info['regions'][region] = info['regions'][region] + 1

        month_labels = info['months'].keys()
        xmonth = np.arange(len(month_labels))
        ymonth = info['months'].values()
        position = xmonth + (w*(1-n)/2) + i*w
        ax.bar(position, ymonth, width=w, label = key)

        region_labels = ['Northwest', 'West', 'West North Central', 'Southwest', 'East North Central', 'South', 'Southeast', 'Central', 'Northeast', 'OCONUS']
        xregion = np.arange(len(region_labels))
        print('region_labels_count', len(region_labels))
        yregion = info['regions'].values()
        position_reg = xregion + (w*(1-n)/2) + i*w
        rects = ax2.bar(position_reg, yregion, width=w, label=key)
        for j, rect in enumerate(rects):
            #if j == 0 or j == 1:
            height = rect.get_height()
            print('height', height)
            ax2.text(rect.get_x() + rect.get_width()/2., height+225, '%.1e' % int(height), ha='center',va='bottom', rotation=90, fontsize=8)

    ax.set_ylabel('Number of Patches')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4), useMathText=True)
    ax.set_xlabel('Month')
    ax.set_xticks(xmonth)
    ax.set_xticklabels(month_labels)
    #ax.set_title('Patches per Month')
    ax.title.set_fontsize(16)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.legend()
    fig.savefig(month_filepath, bbox_inches='tight', dpi=200)

    ax2.set_ylabel('Number of Patches')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(4,4), useMathText=True)
    ax2.set_xlabel('U.S. Climate Region')
    ax2.set_xticks(xregion)
    ax2.set_xticklabels(region_labels, rotation=45, ha='right')
    ax2.set_ylim(0, 4.3e4)
    #ax2.set_title('Patches per U.S. Climate Region')
    ax2.title.set_fontsize(16)
    ax2.xaxis.label.set_fontsize(14)
    ax2.yaxis.label.set_fontsize(14)
    ax2.legend()
    fig2.savefig(region_filepath, bbox_inches='tight', dpi=200)

