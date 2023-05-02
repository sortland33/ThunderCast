#Author: Stephanie M. Ortland
import sys, os
from datetime import datetime, timedelta
import time
from glob import glob
import numpy as np
import shutil
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import shapely.geometry as sgeom
import matplotlib
from satpy import Scene
import pyresample as pr
import argparse
#import utils
import traceback
import gc
import random
import psutil
import errno
import math
import itertools
from solar_angles import solar_angles

#Define functions
def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise

#Sort the existing testing dataset into categories. Save the sorted lists in a txt file. 
if __name__ == "__main__":
  file_path = '/ships19/grain/convective_init/data_lists/testing/test_files.txt'
  save_nc_list = False
  if save_nc_list:
    new_filename = '/ships19/grain/convective_init/data_lists/testing/night_test_files_ncs.txt'
  else:
    new_filename = '/ships19/grain/convective_init/data_lists/testing/sublists/day_test_files_orig.txt'
  keys = ['northwest', 'west', 'west_north_central', 'southwest', 'east_north_central', 'south', 'southeast',
          'central', 'northeast', 'oconus_west', 'oconus_east', 'oconus_north', 'oconus_south']
  basept   = '/ships19/grain/convective_init/pt_tensors/test_files/inputs/'
  day_only = True
  night_only = False
  climate_region_split = False
  climate_regions_wanted = ['oconus_west', 'oconus_east', 'oconus_north', 'oconus_south']
  month_split = False
  months_wanted = [1]
  with open(file_path, 'r') as data_files:
    files = data_files.read().splitlines()

  new_test_files = []
  for file in files:
    parts = file.split('/')
    if save_nc_list:
      file_pt = file
    else:
      file_pt = basept + parts[-1][:-3] + '.pt'
    climate_region = parts[-3]
    other_info = parts[-1][:-3].split('_')
    lat = float(other_info[-2])/100
    lon = float(other_info[-3])/100
    year = int(other_info[0])
    month = int(other_info[1])
    day = int(other_info[2])
    hour = int(other_info[3])
    min = int(other_info[4])
    dt = datetime(year, month, day, hour, min)

    if day_only:
      angle = solar_angles(dt,lon,lat)
      if angle[0] <= 85:
        new_test_files.append(file_pt)
    if night_only:
      angle = solar_angles(dt, lon, lat)
      if angle[0] > 85:
        new_test_files.append(file_pt)
    if climate_region_split:
      if climate_region in climate_regions_wanted:
        if month_split:
          if dt.month in months_wanted:
            new_test_files.append(file_pt)
        else:
          new_test_files.append(file_pt)
    # if month_slit:
    #   if dt.month in months_wanted:
    #     new_test_files.append(file_pt)

  print('Length of files saved:', len(new_test_files))

  f = open(new_filename, 'w')
  for file in new_test_files:
    f.write(file + '\n')
  f.close()