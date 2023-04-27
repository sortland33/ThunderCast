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
import dask.array as da
import dask.dataframe as dd
import argparse
from dask.distributed import Client
import dask
import traceback
from dask import delayed
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

if __name__ == "__main__":
  #Make a list of files for training, validation, and testing
  parser = argparse.ArgumentParser(description='Selects files for deep learning model.')
  parser.add_argument('-d', '--data_dir', help='Data directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/data_netcdfs'])
  parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/data_lists/'])
  parser.add_argument('-y', '--years', help='years to use in the train/val/test files', type=str, nargs=1,
                        default=['2019', '2020', '2021'])
  parser.add_argument('-m', '--months', help='Output directory.', type=str, nargs=1,
                      default=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
  parser.add_argument('-v', '--val', help='If you want validation file instead of regular.', action="store_true")
  parser.add_argument('-t', '--test', help='If you want test file instead of regular.', action="store_true")
  args = parser.parse_args()

  keys = ['northwest', 'west', 'west_north_central', 'southwest', 'east_north_central', 'south', 'southeast',
          'central', 'northeast', 'oconus_west', 'oconus_east', 'oconus_north', 'oconus_south']

  limit_region = False
  limited_keys = ['south', 'southeast']
  limit_pos_perc = False
  limit_pos_threshold = 35
  day_only = False
  main_months = ['05', '06', '07', '08', '09']
  reduce = True

  if args.val or args.test:
    reduce = True
    if args.val:
      main_months = ['03', '04', '05', '06', '07', '08', '09']
    if args.test:
      main_months = ['03', '04', '05', '06', '07', '08', '09', '10']

  if args.val or args.test: #If we don't want all the files (i.e. all the files would take too long to train or something)
    reduce_factor = 3.5
  else:
    reduce_factor = 2

  pos_files_to_write = []
  neg_files_to_write = []
  if limit_region:
    keys = limited_keys
  val_years = ['2020']
  test_years = ['2021']
  for year in args.years: #get only the years for the type of file needed. See parser -y
    if args.val and (year not in val_years):
      continue
    elif args.test and (year not in test_years):
      continue
    elif not args.val and (year in val_years):
      continue
    elif not args.test and (year in test_years):
      continue
    else:
      file_base = '/ships19/grain/convective_init/data/'
      pos_file_counts = {key: [] for key in keys}
      day_counts = {key: 0 for key in keys}
      pos_region_file_dirs = {key: [] for key in keys}
      neg_region_file_dirs = {key: [] for key in keys}
      for month in args.months:
        for day_dir in glob(file_base + year + '/' + month +'/*'):
          for region_dir in glob(day_dir + '/*'):
            region = os.path.basename(region_dir)
            if region not in keys:
              continue
            file_selection = glob(region_dir + '/positive_cases/*')
            number_of_positives = len(file_selection)
            file_count = pos_file_counts[region]
            file_count.append(number_of_positives)
            pos_file_counts[region] = file_count
            day_counts[region] = day_counts[region] + 1
            pos_region_file_dir_list = pos_region_file_dirs[region]
            pos_region_file_dir_list.append(region_dir)
            pos_region_file_dirs[region] = pos_region_file_dir_list
            neg_region_file_dir_list = neg_region_file_dirs[region]
            neg_region_file_dir_list.append(region_dir)
            neg_region_file_dirs[region] = neg_region_file_dir_list

      pos_counts_per_region = {key: np.sum(pos_file_counts[key]) for key in keys}
      pos_conus = []
      pos_oconus = []
      for key in pos_counts_per_region:
        pos_region_directories = pos_region_file_dirs[key]
        neg_region_directories = neg_region_file_dirs[key]
        if key[:4] == 'ocon':
          for direct in pos_region_directories:
            pos_files = glob(direct + '/positive_cases/*')
            if len(pos_files) == 0:
              continue
            else:
              pos_oconus.append(len(pos_files))
        else:
          for direct in pos_region_directories:
            pos_files = glob(direct + '/positive_cases/*')
            if len(pos_files) == 0:
              continue
            else:
              pos_conus.append(len(pos_files))

      num_pos_conus = np.sum(pos_conus)
      num_pos_oconus = np.sum(pos_oconus)

      grab_all_oconus = True
      if num_pos_oconus > (num_pos_conus/2):
        grab_all_oconus = False

      for key in keys:
        pos_region_directories = pos_region_file_dirs[key]
        neg_region_directories = neg_region_file_dirs[key]
        for direct in pos_region_directories:
          parts = direct.split('/')
          dir_month = parts[6]
          pos_files = glob(direct + '/positive_cases/*')
          neg_files = glob(direct + '/negative_cases/*')
          if reduce:
            if dir_month in main_months:
              #We don't need all the files so we reduce here:
              npos = int(len(pos_files)/reduce_factor)
              nneg = int(len(neg_files)/reduce_factor)
              pos_files = random.sample(pos_files, npos)
              neg_files = random.sample(neg_files, nneg)
          if key[:4] == 'ocon':
            # We don't need all OCONUS files because focus is CONUS so we reduce here unless specified
            if grab_all_oconus:
              pos_files_to_write = pos_files_to_write + pos_files
              neg_files_to_write = neg_files_to_write + neg_files
            else:
              pos_num_to_grab = int(np.round((len(pos_files)/2), decimals=0))
              neg_num_to_grab = pos_num_to_grab
              pos_files_to_write = pos_files_to_write + random.sample(pos_files, pos_num_to_grab)
              neg_files_to_write = neg_files_to_write + random.sample(neg_files, neg_num_to_grab)
          else:
            pos_files_to_write = pos_files_to_write + pos_files
            neg_files_to_write = neg_files_to_write + neg_files

  if day_only: #if we want only datetime files, get only files with less than or equal to 85 for the zenith angle
    new_pos = []
    new_neg = []
    for x in pos_files_to_write:
      angle = solar_angles(datetime(int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1]),
                                        int(x.split('/')[-1].split('_')[2]), int(x.split('/')[-1].split('_')[3]),
                                        int(x.split('/')[-1].split('_')[4])),
                               float(str(x.split('/')[-1].split('_')[6][:-2] + '.' + x.split('/')[-1].split('_')[6][-2:])),
                               float(str(x.split('/')[-1].split('_')[7][:-2] + '.' + x.split('/')[-1].split('_')[7][-2:])))[0][0]
      if angle <= 85:
        new_pos.append(x)
    for x in neg_files_to_write:
      angle = solar_angles(datetime(int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1]),
                                        int(x.split('/')[-1].split('_')[2]), int(x.split('/')[-1].split('_')[3]),
                                        int(x.split('/')[-1].split('_')[4])),
                               float(str(x.split('/')[-1].split('_')[6][:-2] + '.' + x.split('/')[-1].split('_')[6][-2:])),
                               float(str(x.split('/')[-1].split('_')[7][:-2] + '.' + x.split('/')[-1].split('_')[7][-2:])))[0][0]
      if angle <= 85:
        new_neg.append(x)
    pos_files_to_write=new_pos
    neg_files_to_write=new_neg

  print('Length/Number of Positive Files: ', len(pos_files_to_write))
  random.shuffle(neg_files_to_write)
  neg_files_to_write = neg_files_to_write[0::100] #downsample number of negative samples for class imbalance
  #neg_files_to_write = []
  print('Length/Number of Negative Files: ', len(neg_files_to_write))
  all_files_to_write = pos_files_to_write + neg_files_to_write
  random.shuffle(all_files_to_write)

  #open file to write to
  if args.val:
    filename = '/ships19/grain/convective_init/data_lists/validation/val_files.txt'
  elif args.test:
    filename = '/ships19/grain/convective_init/data_lists/testing/test_files.txt'
  else:
    filename = '/ships19/grain/convective_init/data_lists/training/train_files.txt'
  f = open(filename, 'w')
  for file in all_files_to_write:
    f.write(file + '\n')
  f.close()
  print('Training/Validation/Testing File Created')