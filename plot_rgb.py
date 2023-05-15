#Author: Stephanie Bradshaw
#This file takes ABI data and uses Satpy to make RGB images. Images are saved in a folder called RGBimages.
#The data from each scene can also be saved if desired.
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
os.environ['SATPY_CONFIG_PATH'] = '/home/sortland/ci_ltg/PPP_config/'
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from glob import glob
from scipy.ndimage import zoom
from matplotlib.lines import Line2D
import argparse
from pyproj import Proj
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyresample as pr
import pygrib as pg
from satpy import Scene
from satpy.writers import get_enhanced_image
import errno

def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise

def get_abifile_lists(rootdatadir, dts):
    list = []
    abi_channels = ['02', '05', '11', '13', '14', '15'] #for this application we do not need all the channels
    for i in range(len(dts)):
        satdir = dts[i].strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        filenames = []
        for ch in abi_channels:
            netcdfpattern = dts[i].strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
            try:
                filenames.append(glob(netcdfpattern)[0])
            except IndexError:
                continue
        list.append(filenames)
    return list

def get_satpy_scene(abi_files):  # get satpy scene to be cropped to patches
    CHs = ['C13', 'C15', 'C11', 'C14', 'C02', 'C05']
    bad_data = False
    try:
        scn = Scene(reader='abi_l1b', filenames=abi_files)
        scn.load(CHs)  # Load the scene and check for data problems
    except:
        print('Scene would not load.')
        bad_data = True
        scn = None
        return scn, bad_data
    return scn, bad_data

if __name__ == "__main__":
    ###User Input
    date = '2022-07-01-14-01' #%Y-%m-%d-%H-%M
    time_delta =120
    center_lat = 28.517
    center_lon = -80.649
    hs = 159

    ###Open and get information from inputs
    rootoutdir = '/ships19/grain/convective_init/rgbs'
    rootdatadir = '/arcdata/goes/grb/goes16'
    exceptions_files = []
    bad_data_files = []

    #Just a file with some plotting projection information in it.
    goesnc = Dataset('/home/sortland/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)

    dts = []
    topdt = datetime.strptime(date, '%Y-%m-%d-%H-%M')
    print('start dt', topdt)
    enddt = topdt + timedelta(minutes=time_delta)
    print('end dt', enddt)
    satdir = topdt.strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
    netcdfpattern = satdir + '*C02*'
    satfiles = {datetime.strptime(os.path.basename(x).split('_')[3][1:-1], '%Y%j%H%M%S'): x for x in
                glob(netcdfpattern)}
    sat_netcdfs = [satfiles[x] for x in satfiles if topdt <= x <= enddt]
    dts = [x for x in satfiles if topdt <= x <= enddt]
    abifile_lists = get_abifile_lists(rootdatadir, dts)

    #For each datetime we plot what we want
    for n in range(len(abifile_lists)):
        abi_files = abifile_lists[n]
        dt = dts[n]
        timestamp = str(dt)
        scn, bad_data = get_satpy_scene(abi_files)
        center_x_idx, center_y_idx = scn['C05'].attrs['area'].get_array_indices_from_lonlat(center_lon, center_lat)

        try:
            x_idx, y_idx = scn['C05'].attrs['area'].get_array_indices_from_lonlat(center_lon, center_lat)
        except KeyError:
            exception_files.append(dt)
            continue

        #Specify domain of plots using the half size specified earlier.
        xmin_index = x_idx - hs
        xmax_index = x_idx + hs
        ymax_index = y_idx - hs
        ymin_index = y_idx + hs
        x_coords, y_coords = scn['C05'].attrs['area'].get_proj_vectors()
        xmin = x_coords[xmin_index]
        xmax = x_coords[xmax_index]
        ymin = y_coords[ymin_index]
        ymax = y_coords[ymax_index]

        #Crop the scene to be the limited domain based on mins and maxes and half size
        crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])

        #Data checking:
        try:
            C13vals = crop_scn['C13'].values
            C15vals = crop_scn['C15'].values
            C11vals = crop_scn['C11'].values
            C14vals = crop_scn['C14'].values
            C02vals = crop_scn['C02'].values
            C05vals = crop_scn['C05'].values
        except KeyError:
            exception_files.append(dt)
            continue

        if(np.isnan(np.min(C13vals)) or np.isnan(np.max(C13vals)) or np.isnan(np.min(C15vals)) or
                np.isnan(np.max(C15vals)) or np.isnan(np.min(C11vals)) or np.isnan(np.max(C11vals)) or
                np.isnan(np.min(C14vals)) or np.isnan(np.max(C14vals)) or np.isnan(np.min(C02vals)) or
                np.isnan(np.max(C02vals)) or np.isnan(np.min(C05vals)) or np.isnan(np.max(C05vals))):
            bad_data_files.append(dt)
            continue

        #Get plotting grid
        sat_grid = crop_scn['C05'].attrs['area']

        #have the scene load the data of interest in a way we want to display it (RGB composites defined in PPP_config)
        #IR images
        IRcrop_scn = crop_scn
        IRcrop_scn.load(['IRCloudPhaseFC'])
        IRresamp = IRcrop_scn.resample(resampler='native')

        #Visible channel images
        VIScrop_scn = crop_scn
        VIScrop_scn.load(['DayCloudPhaseFC'])
        VISresamp = VIScrop_scn.resample(resampler='native')

        #Specify output file locations. Change as needed
        IRoutdir = dt.strftime(os.path.join(rootoutdir, '%Y/%Y_%m_%d_%j/IR'))
        VISoutdir = dt.strftime(os.path.join(rootoutdir, '%Y/%Y_%m_%d_%j/VIS'))

        mkdir_p(IRoutdir)
        mkdir_p(VISoutdir)

        IRfilepath = os.path.join(IRoutdir, dt.strftime('GOES16_IR_%Y_%m_%d_%H_%M_%j.png'))
        VISfilepath = os.path.join(VISoutdir, dt.strftime('GOES16_VIS_%Y_%m_%d_%H_%M_%j.png'))

        # Plot IR False Color Image:
        fig = plt.figure(figsize=(12,12))
        IRimg_data = get_enhanced_image(IRresamp['IRCloudPhaseFC']).data
        IRimg_data.plot.imshow(rgb='bands')
        plt.title((' ').join([timestamp, 'UTC']))
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        #If you want to save the RGB file:
        plt.savefig(IRfilepath,bbox_inches='tight')
        print('IR RGB images were created')
        plt.close(fig)
        #Uncomment code below to save the data as a netcdf:
        #IRresamp.save_dataset(writer='simple_image', dataset_id='IRCloudPhaseFC', filename=IRfilepath)

        # Visible False Color Image:
        fig = plt.figure(figsize=(12,12))
        VISimg_data = get_enhanced_image(VISresamp['DayCloudPhaseFC']).data
        VISimg_data.plot.imshow(rgb='bands')
        #Uncomment code below to save the data as a netcdf:
        #VISresamp.save_dataset(writer='simple_image', dataset_id='DayCloudPhaseFC', filename=VISfilepath)
        plt.title((' ').join([timestamp, 'UTC']))
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(VISfilepath, bbox_inches='tight')
        print('VIS RGB images created.')
        plt.close('all')

        #If you don't unload the scenes, the memory blows up when looping through multiple datetimes.
        VIScrop_scn.unload()
        crop_scn.unload()
        scn.unload()
        IRcrop_scn.unload()

    print('Files with exceptions or bad data:', exceptions_files, bad_data_files)
    print('RGBs plotted.')