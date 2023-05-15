from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SATPY_CONFIG_PATH'] = '/home/sbradshaw/ci_ltg/PPP_config/'
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.models as models
from tensorflow.keras.models import Model, save_model, load_model
import tensorflow.keras.layers as layers

import json
import zarr
from netCDF4 import Dataset
from datetime import datetime, timedelta
from ABIcmaps import irw_ctc
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from glob import glob
import ams_utils
from scipy.ndimage import zoom
import innvestigate
from matplotlib.lines import Line2D
import argparse
import collect_data
import shutil
import utils
from solar_angles import solar_angles
from sat_zen_angle import sat_zen_angle
import pickle
from earth_to_fgf import earth_to_fgf
import collections
import keras_metrics
from pyproj import Proj
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyresample as pr
import pygrib as pg
from metpy.plots import ctables
import cmasher as cmr
from satpy import Scene
from satpy.writers import get_enhanced_image

sys.path.insert(0, '/home/sbradshaw/ci_ltg')
PWD = os.environ['PWD'] + '/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Takes a list of lats/lons/datetimes and plots specified options')
    parser.add_argument('-m', '--model', help='CNN model used for predictions.', type=str, nargs=1,
                        default=['/apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210610/fit_conv_model.h5'])
    parser.add_argument('-mconfig', '--model_config_file',
                        help='This file contains mean and SD of channels (a pickle file).', type=str, nargs=1,
                        default=['/apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210610/model_config.pkl'])
    parser.add_argument('-a', '--abidir',
                        help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/', \
                        default=['/arcdata/goes/grb/goes16/'], type=str, nargs=1)
    parser.add_argument('-r', '--radardir', help='Root directory for radar netcdfs.',
                        default=['/apollo/grain/saved_ci_ltg_data/radar'], nargs=1, type=str)
    parser.add_argument('-l', '--local_dir', help='Local directory for copying files.', type=str, nargs=1,
                        default=['temporary_dir/'])
    parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/apollo/grain/saved_ci_ltg_data/MSthesis'])
    parser.add_argument('-RGB', '--make_RGB', help='Make RGB images with timestamps.', action="store_true")
    parser.add_argument('-preds', '--make_preds', help='Make and plot model predictions.', action="store_true")
    parser.add_argument('-DTfile', '--DTfilelist', help='A list of lats/lons/datetimes. See test_point_plot.config',
                        type=str, nargs=1, default=['nofile.config'])
    parser.add_argument('-hs', '--zoom_hs', help='If zooming, you may want to specify a new hs',
                        type=int, nargs=1)

    args = parser.parse_args()

    ###Open and get information from input
    local_dir = args.local_dir[0]
    model = args.model[0]
    model_config_file = args.model_config_file[0]
    rootoutdir = args.outdir[0]
    rootdatadir = args.abidir[0]
    rootradardir = args.radardir[0]
    utils.mkdir_p(rootoutdir)
    exception_files = []
    bad_data_files = []

    conv_model = load_model(args.model[0], compile=False)
    mc = pickle.load(open(args.model_config_file[0], 'rb'))
    C02mean = mc['scaling_values']['mean']['C02']
    C02stddev = mc['scaling_values']['stddev']['C02']
    C05mean = mc['scaling_values']['mean']['C05']
    C05stddev = mc['scaling_values']['stddev']['C05']
    C13mean = mc['scaling_values']['mean']['C13']
    C13stddev = mc['scaling_values']['stddev']['C13']

    goesnc = Dataset('/home/sbradshaw/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)

    if (args.make_RGB or args.make_preds):
        # Getting info config list
        if (args.DTfilelist[0]):
            file1 = open(args.DTfilelist[0], 'r')
            lines = file1.readlines()
            count = 0
            center_lats = [];
            center_lons = [];
            dts = []
            test_lats = []
            test_lons = []
            count = 0
            if(args.zoom_hs):
                hs = args.zoom_hs[0]
            else:
                hs = 200
            for line in lines:
                if (count == 0):
                    latlon = True if (line[0] == 'L') else False
                    count += 1
                    continue

                if (line[0] == '#'): continue

                if (latlon):
                    parts = line.split()
                    center_lats.append(float(parts[0]));
                    center_lons.append(float(parts[1]));
                    dts.append(parts[2])
                    test_lats.append(float(parts[3]))
                    test_lons.append(float(parts[4]))
                else:
                    parts = line.split()
                    ind.append(int(parts[0]))

                count += 1
        else:
            print('Please provide config file with desired datetimes and lat/lons of interest.')
            sys.exit()

        print(dts)
        print(center_lats)
        print(center_lons)

        print('For each date we do stuff.')
        for n in range(len(dts)):
            topdt = datetime.strptime(dts[n], '%Y%m%d-%H%M')
            print(topdt)
            #enddt = topdt + timedelta(hours=25)
            satdir = topdt.strftime(rootdatadir + '%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
            radardir = topdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
            fileloc = os.path.join(PWD, local_dir)
            center_lon = center_lons[n]
            center_lat = center_lats[n]
            test_lon = test_lons[n]
            test_lat = test_lats[n]
            utils.mkdir_p(fileloc)
            # Copy ABI data into a local directory (avoids messing with stored files)
            abi_channels = ['02', '05', '11', '13', '14', '15']
            print('Copying satellite netcdfs...')
            for ch in abi_channels:
                netcdfpattern = topdt.strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
                sat_netcdfs_sorted = np.sort(glob(netcdfpattern))
                sat_netcdfs = sat_netcdfs_sorted

                # Copy satellite netcdfs
                print(len(sat_netcdfs))
                for i in range(len(sat_netcdfs)):
                    # copy file from satdir
                    try:
                        shutil.copy(sat_netcdfs[i], fileloc)
                    except (PermissionError, OSError, IOError) as err:
                        print(traceback.format_exc())
                        print('Moving on to next sat DT...')
                        continue
            print('Satellite netcdf copying complete.')

            netcdfpattern_C02 = topdt.strftime(fileloc + 'OR_ABI-L1b-RadC-M*C02_G16_s%Y*')
            sat_netcdfs = np.sort(glob(netcdfpattern_C02))

            # Avoid out of bounds center_lat and center_lon
            if center_lon > -74:
                center_lon = -74
            elif center_lon < -117:
                center_lon = -117
            elif center_lat > 43:
                center_lat = 43
            elif center_lat < 32:
                center_lat = 32

            print('Sat_netcdf length', len(sat_netcdfs))
            for i in range(len(sat_netcdfs)):
                time_loop_start = time.time()
                print('Loop number', i)
                print('Of:', len(sat_netcdfs))
                sat_netcdf = sat_netcdfs[i]

                basefile = os.path.basename(sat_netcdf)
                parts = basefile.split('_')
                separator = '_'
                year_day_time = parts[3][1:-1]
                dt_start = datetime.strptime(year_day_time, '%Y%j%H%M%S')
                timestamp = str(dt_start)
                print(timestamp)

                netcdfpattern = dt_start.strftime(fileloc + 'OR_ABI-L1b-RadC-M*C*_G16_s%Y%j%H%M*')
                filenames = glob(netcdfpattern)

                ###Scene definition and cropping
                scn = Scene(reader='abi_l1b', filenames=filenames)
                try:
                    scn.load(['C13', 'C15', 'C11', 'C14', 'C02', 'C05'], calibrations=['brightness_temperature',
                                                                                       'brightness_temperature',
                                                                                       'brightness_temperature',
                                                                                       'brightness_temperature',
                                                                                       'reflectance',
                                                                                       'reflectance'])
                except:
                    exceptions_files.append(dt_start)
                    continue

                try:
                    x_idx, y_idx = scn['C13'].attrs['area'].get_xy_from_lonlat(center_lon, center_lat)
                except KeyError:
                    exception_files.append(dt_start)
                    continue

                xmin_index = x_idx - hs
                xmax_index = x_idx + hs - 1
                ymax_index = y_idx - hs
                ymin_index = y_idx + hs - 1
                x_coords, y_coords = scn['C13'].attrs['area'].get_proj_vectors()
                xmin = x_coords[xmin_index]
                xmax = x_coords[xmax_index]
                ymin = y_coords[ymin_index]
                ymax = y_coords[ymax_index]

                crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
                x_crop_coords, y_crop_coords = crop_scn['C13'].attrs['area'].get_proj_vectors()
                x_test_idx, y_test_idx = crop_scn['C13'].attrs['area'].get_xy_from_lonlat(test_lon, test_lat)
                x_test = x_crop_coords[x_test_idx]
                y_test = y_crop_coords[y_test_idx]


                try:
                    C13vals = crop_scn['C13'].values
                    print('max/min C13', np.max(C13vals), np.min(C13vals))
                    C15vals = crop_scn['C15'].values
                    C11vals = crop_scn['C11'].values
                    C14vals = crop_scn['C14'].values
                    C02vals = crop_scn['C02'].values
                    print('max/min C02vals', np.max(C02vals), np.max(C02vals))
                    C05vals = crop_scn['C05'].values
                    print('max/min C05vals', np.max(C05vals), np.max(C05vals))
                except KeyError:
                    exception_files.append(dt_start)

                if (np.isnan(np.min(C13vals)) or np.isnan(np.max(C13vals)) or np.isnan(np.min(C15vals)) or
                        np.isnan(np.max(C15vals)) or np.isnan(np.min(C11vals)) or np.isnan(np.max(C11vals)) or
                        np.isnan(np.min(C14vals)) or np.isnan(np.max(C14vals)) or np.isnan(np.min(C02vals)) or
                        np.isnan(np.max(C02vals)) or np.isnan(np.min(C05vals)) or np.isnan(np.max(C05vals))):
                    bad_data_files.append(dt_start)

                sat_grid = crop_scn['C05'].attrs['area']

                if (args.make_RGB):
                    time_RGB = time.time()
                    IRoutdir = dt_start.strftime(os.path.join(rootoutdir, 'RGBimages/%Y/%Y_%m_%d_%j/IR'))
                    VISoutdir = dt_start.strftime(os.path.join(rootoutdir, 'RGBimages/%Y/%Y_%m_%d_%j/VIS'))
                    #COMBoutdir = dt_start.strftime(os.path.join(rootoutdir, 'RGBimages/%Y/%Y_%m_%d_%j/VISandIR'))
                    utils.mkdir_p(IRoutdir)
                    utils.mkdir_p(VISoutdir)
                    #utils.mkdir_p(COMBoutdir)
                    IRfilepath = os.path.join(IRoutdir, dt_start.strftime('GOES16_IR_%Y_%m_%d_%H_%M_%j.png'))
                    VISfilepath = os.path.join(VISoutdir, dt_start.strftime('GOES16_VIS_%Y_%m_%d_%H_%M_%j.png'))
                    # IR False Color Image:
                    IRcrop_scn = crop_scn
                    IRcrop_scn.load(['IRCloudPhaseFC'])
                    IRresamp = IRcrop_scn.resample(resampler='native')
                    fig = plt.figure(figsize=(12, 12))
                    IRimg_data = get_enhanced_image(IRresamp['IRCloudPhaseFC']).data
                    IRimg_data.plot.imshow(rgb='bands')
                    plt.title((' ').join([timestamp, 'UTC']))
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.xticks([])
                    plt.yticks([])
                    # ax.set_title((' ').join([timestamp, 'UTC']), fontsize=12)
                    #plt.savefig(IRfilepath, bbox_inches='tight')
                    plt.close(fig)

                    # IRresamp.save_dataset(writer='simple_image', dataset_id='IRCloudPhaseFC', filename=IRfilepath)
                    # Visible False Color Image:
                    VIScrop_scn = crop_scn
                    VIScrop_scn.load(['DayCloudPhaseFC'])
                    VISresamp = VIScrop_scn.resample(resampler='native')
                    fig = plt.figure(figsize=(12, 12))
                    VISimg_data = get_enhanced_image(VISresamp['DayCloudPhaseFC']).data
                    VISimg_data.plot.imshow(rgb='bands')
                    # VISresamp.save_dataset(writer='simple_image', dataset_id='DayCloudPhaseFC', filename=VISfilepath)
                    plt.title((' ').join([timestamp, 'UTC']))
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.xticks([])
                    plt.yticks([])
                    # ax.set_title((' ').join([timestamp, 'UTC']), fontsize=12)
                    plt.savefig(VISfilepath, bbox_inches='tight')
                    print('RGB images created.')
                    print('Time to create RGB images (s):', time.time() - time_RGB)
                    plt.close('all')
                else:
                    print('RGB images were not created.')

                if (args.make_preds):

                    ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
                    rad_end = dt_start + timedelta(minutes=5)

                    radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                                glob(os.path.join(radardir, '*.netcdf'))}
                    radfilelist = [radfiles[x] for x in radfiles if dt_start < x < rad_end]

                    ###Get general information from the radar files (should be the same for all the files)
                    dataset_rad = Dataset(radfilelist[0], 'r')
                    nlat = len(dataset_rad.dimensions['Lat'])
                    nlon = len(dataset_rad.dimensions['Lon'])
                    NW_lat = getattr(dataset_rad, 'Latitude')
                    NW_lon = getattr(dataset_rad, 'Longitude')
                    dy = getattr(dataset_rad, 'LonGridSpacing')
                    dx = getattr(dataset_rad, 'LatGridSpacing')
                    radardataset = dataset_rad.variables['Reflectivity_-10C'][:, :]
                    dataset_rad.close()

                    ###Remap radar data to same domain as satellite data
                    radar_lons = np.zeros((nlat, nlon))
                    radar_lats = np.zeros((nlat, nlon))
                    for ii in range(nlat):
                        radar_lats[ii, :] = NW_lat - ii * dy
                    for ii in range(nlon):
                        radar_lons[:, ii] = NW_lon + ii * dx
                    radar_grid = pr.geometry.GridDefinition(lons=radar_lons, lats=radar_lats)

                    remapped_radar = pr.kd_tree.resample_nearest(radar_grid, radardataset, sat_grid,
                                                                 radius_of_influence=10000, fill_value=-999)

                    state_borders = cfeature.NaturalEarthFeature(category='cultural',
                                                                 name='admin_1_states_provinces_lakes', scale='50m',
                                                                 facecolor='none')
                    NWS_cmap = ctables.registry.get_colortable('NWSStormClearReflectivity')
                    NWS_cmap_sub = cmr.get_sub_cmap(NWS_cmap, 0.51, 0.8)

                    x2km, y2km = VISresamp['C13'].attrs['area'].get_proj_vectors()
                    x1km, y1km = VISresamp['C05'].attrs['area'].get_proj_vectors()
                    xhalfkm, yhalfkm = VISresamp['C02'].attrs['area'].get_proj_vectors()

                    xx2km, yy2km = np.meshgrid(x2km, y2km)
                    xx1km, yy1km = np.meshgrid(x1km, y1km)
                    xxhalfkm, yyhalfkm = np.meshgrid(xhalfkm, yhalfkm)

                    C02vals = crop_scn['C02'].values
                    C02norm_vals = (crop_scn['C02'].values - C02mean) / C02stddev
                    C05norm_vals = (crop_scn['C05'].values - C05mean) / C05stddev
                    C13norm_vals = (crop_scn['C13'].values - C13mean) / C13stddev

                    C13norm = np.zeros((1, C13norm_vals.shape[0], C13norm_vals.shape[1], 1))
                    C13norm[0, :, :, 0] = C13norm_vals
                    C02norm = np.zeros((1, C02norm_vals.shape[0], C02norm_vals.shape[1], 1))
                    C02norm[0, :, :, 0] = C02norm_vals
                    C05norm = np.zeros((1, C05norm_vals.shape[0], C05norm_vals.shape[1], 1))
                    C05norm[0, :, :, 0] = C05norm_vals

                    xmin = np.min(x2km);
                    xmax = np.max(x2km);
                    ymin = np.min(y2km);
                    ymax = np.max(y2km)
                    print('max and mins:')
                    print('x', xmin, xmax)
                    print('y', ymin, ymax)

                    print('pred in shapes of channels:', C02norm.shape, C05norm.shape, C13norm.shape)
                    pred_in = [C02norm, C05norm, C13norm]
                    new_preds = conv_model.predict(pred_in)
                    try:
                        print('newpreds prior', new_preds.shape)
                    except:
                        print('newpreds prior len', len(new_preds))
                    new_preds = new_preds[0]
                    print('new_preds shape', new_preds.shape)
                    print('new_preds of pix with indx', new_preds[x_test_idx, y_test_idx])

                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_axes([0, 0, 1, 1], projection=geoproj)

                    ax.set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
                    ax.add_feature(cfeature.BORDERS)
                    ax.add_feature(state_borders, edgecolor='tan')

                    print('max ch02', np.max(C02vals))
                    print(xxhalfkm.shape, yyhalfkm.shape, C02vals.shape)
                    visimg = ax.pcolormesh(xxhalfkm, yyhalfkm, C02vals, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
                    cbaxes = fig.add_axes([-0.01, -0.05, 0.27, 0.03])  # left bottom width height
                    cbar = plt.colorbar(visimg, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0, ax=ax,
                                        ticks=np.arange(0, 100, 10), cax=cbaxes)
                    cbar.ax.tick_params(labelsize=12)
                    cbar.set_label('ABI 0.64-$\mu$m reflectance [%]', fontsize=14)

                    radimg_10C = ax.pcolormesh(xxhalfkm, yyhalfkm, ma.masked_less(zoom(remapped_radar, 2, order=0), 10),
                                               cmap=NWS_cmap_sub, norm=Normalize(-1, 80), linewidth=0, antialiased=True,
                                               alpha=0.7)
                    cbaxes2 = fig.add_axes([0.28, -0.05, 0.27, 0.03])  # left bottom width height
                    cbar2 = plt.colorbar(radimg_10C, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                                         ax=ax, ticks=np.arange(10, 80, 10), cax=cbaxes2)
                    cbar2.ax.tick_params(labelsize=12)
                    cbar2.set_label('Radar Reflectivity at -10C [dBz]', fontsize=14)
                    cbar2.set_alpha(1.0)
                    cbar2.draw_all()

                    colors = ['#FF8300', '#9700FF', '#00FFFB', '#FF00EC', '#0015FF']
                    values = [0.1, 0.25, 0.5, 0.75, 0.9]
                    #linewidths = [1.6, 1.6, 1.7, 1.8, 2]
                    linewidths = [2.3, 2.4, 2.5, 2.8, 3]
                    labels = [str(int(v * 100)) + '%' for v in values]
                    contour_img = ax.contour(xxhalfkm, yyhalfkm, zoom(new_preds, 2, order=0), values, colors=colors,
                                             linewidths=linewidths)  # , cmap=plt.get_cmap('cividis'))
                    custom_lines = [Line2D([0], [0], color=c, lw=lw * 2) for c, lw in zip(colors, linewidths)]
                    leg2 = fig.legend(custom_lines, labels, fontsize=12, loc='lower right', handlelength=1.5,
                                      ncol=len(labels), title = 'Probability of Convection in 60 min', title_fontsize=14, bbox_to_anchor=(0.81, -0.07, 0.2, 0.03))
                    #ctext = plt.text(0.65, -0.09, 'Probability of Convection in 60 min', fontsize=14,
                    #                 transform=ax.transAxes)

                    pt = ax.plot(x_test, y_test, 'k*', markersize=16)

                    ax.set_title((' ').join(['GOES-East', 'CONUS', timestamp, 'UTC']), fontsize=20)
                    # ax.set_title((' ').join(['MRMS', 'CONUS', timestamp, 'UTC']), fontsize=16)
                    filepath = os.path.join(rootoutdir, dt_start.strftime('point_test_%Y_%m_%d_%H_%M_%j.png'))
                    print('Filepath for figure: ', filepath)
                    plt.savefig(filepath, bbox_inches='tight')
                    plt.close('all')
                    print('Predictions made and plotted.')

                VIScrop_scn.unload()
                crop_scn.unload()
                scn.unload()
                IRcrop_scn.unload()

            # Removes copied data to avoid clutter and taking up space
            try:
                if os.path.exists(fileloc) and os.path.isdir(fileloc):
                    shutil.rmtree(fileloc)
            except:
                print('Removing with rmtree did not work.')
                if os.path.exists(fileloc) and os.path.isdir(fileloc):
                    remove_pattern = fileloc + '*'
                    print(remove_pattern)
                    files_to_delete = glob(remove_pattern)
                    for f in files_to_delete:
                        os.remove(f)

            print('Finished this one.')





