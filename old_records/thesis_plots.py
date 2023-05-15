from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ['KERAS_BACKEND'] = 'tensorflow'
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0,'/home/sbradshaw/ci_ltg')
PWD = os.environ['PWD']+'/'
PWD = '/home/sbradshaw/ci_ltg/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Takes a list of lats/lons/datetimes and plots specified options')
    parser.add_argument('-m', '--model',help='CNN model used for predictions.',type=str,nargs=1,
                        default=['/apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210610/fit_conv_model.h5'])
    parser.add_argument('-mconfig', '--model_config_file', help='This file contains mean and SD of channels (a pickle file).',type=str,nargs=1,
                        default=['/apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210610/model_config.pkl'])
    parser.add_argument('-mversion', '--model_version', help='Provide which model you use.', type=str, nargs=1, default=['20210610'])
    parser.add_argument('-a','--abidir',help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/',\
                        default=['/arcdata/goes/grb/goes16/'],type=str,nargs=1)
    parser.add_argument('-r', '--radardir', help='Root directory for radar netcdfs.',
                        default=['/apollo/grain/saved_ci_ltg_data/radar'], nargs=1, type=str)
    parser.add_argument('-l', '--local_dir', help='Local directory for copying files.', type=str, nargs=1, default=['CopiedData_plot/'])
    parser.add_argument('-o','--outdir',help='Output directory.',type=str, nargs=1, default=['/apollo/grain/saved_ci_ltg_data'])
    parser.add_argument('-RGB', '--make_RGB', help='Make RGB images with timestamps.', action="store_true")
    parser.add_argument('-preds', '--make_preds', help='Make and plot model predictions.', action="store_true")
    #parser.add_argument('-subdt','--sub_DT',help='If you instead supply subtimes in a day use this option',action="store_true")
    #parser.add_argument('-LRPfile', '--LRPfilelist', help='A list of lats/lons/datetimes. See PLOT_LIST.config', type=str, nargs=1)
    parser.add_argument('-DTfile', '--DTfilelist', help='A list of lats/lons/datetimes. See plot_list_LRP.config', type=str, nargs=1)
    parser.add_argument('-z', '--zoom', help='If zooming to a feature use this.',
                        action='store_true')
    parser.add_argument('-fn', '--file_name',
                        help='What you want the file to be named.',
                        type=str, nargs=1, default=['testing_date.png'])
    parser.add_argument('-mm', '--multiple_models', help='If using multiple models us this and then specify the models in the DT file as an extra column.',
                        action='store_true')

    args = parser.parse_args()

    print('Starting document to make plot for thesis.')

    ###Open and get information from input
    local_dir = args.local_dir[0]
    #model = args.model[0]
    model_config_file = args.model_config_file[0]
    rootoutdir = args.outdir[0]
    rootdatadir = args.abidir[0]
    rootradardir = args.radardir[0]
    file_name = args.file_name[0]
    utils.mkdir_p(rootoutdir)
    exception_files = []
    bad_data_files = []
    rgbs = []



    if args.multiple_models:
        mult_models = []
        mc_files = []
        mversion = []
    else:
        model = args.model_version[0]
        conv_model = load_model(args.model[0], compile=False)
        mc = pickle.load(open(args.model_config_file[0], 'rb'))

        try:
            C02mean = mc['scaling_values']['mean']['C02']
            C02stddev = mc['scaling_values']['stddev']['C02']
            C02_present = True
        except:
            print('No C02')
            C02_present = False
        try:
            C05mean = mc['scaling_values']['mean']['C05']
            C05stddev = mc['scaling_values']['stddev']['C05']
            C05_present = True
        except:
            print('No C05')
            C05_present = False
        try:
            C13mean = mc['scaling_values']['mean']['C13']
            C13stddev = mc['scaling_values']['stddev']['C13']
            C13_present = True
        except:
            print('No C13')
            C13_present = False

    goesnc = Dataset('/home/sbradshaw/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)

    # Getting info config list
    if(args.make_RGB or args.make_preds):
        file1 = open(args.DTfilelist[0], 'r')
        lines = file1.readlines()
        count = 0
        center_lats = [];
        center_lons = [];
        dts = []
        count = 0
        if(args.zoom):
            #hs = 30
            #hs = 100
            hs = 120
        else:
            hs = 200
        for line in lines:
            if(count == 0):
                latlon = True if (line[0] == 'L') else False
                count += 1
                continue

            if (line[0] == '#'): continue

            if(latlon):
                parts = line.split()
                center_lats.append(float(parts[0]));
                center_lons.append(float(parts[1]));
                dts.append(parts[2])
                if(args.multiple_models):
                    mult_models.append(parts[3])
                    mc_files.append(parts[4])
                    mversion.append(parts[5])

            else:
                parts = line.split()
                ind.append(int(parts[0]))

            count += 1
    else:
        print('Please provide config file with desired datetimes and lat/lons of interest.')
        sys.exit()

    print('DTs: ', dts)

    nsamples = len(dts)

    print('Starting plot 1.')
    #for each DT we need to make an image and put it in a subplot. So nsamples is the number of subplots we want.
    if(nsamples % 2 == 0):
        #make 2 rows of plots
        numcols = int(nsamples/2)
        numrows = int(nsamples/numcols)
        print('numrows, numcols', numrows, numcols)
        if(args.multiple_models):
            fig, axes = plt.subplots(numrows, numcols, figsize=(14, 14), subplot_kw={'projection': geoproj})
        else:
            fig, axes = plt.subplots(numrows, numcols, figsize=(15, 10), subplot_kw={'projection': geoproj})
        dts_array = np.array(dts).reshape((numrows,numcols))
        center_lats = np.array(center_lats).reshape((numrows,numcols))
        center_lons = np.array(center_lons).reshape((numrows,numcols))
        if(args.multiple_models):
            mult_models = np.array(mult_models).reshape((numrows, numcols))
            mc_files = np.array(mc_files).reshape((numrows, numcols))
            mversion = np.array(mversion).reshape((numrows, numcols))
    else:
        print('You need to code for this')

    print('DTs reshaped: ', dts_array)

    for i in range(numrows):
        for j in range(numcols):
            print('i:', i)
            print('j:', j)
            topdt = datetime.strptime(dts_array[i,j], '%Y%m%d-%H%M')
            center_lon = center_lons[i,j]
            center_lat = center_lats[i,j]
            if(args.multiple_models):
                model = mversion[i,j]
                conv_model = load_model(mult_models[i,j], compile=False)
                mc = pickle.load(open(mc_files[i,j], 'rb'))
                try:
                    C02mean = mc['scaling_values']['mean']['C02']
                    C02stddev = mc['scaling_values']['stddev']['C02']
                    C02_present = True
                except:
                    print('No C02')
                    C02_present = False
                try:
                    C05mean = mc['scaling_values']['mean']['C05']
                    C05stddev = mc['scaling_values']['stddev']['C05']
                    C05_present = True
                except:
                    print('No C05')
                    C05_present = False
                try:
                    C13mean = mc['scaling_values']['mean']['C13']
                    C13stddev = mc['scaling_values']['stddev']['C13']
                    C13_present = True
                except:
                    print('No C13')
                    C13_present = False


            # Avoid out of bounds center_lat and center_lon
            if center_lon > -74:
                center_lon = -74
            elif center_lon < -117:
                center_lon = -117
            elif center_lat > 43:
                center_lat = 43
            elif center_lat < 32:
                center_lat = 32

            satdir = topdt.strftime(rootdatadir + '%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
            radardir = topdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
            fileloc = os.path.join(PWD, local_dir)
            utils.mkdir_p(fileloc)
            if(args.multiple_models):
                preddir = topdt.strftime(os.path.join(rootoutdir, 'MSthesis/' + 'Multiple Models' + '/%Y/%Y%m%d/'))
            else:
                preddir = topdt.strftime(os.path.join(rootoutdir, 'MSthesis/' + model + '/%Y/%Y%m%d/'))
            utils.mkdir_p(preddir)

            abi_channels = ['02', '05', '11', '13', '14', '15']
            print('Copying satellite netcdfs...')
            for ch in abi_channels:
                netcdfpattern = topdt.strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
                sat_netcdfs = np.sort(glob(netcdfpattern))

                # Copy satellite netcdfs
                print('len(sat_netcdfs):', len(sat_netcdfs))
                for k in range(len(sat_netcdfs)):
                    # copy file from satdir
                    try:
                        shutil.copy(sat_netcdfs[k], fileloc)
                    except (PermissionError, OSError, IOError) as err:
                        print(traceback.format_exc())
                        print('Moving on to next sat DT...')
                        continue
            print('Satellite netcdf copying complete.')

            netcdfpattern_C02 = topdt.strftime(fileloc + 'OR_ABI-L1b-RadC-M*C02_G16_s%Y*')
            sat_netcdfs = np.sort(glob(netcdfpattern_C02))

            if len(sat_netcdfs) != 1:
                print('Since we have only one datetime of interest for each panel, something is wrong here.')
                sys.exit()

            sat_netcdf = sat_netcdfs[0]
            basefile = os.path.basename(sat_netcdf)
            parts = basefile.split('_')
            separator = '_'
            year_day_time = parts[3][1:-1]
            dt_start = datetime.strptime(year_day_time, '%Y%j%H%M%S')
            timestamp = str(dt_start)
            print('timestamp: ', timestamp)

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

            try:
                C13vals = crop_scn['C13'].values
                C15vals = crop_scn['C15'].values
                C11vals = crop_scn['C11'].values
                C14vals = crop_scn['C14'].values
                C02vals = crop_scn['C02'].values
                C05vals = crop_scn['C05'].values
            except KeyError:
                exception_files.append(dt_start)

            if (np.isnan(np.min(C13vals)) or np.isnan(np.max(C13vals)) or np.isnan(np.min(C15vals)) or
                    np.isnan(np.max(C15vals)) or np.isnan(np.min(C11vals)) or np.isnan(np.max(C11vals)) or
                    np.isnan(np.min(C14vals)) or np.isnan(np.max(C14vals)) or np.isnan(np.min(C02vals)) or
                    np.isnan(np.max(C02vals)) or np.isnan(np.min(C05vals)) or np.isnan(np.max(C05vals))):
                bad_data_files.append(dt_start)

            sat_grid = crop_scn['C05'].attrs['area']

            VIScrop_scn = crop_scn
            VIScrop_scn.load(['DayCloudPhaseFC'])
            VISresamp = VIScrop_scn.resample(resampler='native')

            time_RGB = time.time()

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

            remapped_radar = pr.kd_tree.resample_nearest(radar_grid, radardataset, sat_grid, radius_of_influence=10000,
                                                         fill_value=-999)

            state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m',
                                                         facecolor='none')
            NWS_cmap = ctables.registry.get_colortable('NWSStormClearReflectivity')
            NWS_cmap_sub = cmr.get_sub_cmap(NWS_cmap, 0.51, 0.8)

            x2km, y2km = VISresamp['C13'].attrs['area'].get_proj_vectors()
            x1km, y1km = VISresamp['C05'].attrs['area'].get_proj_vectors()
            xhalfkm, yhalfkm = VISresamp['C02'].attrs['area'].get_proj_vectors()

            projx = VISresamp['C05'].x.compute().data;
            projy = VISresamp['C05'].y.compute().data;

            xx2km, yy2km = np.meshgrid(x2km, y2km)
            xx1km, yy1km = np.meshgrid(x1km, y1km)
            xxhalfkm, yyhalfkm = np.meshgrid(xhalfkm, yhalfkm)
            print('2km: ', xx2km.shape, yy2km.shape)
            print('1km: ', xx1km.shape, yy1km.shape)
            print('0.5km: ', xxhalfkm.shape, yyhalfkm.shape)

            if (C02_present):
                C02vals = crop_scn['C02'].values
                print('Shape of C02: ', C02vals.shape)
                C02norm_vals = (crop_scn['C02'].values - C02mean) / C02stddev
                C02norm = np.zeros((1, C02norm_vals.shape[0], C02norm_vals.shape[1], 1))
                C02norm[0, :, :, 0] = C02norm_vals
            if (C05_present):
                C05vals = crop_scn['C05'].values
                print('Shape of C05: ', C05vals.shape)
                C05norm_vals = (crop_scn['C05'].values - C05mean) / C05stddev
                C05norm = np.zeros((1, C05norm_vals.shape[0], C05norm_vals.shape[1], 1))
                C05norm[0, :, :, 0] = C05norm_vals
            if (C13_present):
                C13vals = crop_scn['C13'].values
                print('Shape of C13: ', C13vals.shape)
                C13norm_vals = (crop_scn['C13'].values - C13mean) / C13stddev
                C13norm = np.zeros((1, C13norm_vals.shape[0], C13norm_vals.shape[1], 1))
                C13norm[0, :, :, 0] = C13norm_vals

            xmin = np.min(x2km);
            xmax = np.max(x2km);
            ymin = np.min(y2km);
            ymax = np.max(y2km)
            print('max and mins:')
            print('x', xmin, xmax)
            print('y', ymin, ymax)

            if (C02_present and C05_present and C13_present):
                print('Channel combination: C02C05C13')
                CH_COMBO = 'C02C05C13'
                pred_in = [C02norm, C05norm, C13norm]
            if (C02_present and C05_present and not C13_present):
                print('Channel combination: C02C05')
                pred_in = [C02norm, C05norm]
                CH_COMBO = 'C02C05'
            if (C02_present and C13_present and not C05_present):
                print('Channel combination: C02C13')
                CH_COMBO = 'C02C13'
                pred_in = [C02norm, C13norm]
            if (C05_present and C13_present and not C02_present):
                print('Channel combination: C05C13')
                CH_COMBO = 'C05C13'
                pred_in = [C05norm, C13norm]
            new_preds = conv_model.predict(pred_in)
            new_preds = new_preds[0]

            axes[i, j].set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
            axes[i, j].add_feature(cfeature.BORDERS)
            axes[i, j].add_feature(state_borders, edgecolor='tan')

            if(C02_present):
                print('max ch02', np.max(C02vals))
                print(xxhalfkm.shape, yyhalfkm.shape, C02vals.shape)
                visimg = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, C02vals, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
            else:
                if(C05_present):
                    print('max ch05', np.max(C05vals))
                    print(xx2km.shape, yy2km.shape, C05vals.shape)
                    #visimg = ax.pcolormesh(xx2km, yy2km, C05vals, vmin=0, vmax=100,
                    #                       cmap=plt.get_cmap('Greys_r'))
                    visdata= VISresamp['C05'].compute().data
                    crs = VISresamp['C05'].attrs['area'].to_cartopy_crs()
                    visimg = axes[i, j].imshow(visdata, transform=crs, extent=crs.bounds, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
                else:
                    print('max ch13', np.max(C13vals))
                    print(xx2km.shape, yy2km.shape, C13vals.shape)
                    irdata = VISresamp['C13'].compute().data
                    crs = VISresamp['C13'].attrs['area'].to_cartopy_crs()
                    visimg = axes[i, j].imshow(irdata, transform=crs, extent=crs.bounds, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))

            radimg_10C = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, ma.masked_less(zoom(remapped_radar, 2, order=0), 10),
                                               cmap=NWS_cmap_sub, norm=Normalize(-1, 80), linewidth=0, antialiased=True,
                                               alpha=0.7)
            colors = ['#FF8300', '#9700FF', '#00FFFB', '#FF00EC', '#0015FF']
            values = [0.1, 0.25, 0.5, 0.75, 0.9]
            #linewidths = [2.3, 2.4, 2.5, 2.8, 3]
            #linewidths = [1.3, 1.4, 1.5, 1.6, 1.8]
            #linewidths = [1.0, 1.1, 1.2, 1.3, 1.5]
            linewidths = [1.0, 1.0, 1.0, 1.0, 1.1]
            labels = [str(int(v * 100)) + '%' for v in values]
            contour_img = axes[i, j].contour(xxhalfkm, yyhalfkm, zoom(new_preds, 2, order=0), values, colors=colors,
                                             linewidths=linewidths)  # , cmap=plt.get_cmap('cividis'))
            custom_lines = [Line2D([0], [0], color=c, lw=lw * 2) for c, lw in zip(colors, linewidths)]

            if(args.multiple_models):
                axes[i, j].set_title((' ').join([CH_COMBO, ' ', timestamp, 'UTC']), fontsize=14)
            else:
                axes[i, j].set_title((' ').join([timestamp, 'UTC']), fontsize=16)
            # ax.set_title((' ').join(['MRMS', 'CONUS', timestamp, 'UTC']), fontsize=16)

            VIScrop_scn.unload()
            crop_scn.unload()
            scn.unload()

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

    cbaxes = fig.add_axes([0.125, .06, 0.2, 0.03])  # left bottom width height
    #cbaxes = fig.add_axes([0.01, -0.05, 0.27, 0.03])  # left bottom width height
    cbar = plt.colorbar(visimg, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0,
                        ticks=np.arange(0, 100, 10), cax=cbaxes)
    cbar.ax.tick_params(labelsize=12)
    cbaxes2 = fig.add_axes([0.34, .06, 0.2, 0.03])  # left bottom width height
    #cbaxes2 = fig.add_axes([0.31, -0.05, 0.27, 0.03])  # left bottom width height
    cbar2 = plt.colorbar(radimg_10C, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                         ticks=np.arange(10, 80, 10), cax=cbaxes2)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label('Radar Reflectivity at -10C [dBz]', fontsize=12)
    cbar2.set_alpha(1.0)
    cbar2.draw_all()
    leg2 = fig.legend(custom_lines, labels, fontsize=12, loc='lower right', handlelength=1.5,
                     ncol=len(labels), title = 'Probability of Convection in 60 min', title_fontsize=12, bbox_to_anchor=(0.8, .05, 0.13, 0.03))
    #leg2 = fig.legend(custom_lines, labels, fontsize=12, loc='lower right', handlelength=1.5,
    #                  ncol=len(labels), title='Probability of Convection in 60 min', title_fontsize=14,
    #                  bbox_to_anchor=(0.675, .03, 0.23, 0.03))
    #plt.setp(leg2.get_title(), fontsize=14)
    #ctext = plt.text(0.65, -0.09, 'Probability of Convection in 60 min', fontsize=14)#, transform=ax.transAxes)

    if(args.multiple_models):
        cbar.set_label('ABI 0.64-$\mu$m reflectance [%] \n or ABI 1.6-$\mu$m reflectance [%]', fontsize=12)
    else:
        if (C02_present):
            cbar.set_label('ABI 0.64-$\mu$m reflectance [%]', fontsize=14)
        else:
            if (C05_present):
                cbar.set_label('ABI 1.6-$\mu$m reflectance [%]', fontsize=14)
            else:
                cbar.set_label('ABI 10.3-$\mu$m BT [%]', fontsize=14)

    PREDfilepath = os.path.join(preddir, file_name)
    plt.savefig(PREDfilepath, bbox_inches='tight', dpi=300)
    plt.close('all')
    print('Predictions made and plotted.')