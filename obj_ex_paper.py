from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
os.environ['SATPY_CONFIG_PATH'] = '/home/sortland/thundercast/PPP_config/'
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm as cmaps
from glob import glob
from scipy.ndimage import zoom
from matplotlib.lines import Line2D
import matplotlib
import argparse
from pyproj import Proj
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyresample as pr
from pyresample import data_reduce
import pygrib as pg
from satpy import Scene
from satpy.modifiers.parallax import get_parallax_corrected_lonlats
from satpy.writers import get_enhanced_image
# import tobac
# import iris
import errno
import pyart
import numpy.ma as ma
from scipy.ndimage import zoom
import random
from skimage.morphology import binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.measure import label
import dask.array as da
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"
#print('using satpy version', str(sp.__version__))

def get_remap_cache(input_def, target_def, radius_of_influence=4000):
  input_index, output_index, index_array, distance_array = \
      pr.kd_tree.get_neighbour_info(input_def, target_def, radius_of_influence, neighbours=1)
  remap_cache = {'input_index':input_index, 'output_index':output_index,
                 'index_array':index_array, 'target_def':target_def}
  return remap_cache

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
    try:
        scn = Scene(reader='abi_l1b', filenames=abi_files)
        scn.load(CHs)  # Load the scene and check for data problems
    except:
        print('Scene would not load.')
        sys.exit()
    return scn


if __name__ == "__main__":
    ###Directory Specs
    rootoutdir = '/ships22/grain/sortland/objects/plots/'
    rootobjdir = '/ships22/grain/sortland/objects'
    rootsatdir = '/arcdata/goes/grb/goes16'
    rootdatadir = '/ships22/grain/sortland/remapped_obj_data'
    rootpreddir = '/ships22/grain/sortland/thundercast_ncs'
    rootglmdir = rootenidir = '/ships19/grain/probsevere/lightning' #2014, 2015, 2016, 2017, 2018, 2022, 2023
    #rootglmdir = rootenidir = '/ships22/grain/probsevere/lightning' #2018, 2019, 2020, 2021

    dates = ['2022-08-03']
    state_borders = cfeature.NaturalEarthFeature(category='cultural',
                                                 name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')
    ####
    # hs = 100
    # lat = 32.59
    # lon = -80.66
    # limit_time = True
    # if limit_time:
    #     start = datetime(2022, 7, 24, 13, 30)
    #     end = datetime(2022, 7, 24, 15)
    t_half = timedelta(minutes=2, seconds=30)
    sub_name = '6091'
    ####
    hs = 160
    lat = 36.976
    lon = -105.69
    limit_time = True
    if limit_time:
        start = datetime(2022, 8, 3, 17)
        end = datetime(2022, 8, 3, 23)
    CVD_cmap_all = matplotlib.colormaps.get_cmap('pyart_HomeyerRainbow')
    colors1 = CVD_cmap_all(np.linspace(0, 0.25, 98))
    colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 158))
    cmap_colors = np.vstack((colors1, colors2))
    CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)

    #ENI Specs
    time_w = lambda x: (x.replace('time:', ''))
    type_w = lambda x: (x.replace('type:', ''))
    lat_w = lambda x: (x.replace('latitude:', ''))
    lon_w = lambda x: (x.replace('longitude:', ''))
    pC_w = lambda x: (x.replace('peakCurrent:', ''))
    ic_w = lambda x: (x.replace('icHeight:', ''))
    ns_w = lambda x: (x.replace('numSensors:', ''))
    icM_w = lambda x: (x.replace('icMultiplicity:', ''))
    cgM_w = lambda x: (x.replace('cgMultiplicity:', ''))

    for date in dates:
        dt = datetime.strptime(date, '%Y-%m-%d')
        directory = dt.strftime(rootobjdir + '/%Y/%Y%m%d/')
        outdir = dt.strftime(rootoutdir + 'track_plots/%Y%m%d_' + sub_name + '/')
        os.makedirs(outdir, exist_ok=True)
        segmentation_file = directory + 'segments.netcdf'
        feature_file = directory + 'features_filtered.pickle'
        feature_file_raw = directory + 'features.pickle'

        segment_info = xr.open_dataset(segmentation_file, engine='netcdf4')

        feature_info = pd.read_pickle(feature_file)
        all_features = feature_info['feature']
        raw_feature_info = pd.read_pickle(feature_file_raw)
        # for col in feature_info:
        #     print(col)

        ###Limit Features to Desired Times####
        if limit_time:
            feature_info = feature_info[feature_info['time'] > start]
            feature_info = feature_info[feature_info['time'] < end]
        #########

        #get prediction area information
        pred_area = pr.geometry.AreaDefinition(segment_info.attrs['area_id'], segment_info.attrs['description'], segment_info.attrs['proj_id'],
                                               segment_info.attrs['projection'], segment_info.attrs['width'], segment_info.attrs['height'], segment_info.attrs['area_extent'])

        center_x_idx, center_y_idx =  pred_area.get_array_indices_from_lonlat(lon, lat)
        ymin_index = int(center_y_idx - hs)
        ymax_index = int(center_y_idx + hs)
        xmin_index = int(center_x_idx - hs)
        xmax_index = int(center_x_idx + hs)

        lon_min, lat_max = pred_area.get_lonlat_from_array_coordinates(xmin_index, ymin_index)
        lon_max, lat_min = pred_area.get_lonlat_from_array_coordinates(xmax_index, ymax_index)

        proj_dict = pr.utils.proj4.proj4_str_to_dict(pred_area.proj_str)
        geoproj = ccrs.Geostationary(central_longitude=proj_dict['lon_0'], sweep_axis=proj_dict['sweep'],
                                     satellite_height=proj_dict['h'])
        #get dictionary of radar files
        radardir = dt.strftime(os.path.join(rootdatadir, '%Y/%Y%m%d/radar_-10C/'))
        rad_dict = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(radardir, '*.netcdf'))}
        #get dictionary of radar qi files
        qidir = dt.strftime(os.path.join(rootdatadir, '%Y/%Y%m%d/radar_qi/'))
        qi_dict = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                   glob(os.path.join(qidir, '*.netcdf'))}
        #get dictionary of hrrr files
        hrrrdir = dt.strftime(os.path.join(rootdatadir, '%Y/%Y%m%d/hrrr/'))
        hrrr_dict = {datetime.strptime(x.split('/')[-1], '%Y-%m-%d-%H.netcdf'): x for x in
                   glob(os.path.join(hrrrdir, '*.netcdf'))}
        #get dictionary of eni files
        eni_hr_dirs = np.sort(glob(dt.strftime('/ships19/grain/lightning' + '/%Y/%Y_%m_%d_%j/') + '*'))
        all_eni_txt = []
        for dir in eni_hr_dirs:
            all_eni_txt = all_eni_txt + glob(dir + '/*')
        eni_dict = {datetime.strptime(x.split('/')[-1][6:], '%Y%j_%H%M%S.txt'): x for x in all_eni_txt}
        #get dictionary of glm files
        glmdir = dt.strftime('/arcdata/goes/grb/goes16/%Y/%Y_%m_%d_%j/glm/L2/LCFA/')
        glm_dict = {datetime.strptime(os.path.basename(x).split('_')[3], 's%Y%j%H%M%S%f'): x for x in glob(glmdir + '*')}

        #Make a plot for each frame
        track_colors = {}
        for frame_count, f in enumerate(np.unique(feature_info['frame'])):
            subset = feature_info[feature_info['frame'] == f] #select only information for the frame
            raw_subset = raw_feature_info[raw_feature_info['frame'] == f]
            frame_time = datetime.strptime(subset['timestr'].iloc[0], '%Y-%m-%d %H:%M:%S')
            frame = subset['frame'].values[0]
            segment_at_time = segment_info.where(segment_info.coords['time'] == segment_info.coords['time'][frame],
                                                 drop=True)  # gets a subset of dataset for timestamp
            features = subset['feature'].values
            test = np.isin(segment_at_time['segmentation_mask'], features)
            segment_at_time['sub_mask'] = (('time', 'y', 'x'), test)
            cell_obj = segment_at_time.where(segment_at_time['sub_mask'])

            # xmax_index, ymin_index = pred_area.get_array_indices_from_lonlat(lon_max, lat_max)
            # xmin_index, ymax_index = pred_area.get_array_indices_from_lonlat(lon_min, lat_min)

            #Get cropped info for plotting
            crop_xs = cell_obj.coords['proj_x'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_ys = cell_obj.coords['proj_y'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_lats = cell_obj.coords['latitude'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_lons = cell_obj.coords['longitude'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_area = pr.geometry.GridDefinition(lons=crop_lons, lats=crop_lats)

            #Get segmentation information
            all_segment_mask = segment_at_time['segmentation_mask'][0, ymin_index:ymax_index, xmin_index:xmax_index]
            all_segment_values = np.unique(all_segment_mask)

            #Get prediction information
            predfile = frame_time.strftime(rootpreddir + '/%Y/%Y%m%d/%Y%m%d-%H%M%S.netcdf')
            ds_preds = xr.open_dataset(predfile, engine='netcdf4')
            predictions_0 = ds_preds['thundercast_preds'][ymin_index:ymax_index, xmin_index:xmax_index]
            predictions = gaussian_filter(predictions_0, sigma=2) #apply same filter as used to make objects in pred_objects.py
            ds_preds.close()

            # For plotting filter
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': geoproj})
            # fig.subplots_adjust(wspace=0.1, hspace=0.15)
            # #Subplot 0,0
            # axes[0].set_extent([np.min(crop_xs), np.max(crop_xs), np.min(crop_ys), np.max(crop_ys)], crs=geoproj)
            # axes[0].add_feature(state_borders, edgecolor='plum')
            # predIMG = axes[0].pcolormesh(crop_xs, crop_ys, predictions_0, vmin=0, vmax=1, cmap='bone')
            # cbaxes = fig.add_axes([0.125, .15, 0.37, 0.03])  # left bottom width height
            # cbar = plt.colorbar(predIMG, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0,
            #                     ticks=np.arange(0, 1.1, 0.2), cax=cbaxes)
            # cbar.ax.tick_params(labelsize=12)
            # cbar.set_label(r'ThunderCast Predictions', fontsize=12)
            # axes[0].set_title('No-Filter', fontsize=14)
            # axes[1].set_extent([np.min(crop_xs), np.max(crop_xs), np.min(crop_ys), np.max(crop_ys)], crs=geoproj)
            # axes[1].add_feature(state_borders, edgecolor='plum')
            # predIMG = axes[1].pcolormesh(crop_xs, crop_ys, predictions, vmin=0, vmax=1, cmap='bone')
            # axes[1].set_title('Gaussian Filter, Sigma = 2', fontsize=14)
            # plt.show()
            # sys.exit()

            #Get Radar at -10C, Find the file closest to the frame time
            pos_deltat = [np.absolute((x-frame_time).total_seconds()) for x in list(rad_dict.keys())]
            min_ind = pos_deltat.index(min(pos_deltat))  # find the file closest in time to the glm file
            radar_file = rad_dict[list(rad_dict.keys())[min_ind]]
            ds_rad = xr.open_dataset(radar_file, engine='netcdf4')
            refl = ds_rad['Reflectivity_-10C'][ymin_index:ymax_index, xmin_index:xmax_index]
            ds_rad.close()

            # #Get radar QI
            # pos_deltat = [np.absolute((x-frame_time).total_seconds()) for x in list(qi_dict.keys())]
            # min_ind = pos_deltat.index(min(pos_deltat))  # find the file closest in time to the glm file
            # qi_file = qi_dict[list(qi_dict.keys())[min_ind]]
            # ds_qi = xr.open_dataset(qi_file, engine='netcdf4')
            # qi = ds_qi['Reflectivity_QI'][ymin_index:ymax_index, xmin_index:xmax_index]
            # ds_qi.close()

            # #Get HRRR
            # pos_deltat = [np.absolute((x-frame_time).total_seconds()) for x in list(hrrr_dict.keys())]
            # min_ind = pos_deltat.index(min(pos_deltat))  # find the file closest in time to the glm file
            # hrrr_file = hrrr_dict[list(hrrr_dict.keys())[min_ind]]
            # ds_hrrr = xr.open_dataset(hrrr_file, engine='netcdf4')
            # hrrr_ebs = ds_hrrr['EBS_magnitude'][ymin_index:ymax_index, xmin_index:xmax_index]
            # hrrr_mlcape = ds_hrrr['MLCAPE'][ymin_index:ymax_index, xmin_index:xmax_index]
            # ds_hrrr.close()

            lightning_start = frame_time - t_half
            lightning_end = frame_time + t_half
            #Get GLM
            glm_files = [glm_dict[x] for x in glm_dict.keys() if lightning_start <= x < lightning_end]
            glm_dfs = []
            for glm_file in glm_files:
                ds_glm = xr.open_dataset(glm_file, engine='netcdf4')
                d = {'event_id':ds_glm.variables['event_id'], 'longitude':ds_glm.variables['event_lon'], 'latitude':ds_glm.variables['event_lat']}
                glm_df = pd.DataFrame(data=d)
                glm_df = glm_df[glm_df['latitude'] >= np.min(crop_lats.to_numpy())]
                glm_df = glm_df[glm_df['latitude'] <= np.max(crop_lats.to_numpy())]
                glm_df = glm_df[glm_df['longitude'] >= np.min(crop_lons.to_numpy())]
                glm_df = glm_df[glm_df['longitude'] <= np.max(crop_lons.to_numpy())]
                glm_dfs.append(glm_df)
                ds_glm.close()

            glm_df = pd.concat(glm_dfs, ignore_index=True)
            glm_df = glm_df.drop_duplicates(subset=['longitude', 'latitude'])
            glm_df['proj_x'], glm_df['proj_y'] = pred_area.get_projection_coordinates_from_lonlat(glm_df['longitude'],
                                                                                                  glm_df['latitude'])

            #Get ENI
            eni_files = [eni_dict[x] for x in eni_dict.keys() if lightning_start <= x < lightning_end]
            # eni
            eni_dfs = []
            for eni_file in eni_files:
                eni = pd.read_csv(eni_file, names=['time', 'type', 'latitude', 'longitude', 'peakCurrent', 'icHeight',
                                                   'numSensors', 'icMultiplicity', 'cgMultiplicity'],
                                  converters={'time': time_w, 'type': type_w, 'latitude': lat_w, 'longitude': lon_w,
                                              'peakCurrent': pC_w, 'icHeight': ic_w, 'numSensors': ns_w,
                                              'icMultiplicity': icM_w, 'cgMultiplicity': cgM_w})
                eni['latitude'] = eni['latitude'].astype(float)
                eni['longitude'] = eni['longitude'].astype(float)
                eni['time'] = eni['time'].astype('string')
                eni['time'] = eni['time'].str[1:-12]
                eni['time'] = pd.to_datetime(eni['time'], yearfirst=True, format='mixed')
                eni = eni[eni['latitude'] >= np.min(crop_lats.to_numpy())]
                eni = eni[eni['latitude'] <= np.max(crop_lats.to_numpy())]
                eni = eni[eni['longitude'] >= np.min(crop_lons.to_numpy())]
                eni = eni[eni['longitude'] <= np.max(crop_lons.to_numpy())]
                eni_dfs.append(eni)
            eni = pd.concat(eni_dfs, ignore_index=True)
            eni = eni.drop_duplicates(subset=['longitude', 'latitude'])
            eni['proj_x'], eni['proj_y'] = pred_area.get_projection_coordinates_from_lonlat(eni['longitude'],
                                                                                            eni['latitude'])

            #Get ABI
            abi_files = get_abifile_lists(rootsatdir, [frame_time])[0]
            scn = get_satpy_scene(abi_files)
            remapped_C13 = pr.kd_tree.resample_nearest(scn['C13'].attrs['area'], scn['C13'].values, crop_area,
                                                       radius_of_influence=10000,
                                                       fill_value=-999)
            sat_lon = scn['C13'].attrs['orbital_parameters']['satellite_nominal_longitude']
            sat_lat = scn['C13'].attrs['orbital_parameters']['satellite_nominal_latitude']
            sat_alt = scn['C13'].attrs['orbital_parameters']['satellite_nominal_altitude']
            lon, lat = scn['C13'].attrs['area'].get_lonlats()
            alt = 8000 #m
            #lon, lat = crop_area.get_lonlats()
            corr_lon, corr_lat = get_parallax_corrected_lonlats(sat_lon, sat_lat, sat_alt, lon, lat, alt)
            abi_swath = pr.SwathDefinition(lons=corr_lon, lats=corr_lat)
            remap_cache = get_remap_cache(abi_swath, crop_area, radius_of_influence=5000)
            C13_resamp = pr.kd_tree.get_sample_from_neighbour_info('nn', remap_cache['target_def'].shape, scn['C13'].values,
                                                                     remap_cache['input_index'],
                                                                     remap_cache['output_index'],
                                                                     remap_cache['index_array'])
            # C13_resamp = pr.kd_tree.resample_nearest(abi_swath, remapped_C13, crop_area,
            #                                          radius_of_influence=5000, fill_value=None)
            del scn

            #Start Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 6), subplot_kw={'projection': geoproj})
            fig.subplots_adjust(wspace=0.1, hspace=0.2)
            labels = ['A)', 'B)', 'C)']
            for n, ax in enumerate(axes.flat):
                name = labels[n]
                ax.set_title(name, fontsize=14, loc='left')

            #Subplot 0,0
            axes[0].set_extent([np.min(crop_xs), np.max(crop_xs), np.min(crop_ys), np.max(crop_ys)], crs=geoproj)
            axes[0].add_feature(state_borders, edgecolor='plum')
            predIMG = axes[0].pcolormesh(crop_xs, crop_ys, predictions, vmin=0, vmax=1, cmap='bone')
            cbaxes = fig.add_axes([0.125, .12, 0.245, 0.04])  # left bottom width height
            cbar = plt.colorbar(predIMG, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0,
                                ticks=np.arange(0, 1.1, 0.2), cax=cbaxes)
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label(r'ThunderCast Predictions', fontsize=14)
            #Subplot 0,1
            axes[1].set_extent([np.min(crop_xs), np.max(crop_xs), np.min(crop_ys), np.max(crop_ys)], crs=geoproj)
            axes[1].add_feature(state_borders, edgecolor='plum')
            SCNimg = axes[1].pcolormesh(crop_xs, crop_ys, C13_resamp, vmin=200, vmax=300,
                                          cmap=matplotlib.colormaps['Greys'])
            RADimg = axes[1].pcolormesh(crop_xs, crop_ys, ma.masked_less(refl, 10), cmap=CVD_cmap,
                                        norm=Normalize(-1, 80),
                                        linewidth=0, antialiased=True)
            cbaxes2 = fig.add_axes([0.39, .12, 0.245, 0.04])  # left bottom width height
            cbar2 = plt.colorbar(RADimg, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                                 ticks=np.arange(10, 80, 10), cax=cbaxes2)
            cbar2.ax.tick_params(labelsize=13)
            cbar2.set_label(r'Radar Reflectivity at -10$^\circ$C [dBZ]', fontsize=14)
            cbar2.set_alpha(1.0)
            #Subplot 0,2
            axes[2].set_extent([np.min(crop_xs), np.max(crop_xs), np.min(crop_ys), np.max(crop_ys)], crs=geoproj)
            axes[2].add_feature(state_borders, edgecolor='plum')
            ABIimg = axes[2].pcolormesh(crop_xs, crop_ys, C13_resamp, cmap=matplotlib.colormaps['Greys'], linewidth=0, antialiased=True)
            axes[2].scatter(glm_df['proj_x'], glm_df['proj_y'], color='b', s=8, label='GLM Event')
            axes[2].scatter(eni['proj_x'], eni['proj_y'], color='r', s=8, label='ENTLN Event')
            axes[2].legend(loc='upper left', fontsize=12)
            cbaxes5 = fig.add_axes([0.655, .12, 0.245, 0.04])  # left bottom width height
            cbar5 = plt.colorbar(ABIimg, orientation='horizontal', pad=0.01, drawedges=0,
                            ticks=np.arange(200, 305, 20), cax=cbaxes5, extend='both')
            cbar5.ax.tick_params(labelsize=13)
            cbar5.set_label(r'ABI 10.3-$\mu$m BT [K]', fontsize=14)
            cbar5.set_alpha(1.0)

            all_tracks = []
            for seg_num in all_segment_values:
                if seg_num in features:
                    track_num = feature_info.loc[feature_info['feature'] == seg_num]['ms_track_id'].values[0]
                    all_tracks.append(int(track_num))

            colors = cmaps.rainbow(np.linspace(0, 1, len(all_tracks)))
            rand_indices = np.arange(0, len(all_tracks), 1)
            random.shuffle(rand_indices)
            for i, track_num in enumerate(all_tracks):
                if track_num not in track_colors.keys():
                    track_colors[track_num] = colors[rand_indices[i]]

            #Add in objects
            for seg_num in all_segment_values:
                line_thickness = 1
                curr_seg = (all_segment_mask == seg_num).astype(int)
                if seg_num == 0 or seg_num == -1:
                    continue
                if seg_num in features:
                    xval = feature_info.loc[feature_info['feature'] == seg_num]['proj_x']
                    yval = feature_info.loc[feature_info['feature'] == seg_num]['proj_y']
                    track_num = feature_info.loc[feature_info['feature'] == seg_num]['ms_track_id'].values[0]
                    color = track_colors[int(track_num)]
                    # print('track_number:', track_num)
                    # print('latitude', feature_info.loc[feature_info['feature'] == seg_num]['latitude'])
                    # print('longitude', feature_info.loc[feature_info['feature'] == seg_num]['longitude'])
                    axes[1].contour(crop_xs, crop_ys, curr_seg, colors=[color, ], linewidths=line_thickness)
                    axes[2].contour(crop_xs, crop_ys, curr_seg, colors=[color, ], linewidths=line_thickness)
                if seg_num not in all_features.unique():
                    xval = raw_subset.loc[raw_feature_info['feature'] == seg_num]['proj_x']
                    yval = raw_subset.loc[raw_feature_info['feature'] == seg_num]['proj_y']
                    color = 'tab:brown'
                    line_thickness = 0.5
                axes[0].contour(crop_xs, crop_ys, curr_seg, colors=[color, ], linewidths=line_thickness)
                axes[0].scatter(xval, yval, s=8, color=color)

            #Set Titles
            #axes[0,0].set_title('All Pred Objects (Non-Brown = Tracked)')
            titlename = (' ').join(['Prediction Time:', str(frame_time), 'UTC'])
            plt.suptitle(titlename, fontsize=16, y=0.9)
            outfile = outdir + frame_time.strftime('%Y%m%d-%H%M%S.png')
            # plt.show()
            # sys.exit()
            print(outfile)
            plt.savefig(outfile, dpi=200, bbox_inches='tight')
            sys.exit()