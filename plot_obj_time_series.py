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
import pygrib as pg
from satpy import Scene
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

os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

if __name__ == "__main__":
    ###Directory Specs
    rootoutdir = '/ships22/grain/sortland/objects/plots'
    rootobjdir = '/ships22/grain/sortland/objects'
    rootpreddir = '/ships22/grain/sortland/thundercast_ncs'

    dates = ['2022-08-06']
    selected_track = 8075
    print(dates, selected_track)
    state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')
    ####
    hs = 115
    lat = 27.2#34.7 #30.7
    lon = -80.42#-82.8 #-91.25
    ###
    limit_time = True
    if limit_time:
        start = datetime(2022, 8, 6, 19, 31)
        end = datetime(2022, 8, 6, 20, 12)
        # start = datetime(2022, 7, 19, 21, 51)
        # end = datetime(2022, 7, 19, 23, 42)
    t_half = timedelta(minutes=2, seconds=30)

    for date in dates:
        dt = datetime.strptime(date, '%Y-%m-%d')
        directory = dt.strftime(rootobjdir + '/%Y/%Y%m%d/')
        segmentation_file = directory + 'segments.netcdf'
        feature_file = directory + 'features_filtered.pickle'
        feature_file_raw = directory + 'features.pickle'

        segment_info = xr.open_dataset(segmentation_file, engine='netcdf4')
        feature_info = pd.read_pickle(feature_file)
        all_features = feature_info['feature']
        raw_feature_info = pd.read_pickle(feature_file_raw)

        ###Limit Features to Desired Times####
        if limit_time:
            feature_info = feature_info[feature_info['time'] > start]
            feature_info = feature_info[feature_info['time'] < end]
        #########

        pred_area = pr.geometry.AreaDefinition(segment_info.attrs['area_id'], segment_info.attrs['description'],
                                               segment_info.attrs['proj_id'], segment_info.attrs['projection'],
                                               segment_info.attrs['width'], segment_info.attrs['height'],
                                               segment_info.attrs['area_extent'])

        center_x_idx, center_y_idx =  pred_area.get_array_indices_from_lonlat(lon, lat)
        ymin_index = int(center_y_idx - hs)
        ymax_index = int(center_y_idx + hs)
        xmin_index = int(center_x_idx - hs)
        xmax_index = int(center_x_idx + hs)

        lon_min, lat_max = pred_area.get_lonlat_from_array_coordinates(xmin_index, ymin_index)
        lon_max, lat_min = pred_area.get_lonlat_from_array_coordinates(xmax_index, ymax_index)

        proj_dict = pr.utils.proj4.proj4_str_to_dict(pred_area.proj4_string)
        geoproj = ccrs.Geostationary(central_longitude=proj_dict['lon_0'], sweep_axis=proj_dict['sweep'],
                                     satellite_height=proj_dict['h'])

        #get track numbers and assign colors
        all_tracks = []
        all_track_test = []
        for frame_count, f in enumerate(np.unique(feature_info['frame'])):
            subset = feature_info[feature_info['frame'] == f]  # select only information for the frame
            frame_time = datetime.strptime(subset['timestr'].iloc[0], '%Y-%m-%d %H:%M:%S')
            frame = subset['frame'].values[0]
            segment_at_time = segment_info.where(segment_info.coords['time'] == segment_info.coords['time'][frame],
                                                 drop=True)  # gets a subset of dataset for timestamp
            features = subset['feature'].values
            all_segment_mask = segment_at_time['segmentation_mask'][0, ymin_index:ymax_index, xmin_index:xmax_index]
            all_segment_values = np.unique(all_segment_mask)
            for seg_num in all_segment_values:
                if seg_num in features:
                    track_num = feature_info.loc[feature_info['feature'] == seg_num]['ms_track_id'].values[0]
                    #print(track_num)
                    if track_num not in all_tracks:
                        all_tracks.append(track_num)
        track_colors = {}
        colors = cmaps.rainbow(np.linspace(0, 1, len(all_tracks)))
        rand_indices = np.arange(0, len(all_tracks), 1)
        random.shuffle(rand_indices)
        for p, track_num in enumerate(all_tracks):
            if track_num not in track_colors.keys():
                track_colors[track_num] = colors[rand_indices[p]]
                line_colors = track_colors[track_num]

        numrows = 3
        numcols = 3
        fig, axes = plt.subplots(numrows, numcols, figsize=(10, 10), subplot_kw={'projection': geoproj})
        fig.subplots_adjust(wspace=0.05, hspace=0.2)

        desired_frames = np.unique(feature_info['frame'])
        print(desired_frames)
        print(len(desired_frames))
        desired_frames = desired_frames[:9]
        #desired_frames = desired_frames[::2]
        frame_place_holder = desired_frames.reshape((numrows, numcols))

        for frame_count, f in enumerate(desired_frames):
            i, j = np.where(frame_place_holder==f)
            i = i[0]
            j = j[0]
            subset = feature_info[feature_info['frame'] == f]  # select only information for the frame
            raw_subset = raw_feature_info[raw_feature_info['frame'] == f]
            frame_time = datetime.strptime(subset['timestr'].iloc[0], '%Y-%m-%d %H:%M:%S')
            frame_title = str(frame_time.hour)+ ':' + str(frame_time.minute) + ' UTC'
            frame = subset['frame'].values[0]
            segment_at_time = segment_info.where(segment_info.coords['time'] == segment_info.coords['time'][frame],
                                                 drop=True)  # gets a subset of dataset for timestamp

            features = subset['feature'].values
            test = np.isin(segment_at_time['segmentation_mask'], features)
            segment_at_time['sub_mask'] = (('time', 'y', 'x'), test)
            cell_obj = segment_at_time.where(segment_at_time['sub_mask'])

            # Get cropped info for plotting
            crop_xs = cell_obj.coords['proj_x'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_ys = cell_obj.coords['proj_y'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_lats = cell_obj.coords['latitude'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_lons = cell_obj.coords['longitude'][ymin_index:ymax_index, xmin_index:xmax_index]
            crop_area = pr.geometry.GridDefinition(lons=crop_lons, lats=crop_lats)

            # Get segmentation information
            all_segment_mask = segment_at_time['segmentation_mask'][0, ymin_index:ymax_index, xmin_index:xmax_index]
            all_segment_values = np.unique(all_segment_mask)

            # Get prediction information
            predfile = frame_time.strftime(rootpreddir + '/%Y/%Y%m%d/%Y%m%d-%H%M%S.netcdf')
            ds_preds = xr.open_dataset(predfile, engine='netcdf4')
            predictions_0 = ds_preds['thundercast_preds'][ymin_index:ymax_index, xmin_index:xmax_index]
            predictions = gaussian_filter(predictions_0, sigma=2)  # apply same filter as used to make objects in pred_objects.py
            ds_preds.close()

            axes[i,j].set_extent([np.min(crop_xs), np.max(crop_xs), np.min(crop_ys), np.max(crop_ys)], crs=geoproj)
            axes[i,j].add_feature(state_borders, edgecolor='plum')
            predIMG = axes[i,j].pcolormesh(crop_xs, crop_ys, predictions, vmin=0, vmax=1, cmap='bone')
            cbaxes = fig.add_axes([0.14, .06, 0.35, 0.03])  # left bottom width height
            cbar = plt.colorbar(predIMG, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0,
                                ticks=np.arange(0, 1.1, 0.2), cax=cbaxes)
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(r'ThunderCast Predictions', fontsize=11)

            #Add in objects
            for seg_num in all_segment_values:
                line_thickness = 1.2
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
                    #axes[i,j].contour(crop_xs, crop_ys, curr_seg, colors=[color, ], linewidths=line_thickness)
                    #axes[2].contour(crop_xs, crop_ys, curr_seg, colors=[color, ], linewidths=line_thickness)
                if seg_num not in all_features.unique():
                    xval = raw_subset.loc[raw_feature_info['feature'] == seg_num]['proj_x']
                    yval = raw_subset.loc[raw_feature_info['feature'] == seg_num]['proj_y']
                    color = 'tab:brown'
                    line_thickness = 1.2
                axes[i,j].contour(crop_xs, crop_ys, curr_seg, colors=[color, ], linewidths=line_thickness)
                axes[i,j].set_title(frame_title, fontsize=11, loc='left')
                #axes[i,j].scatter(xval, yval, s=8, color=color)

        custom_lines = [Line2D([0], [0], color='red', lw=2),
                        Line2D([0], [0], color='blue', lw=2)]
        custom_lines=[]
        for color in list(track_colors.values()):
            custom_lines.append(Line2D([0], [0], color=color, lw=2))
        custom_lines.append(Line2D([0], [0], color='tab:brown', lw=2))
        track_labels = [str(x) for x in list(track_colors.keys())]
        track_labels.append('RMVD')

        fig.legend(custom_lines, track_labels, fontsize=10, loc='lower right', handlelength=1.24,
                   ncol=4, title='ThunderCast Track #', title_fontsize=11,
                   bbox_to_anchor=(0.67, 0, 0.225, 0.04) # left bottom width height
)
        outfile = rootoutdir + '/OTS-' + dt.strftime('%Y%m%d-%H%M%S.png')
        print(outfile)
        plt.savefig(outfile, dpi=200, bbox_inches='tight')