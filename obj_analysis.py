import numpy as np
import xarray as xr
import tobac
import os, sys
import errno
from datetime import datetime,timedelta
from glob import glob
import tobac
import iris
import matplotlib.pyplot as plt
import numpy as np
import pyresample as pr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from satpy import Scene
import matplotlib
from matplotlib.pyplot import cm as cmaps
from netCDF4 import Dataset
from model_run_defs import format_inputs, UNet2D_LitModel
import torch
import time
import pandas as pd
import math
from scipy import stats as st
import geopandas as gpd
import shapely.geometry as sgeom
import matplotlib.colors as mcolors
import pyart
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"
#print('using tobac version', str(tobac.__version__))

def get_stats(ds, data_name, stat_dict, max=True, min=True, mean=True, std=True, median=True, time=False):
    all_stats = {}
    data = ds[data_name].values
    if time:
        data = data/60 #gets it into minutes
    if max:
        all_stats['maximum'] = np.nanmax(data)
    if min:
        all_stats['minimum'] = np.nanmin(data)
    if mean:
        all_stats['mean'] = np.nanmean(data)
    if std:
        all_stats['std'] = np.nanstd(data)
    if median:
        all_stats['median'] = np.nanmedian(data)
    stat_dict[data_name] = all_stats
    return stat_dict

def gen_hist(sub_df, outdir, var, bins, range_i, range_f, ticks, xlabel, fig_name, sciy=False):
    fig_width_cm = 25
    fig_height_cm = 25
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]
    fig, ax = plt.subplots(figsize=fig_size)

    fp_wl = sub_df[(sub_df['lightning'] == 1) & (sub_df['class'] == 'fp')]
    fp_wol = sub_df[(sub_df['lightning'] == 0) & (sub_df['class'] == 'fp')]
    tp_wl = sub_df[(sub_df['lightning'] == 1) & (sub_df['class'] == 'tp')]
    tp_wol = sub_df[(sub_df['lightning'] == 0) & (sub_df['class'] == 'tp')]

    #radar vals
    fp_wl = fp_wl[var]
    fp_wol = fp_wol[var]
    tp_wl = tp_wl[var]
    tp_wol = tp_wol[var]

    bins = bins
    labels = ['TP w/ Lightning', 'TP w/o Lightning', 'FP w/ Lightning', 'FP w/o Lightning']
    colors = ['tab:orange', 'tab:blue', 'tab:red', 'tab:green']
    ax.hist([tp_wl, tp_wol, fp_wl, fp_wol], bins, range=(range_i, range_f), histtype='bar', align = 'mid', label=labels, color=colors, rwidth=0.9)
    ax.legend(fontsize=16)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel('Number of ThunderCast Tracks', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.yaxis.offsetText.set_fontsize(16)
    if sciy:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
    figname = outdir + fig_name
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close()
    return

def get_2Dhist(sub_df, figname, xvar, yvar, xmin, xmax, ymin, ymax, stepx, stepy):
    fig_width_cm = 25
    fig_height_cm = 25
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]
    nx = 2
    ny = 2
    fig, ax = plt.subplots(nx, ny, figsize=fig_size, sharex=True, sharey=True)

    fp_wl = sub_df[(sub_df['lightning'] == 1) & (sub_df['class'] == 'fp')]
    fp_wol = sub_df[(sub_df['lightning'] == 0) & (sub_df['class'] == 'fp')]
    tp_wl = sub_df[(sub_df['lightning'] == 1) & (sub_df['class'] == 'tp')]
    tp_wol = sub_df[(sub_df['lightning'] == 0) & (sub_df['class'] == 'tp')]

    var_labels = {'min_C13':r'Minimum 10.3 $\mu$m BT [K]', 'C05':r'1.6 $\mu$m Reflectance at Min. 10.3 $\mu$m BT',
                  'C02':r'0.64 $\mu$m Reflectance at Min. 10.3 $\mu$m BT', 'C11C13_BTD':r'8.4 $\mu$m - 10.3 $\mu$m at Min. 10.3 $\mu$m BT',
                  'C11C14_BTD':r'8.4 $\mu$m - 11.2 $\mu$m at Min. 10.3 $\mu$m BT', 'C13C15_BTD':r'10.3 $\mu$m - 12.3 $\mu$m at Min. 10.3 $\mu$m BT',
                  'max_EBS_magnitude':r'Maximum EBS $\left[\frac{m}{s}\right]$', 'max_dBZ_-10C':r'Maximum Radar at -10$^{\circ}$C', 'max_MUCAPE':'Max. MUCAPE',
                  'max_MLCAPE180':r'Max. MLCAPE $\left[\frac{J}{kg}\right]$', 'min_MUCIN':'Min. MUCIN', 'min_MLCIN':'Min. MLCIN',
                  'initial_max_MLCAPE180':r'Initial Max. MLCAPE $\left[\frac{J}{kg}\right]$', 'initial_avg_MLCAPE180':r'Initial Avg. MLCAPE $\left[\frac{J}{kg}\right]$',
                  'initial_max_EBS_magnitude':r'Initial Max. EBS $\left[\frac{m}{s}\right]$', 'initial_avg_EBS_magnitude':r'Initial Avg. EBS $\left[\frac{m}{s}\right]$'}
    labels = np.array([['TPs w/ Lightning', 'TPs w/o Lightning'],['FPs w/ Lightning', 'FPs w/o Lightning']])
    #colors = np.array([['Oranges', 'Blues'],['Reds', 'Greens']])

    cmap_holder = matplotlib.colormaps.get_cmap('Oranges')
    cmap_colors = cmap_holder(np.linspace(0.08, 1.0, 256))
    orange_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)
    cmap_holder = matplotlib.colormaps.get_cmap('Blues')
    cmap_colors = cmap_holder(np.linspace(0.08, 1.0, 256))
    blue_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)
    cmap_holder = matplotlib.colormaps.get_cmap('Reds')
    cmap_colors = cmap_holder(np.linspace(0.08, 1.0, 256))
    red_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)
    cmap_holder = matplotlib.colormaps.get_cmap('Greens')
    cmap_colors = cmap_holder(np.linspace(0.08, 1.0, 256))
    green_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)
    colors = np.array([[orange_cmap, blue_cmap], [red_cmap, green_cmap]])

    for i in range(nx):
        for j in range(ny):
            if labels[i,j] == 'TPs w/ Lightning':
                use_df = tp_wl
            elif labels[i,j] == 'FPs w/ Lightning':
                use_df = fp_wl
            elif labels[i,j] == 'TPs w/o Lightning':
                use_df = tp_wol
            else:
                use_df = fp_wol
            hplt = ax[i,j].hist2d(use_df[xvar], use_df[yvar], bins=(np.arange(xmin, xmax, stepx), np.arange(ymin, ymax, stepy)), cmin=1, cmap=colors[i,j])
            if xvar == 'max_dBZ_-10C':
                multiple = 2
            elif xvar == 'min_C13':
                multiple = 4
            elif xvar == 'max_MLCAPE180':
                multiple = 2
            else:
                multiple = 1
            ax[i,j].set_xticks(np.arange(xmin, xmax+1, stepx*multiple))

            if j == 0:
                ax[i, j].set_ylabel(var_labels[yvar], fontsize=12)
                ax[i, j].tick_params(labelsize=12)
            if i == 1:
                ax[i, j].set_xlabel(var_labels[xvar], fontsize=12)
                ax[i, j].tick_params(labelsize=10)

            if labels[i,j] == 'FPs w/ Lightning':
                cbar = fig.colorbar(hplt[3], ax=ax[i, j], ticks=np.arange(0,len(use_df.index)+1, 1))
            elif labels[i,j] == 'FPs w/o Lightning' and xvar == 'max_MLCAPE180':
                cbar = fig.colorbar(hplt[3], ax=ax[i, j], ticks=np.arange(0, len(use_df.index) + 1, 100))
            elif labels[i,j] == 'TPs w/o Lightning' and xvar == 'max_MLCAPE180':
                cbar = fig.colorbar(hplt[3], ax=ax[i, j], ticks=np.arange(0, len(use_df.index) + 1, 100))
            else:
                cbar = fig.colorbar(hplt[3], ax=ax[i, j], ticks=np.arange(0, len(use_df.index) + 1, 50))

            if yvar == 'min_C13':
                lx = np.linspace(xmin, xmax+1, stepx)
                ly = [273.15 for x in lx]
                ax[i,j].plot(lx, ly, color='black', marker='none', linestyle='--')
            if xvar == 'min_C13':
                ly = np.linspace(ymin, ymax+1, stepy*10)
                lx = [273.15 for y in ly]
                ax[i, j].plot(lx, ly, color='black', marker='none', linestyle='--')
            if xvar == 'max_MLCAPE180':
                ax[i,j].ticklabel_format(axis='x', style='sci', scilimits=(3, 3), useMathText=True)

            cbar.set_label('Number of ' + labels[i,j], fontsize=12, labelpad=8)
            cbar.ax.tick_params(labelsize=10)
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    return

def get_reg_plot(sub_df, figname, xvar, yvar):
    fig_width_cm = 25
    fig_height_cm = 25
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]
    fig, ax = plt.subplots(figsize=fig_size)

    fp_wl = sub_df[(sub_df['lightning'] == 1) & (sub_df['class'] == 'fp')]
    fp_wol = sub_df[(sub_df['lightning'] == 0) & (sub_df['class'] == 'fp')]
    tp_wl = sub_df[(sub_df['lightning'] == 1) & (sub_df['class'] == 'tp')]
    tp_wol = sub_df[(sub_df['lightning'] == 0) & (sub_df['class'] == 'tp')]

    if 'min_C13' in yvar:
        lx = np.linspace(0, 65, 10)
        ly = [273.15 for x in lx]
        ax.plot(lx, ly, color='black', marker='none', linestyle='-')
    if 'min_C13' in xvar:
        ly = np.linspace(190, 290, 10)
        lx = [273.15 for y in ly]
        ax.plot(lx, ly, color='black', marker='none', linestyle='-')
    ax.plot(fp_wol[xvar], fp_wol[yvar], color='tab:green', marker='v', linestyle='none', label = 'FP w/o Lightning')
    ax.plot(fp_wl[xvar], fp_wl[yvar], color='tab:red', marker='s', linestyle='none', label = 'FP w/ Lightning')
    ax.plot(tp_wol[xvar], tp_wol[yvar], color='tab:blue', marker='*', linestyle='none', label = 'TP w/o Lightning')
    ax.plot(tp_wl[xvar], tp_wl[yvar], color = 'tab:orange', marker='.', linestyle='none', label = 'TP w/ Lightning')
    ax.legend()
    if 'max_dBZ_-10C' in xvar:
        ax.set_xlabel(r'Maximum Radar at -10$^{\circ}$C [dBZ]', fontsize=14)
        ax.set_xlim(0, 65)
    if 'min_C13' in yvar:
        ax.set_ylabel('Minimum C13 BT [K]', fontsize=14)
    ax.tick_params(labelsize=12)
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    return

def bars_month_region(df, outdir):
    info = {'months': {'04': 0, '05': 0, '06': 0, '07': 0, '08': 0, '09': 0},
            'regions': {'northwest': 0, 'west': 0, 'west_north_central': 0, 'southwest': 0, 'east_north_central': 0,
                        'south': 0, 'southeast': 0, 'central': 0, 'northeast': 0, 'oconus': 0}}

    df['month'] = pd.DatetimeIndex(df['object_start_time']).month
    for key in info['months'].keys():
        month_df = df[pd.DatetimeIndex(df['object_start_time']).month == int(key)]
        info['months'][key] = len(month_df.index)
    for key in info['regions'].keys():
        region_df = df[df['initial_region'] == key]
        info['regions'][key] = len(region_df.index)

    fig_width_cm = 20
    fig_height_cm = 20
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]

    fig, ax = plt.subplots()
    month_labels = info['months'].keys()
    xmonth = np.arange(len(month_labels))
    ymonth = info['months'].values()
    ax.bar(xmonth, ymonth)
    ax.set_ylabel('Number of ThunderCast Tracks')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
    ax.set_xlabel('Month')
    ax.set_xticks(xmonth)
    ax.set_xticklabels(month_labels)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    figname = outdir + 'cases_per_month.png'
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')

    fig2, ax2 = plt.subplots(figsize = fig_size)
    region_labels = ['Northwest', 'West', 'West North Central', 'Southwest', 'East North Central', 'South', 'Southeast',
                     'Central', 'Northeast', 'OCONUS']
    xregion = np.arange(len(region_labels))
    yregion = info['regions'].values()
    rects = ax2.bar(xregion, yregion, color = 'tab:purple')
    ax2.set_ylabel('Number of ThunderCast Tracks')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(3,3), useMathText=True)
    ax2.set_xlabel('U.S. Climate Region')
    ax2.set_xticks(xregion)
    ax2.set_xticklabels(region_labels, rotation=45, ha='right')
    #ax2.set_ylim(0, 4.3e4)
    ax2.xaxis.label.set_fontsize(12)
    ax2.yaxis.label.set_fontsize(12)
    # for j, rect in enumerate(rects):
    #     # if j == 0 or j == 1:
    #     height = rect.get_height()
    #     print('height', height)
    #     ax2.text(rect.get_x() + rect.get_width() / 2., height + 225, '%.1e' % int(height), ha='center', va='bottom',
    #              rotation=90, fontsize=8)
    fig2.tight_layout(pad=2)
    figname = outdir + 'cases_per_region.png'
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')
    return

def bars_month_region_class(df, outdir):
    info = {'months': {'04': 0, '05': 0, '06': 0, '07': 0, '08': 0, '09': 0},
            'regions': {'northwest': 0, 'west': 0, 'west_north_central': 0, 'southwest': 0, 'east_north_central': 0,
                        'south': 0, 'southeast': 0, 'central': 0, 'northeast': 0, 'oconus': 0}}

    df['month'] = pd.DatetimeIndex(df['object_start_time']).month

    class_by_month = {'TPs w/ Lightning':[], 'TPs w/o Lightning':[], 'FPs w/ Lightning':[], 'FPs w/o Lightning':[]}
    for key in info['months'].keys():
        month_df = df[df['month'] == int(key)]
        info['months'][key] = len(month_df.index)
        for c in class_by_month.keys():
            if c == 'TPs w/ Lightning':
                cdf = month_df[(month_df['class'] == 'tp') & (month_df['lightning'] == 1)]
            if c == 'TPs w/o Lightning':
                cdf = month_df[(month_df['class'] == 'tp') & (month_df['lightning'] == 0)]
            if c == 'FPs w/ Lightning':
                cdf = month_df[(month_df['class'] == 'fp') & (month_df['lightning'] == 1)]
            if c == 'FPs w/o Lightning':
                cdf = month_df[(month_df['class'] == 'fp') & (month_df['lightning'] == 0)]
            m_list = class_by_month[c]
            #m_list.append((len(cdf.index) / len(month_df.index))*100)
            m_list.append(len(cdf.index))
            class_by_month[c] = m_list

    colors = {'TPs w/ Lightning':'tab:orange', 'TPs w/o Lightning':'tab:blue', 'FPs w/ Lightning':'tab:red', 'FPs w/o Lightning':'tab:green'}
    class_by_region = {'TPs w/ Lightning': [], 'TPs w/o Lightning': [], 'FPs w/ Lightning': [], 'FPs w/o Lightning': []}
    for key in info['regions'].keys():
        region_df = df[df['initial_region'] == key]
        info['regions'][key] = len(region_df.index)
        for c in class_by_region.keys():
            if c == 'TPs w/ Lightning':
                cdf = region_df[(region_df['class'] == 'tp') & (region_df['lightning'] == 1)]
            if c == 'TPs w/o Lightning':
                cdf = region_df[(region_df['class'] == 'tp') & (region_df['lightning'] == 0)]
            if c == 'FPs w/ Lightning':
                cdf = region_df[(region_df['class'] == 'fp') & (region_df['lightning'] == 1)]
            if c == 'FPs w/o Lightning':
                cdf = region_df[(region_df['class'] == 'fp') & (region_df['lightning'] == 0)]
            r_list = class_by_region[c]
            r_list.append(len(cdf.index))
            #r_list.append((len(cdf.index) / len(region_df.index)) * 100)
            class_by_region[c] = r_list

    fig_width_cm = 18
    fig_height_cm = 18
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]
    fig, ax = plt.subplots(figsize=fig_size)

    month_labels = info['months'].keys()
    region_labels = ['Northwest', 'West', 'West North Central', 'Southwest', 'East North Central', 'South', 'Southeast',
                     'Central', 'Northeast', 'OCONUS']

    x = np.arange(len(month_labels))
    width = 0.7 #width of bars
    bottom = np.zeros(len(month_labels))
    for attribute, measurement in class_by_month.items():
        rects = ax.bar(x, measurement, width, label=attribute, color=colors[attribute], bottom=bottom)
        #ax.bar_label(rects, padding=3)
        bottom += measurement

    ax.set_ylabel(r'Number of ThunderCast Tracks')
    #ax.set_ylabel(r'% of Predicted Storm Tracks $\left[100*\frac{\text{tracks in month}}{\text{total in month}}\right]$')
    ax.set_xlabel('Month')
    ax.set_xticks(x, month_labels)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
    ax.tick_params(labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.legend(loc='upper left', fontsize=14)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    figname = outdir + 'tp_fps_cases_per_month.png'
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')

    fig_width_cm = 20
    fig_height_cm = 23
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(region_labels))
    width = 0.7  # width of bars
    multiplier = 0
    bottom = np.zeros(len(region_labels))
    for attribute, measurement in class_by_region.items():
        offset = width * multiplier
        #rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute])
        rects = ax.bar(x, measurement, width, label=attribute, color=colors[attribute], bottom=bottom)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
        bottom += measurement

    ax.set_ylabel(r'Number of ThunderCast Tracks')
    #ax.set_ylabel(r'% of Predicted Storm Tracks $\left[100*\frac{\text{tracks in region}}{\text{total in region}}\right]$')
    ax.set_xlabel('U.S. Climate Region')
    ax.set_xticks(x, region_labels, rotation=45, ha='right')
    #ax.set_xticks(x+width+0.5*width, region_labels, rotation=45, ha='right')
    ax.set_ylim(0, 5e3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
    ax.tick_params(labelsize=16)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.legend(loc='upper left', ncols=1, fontsize=14)
    fig.tight_layout(pad=2)
    figname = outdir + 'tp_fp_cases_per_region.png'
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')
    return

def gridded_avg(segmentation_file, df, var, agg, outdir, legx=45, legy=-2, left=0.125, bottom=.25, width=0.38, height=0.03, vmin=0, vmax=90):
    #based on https://james-brennan.github.io/posts/fast_gridding_geopandas/
    var_labels = {'lead_time_30dBZ':'Lead Time to 30 dBZ [min]', 'lead_time_eni':'Lead Time to ENTLN [min]',
                  'lead_time_glm':'Lead Time to GLM [min]'}
    df = df[['initial_lon', 'initial_lat', var]].dropna()
    if var == 'lead_time_30dBZ' or 'lead_time_eni' or 'lead_time_glm':
        df[var] = df[var]/60
    segment_info = xr.open_dataset(segmentation_file, engine='netcdf4')
    pred_area = pr.geometry.AreaDefinition(segment_info.attrs['area_id'], segment_info.attrs['description'],
                                           segment_info.attrs['proj_id'],
                                           segment_info.attrs['projection'], segment_info.attrs['width'],
                                           segment_info.attrs['height'], segment_info.attrs['area_extent'])
    proj_dict = pr.utils.proj4.proj4_str_to_dict(pred_area.proj_str)
    geoproj = ccrs.Geostationary(central_longitude=proj_dict['lon_0'], sweep_axis=proj_dict['sweep'],
                                 satellite_height=proj_dict['h'])
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': geoproj})
    df['x'], df['y'] = pred_area.get_projection_coordinates_from_lonlat(df['initial_lon'], df['initial_lat'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs=geoproj)
    gdf = gdf.drop(columns=['initial_lon', 'initial_lat', 'x', 'y'])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_extent([xmin, xmax, ymin, ymax], crs=geoproj)

    n_cells = 60
    cell_size = (xmax - xmin) / n_cells
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            # bounds
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            grid_cells.append(sgeom.box(x0, y0, x1, y1))
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=geoproj)

    merged = gpd.sjoin(gdf, cell, how='left', op='within')
    dissolve = merged.dissolve(by='index_right', aggfunc=agg)
    cell.loc[dissolve.index, var] = dissolve[var].values
    cbaxes = fig.add_axes([left, bottom, width, height]) # left bottom width height
    test = cell.plot(column=var,cmap='viridis', ax=ax, vmin=vmin, vmax=vmax, legend=True, cax=cbaxes, legend_kwds={'orientation': 'horizontal', 'extend':'both'})#, legend_kwds={'label': 'Lead Time to 30 dBZ'})
    cbaxes.tick_params(labelsize=12)
    cbaxes.text(legx,legy, var_labels[var], fontsize=14, rotation='horizontal', horizontalalignment='center')
    figname = outdir + 'gridded_' + str(var)
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')
    return

def gridded4(segmentation_file, sub_df, var, agg, outdir, figname, vmin=None, vmax=None, colormap='viridis', wspace=0.1, hspace=0.1, left=0.125, bottom=.1, width=0.38, height=0.03, legx=30, legy=-2, leg_align='center'):
    segment_info = xr.open_dataset(segmentation_file, engine='netcdf4')
    pred_area = pr.geometry.AreaDefinition(segment_info.attrs['area_id'], segment_info.attrs['description'],
                                           segment_info.attrs['proj_id'],
                                           segment_info.attrs['projection'], segment_info.attrs['width'],
                                           segment_info.attrs['height'], segment_info.attrs['area_extent'])
    proj_dict = pr.utils.proj4.proj4_str_to_dict(pred_area.proj_str)
    geoproj = ccrs.Geostationary(central_longitude=proj_dict['lon_0'], sweep_axis=proj_dict['sweep'],
                                 satellite_height=proj_dict['h'])

    fig_width_cm = 30
    fig_height_cm = 25
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]
    nx = 2
    ny = 2
    fig, ax = plt.subplots(nx, ny, figsize=fig_size, subplot_kw={'projection': geoproj})

    if var == 'class':
        sub_df = sub_df[['initial_lon', 'initial_lat', 'lightning', var]].dropna()
    else:
        sub_df = sub_df[['initial_lon', 'initial_lat', 'lightning', 'class', var]].dropna()

    sub_df['x'], sub_df['y'] = pred_area.get_projection_coordinates_from_lonlat(sub_df['initial_lon'],
                                                                                sub_df['initial_lat'])
    gdf = gpd.GeoDataFrame(sub_df, geometry=gpd.points_from_xy(sub_df['x'], sub_df['y']), crs=geoproj)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    n_cells = 50
    cell_size = (xmax - xmin) / n_cells
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            # bounds
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            grid_cells.append(sgeom.box(x0, y0, x1, y1))

    fp_wl = gdf[(gdf['lightning'] == 1) & (gdf['class'] == 'fp')]
    fp_wol = gdf[(gdf['lightning'] == 0) & (gdf['class'] == 'fp')]
    tp_wl = gdf[(gdf['lightning'] == 1) & (gdf['class'] == 'tp')]
    tp_wol = gdf[(gdf['lightning'] == 0) & (gdf['class'] == 'tp')]

    var_labels = {'min_C13': r'Minimum 10.3 $\mu$m BT [K]', 'C05': r'1.6 $\mu$m Reflectance at Min. 10.3 $\mu$m BT [K]',
                  'C02': r'0.64 $\mu$m Reflectance at Min. 10.3 $\mu$m BT',
                  'C11C13_BTD': r'8.4 $\mu$m - 10.3 $\mu$m at Min. 10.3 $\mu$m BT [K]',
                  'C11C14_BTD': r'8.4 $\mu$m - 11.2 $\mu$m at Min. 10.3 $\mu$m BT [K]',
                  'C13C15_BTD': r'10.3 $\mu$m - 12.3 $\mu$m at Min. 10.3 $\mu$m BT [K]',
                  'max_EBS_magnitude': r'Maximum EBS $\left[\frac{m}{s}\right]$',
                  'max_dBZ_-10C': r'Maximum Radar at -10$^{\circ}$C', 'max_MUCAPE': r'Max. MUCAPE $\left[\frac{J}{kg}\right]$',
                  'max_MLCAPE180': r'Max. MLCAPE $\left[\frac{J}{kg}\right]$', 'min_MUCIN': r'Min. MUCIN $\left[\frac{J}{kg}\right]$', 'min_MLCIN': r'Min. MLCIN $\left[\frac{J}{kg}\right]$',
                  'initial_max_MLCAPE180':r'Initial Max. MLCAPE $\left[\frac{J}{kg}\right]$', 'initial_avg_MLCAPE180':r'Initial Avg. MLCAPE $\left[\frac{J}{kg}\right]$',
                  'initial_max_EBS_magnitude':r'Initial Max. EBS $\left[\frac{m}{s}\right]$', 'initial_avg_EBS_magnitude':r'Initial Avg. EBS $\left[\frac{m}{s}\right]$'}
    labels = np.array([['TPs w/ Lightning', 'TPs w/o Lightning'], ['FPs w/ Lightning', 'FPs w/o Lightning']])
    colors = np.array([['Oranges', 'Blues'], ['Reds', 'Greens']])
    for i in range(nx):
        for j in range(ny):
            ax[i,j].add_feature(cfeature.COASTLINE)
            ax[i,j].add_feature(cfeature.STATES)
            ax[i, j].set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
            if labels[i, j] == 'TPs w/ Lightning':
                use_gdf = tp_wl
            elif labels[i, j] == 'FPs w/ Lightning':
                use_gdf = fp_wl
            elif labels[i, j] == 'TPs w/o Lightning':
                use_gdf = tp_wol
            else:
                use_gdf = fp_wol

            use_gdf = use_gdf.drop(columns=['initial_lon', 'initial_lat', 'x', 'y', 'lightning', 'class'])

            cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=geoproj)
            merged = gpd.sjoin(use_gdf, cell, how='left', op='within')
            if agg == 'count':
                merged['num'] = 1
            dissolve = merged.dissolve(by='index_right', aggfunc=agg)

            if var == 'class':
                cell.loc[dissolve.index, 'num'] = dissolve.num.values
                test = cell.plot(column='num', cmap=colors[i,j], ax=ax[i,j], legend=True, legend_kwds={'orientation': 'horizontal', 'label': '# of ' + labels[i,j], 'pad':0.1})
            else:
                cell.loc[dissolve.index, var] = dissolve[var].values
                if i == 0 and j == 0:
                    cbaxes = fig.add_axes([left, bottom, width, height])  # left bottom width height
                    if var == 'max_dBZ_-10C':
                        test = cell.plot(column=var, cmap=colormap, ax=ax[i, j], norm=Normalize(-1, 80), antialiased=True, vmin=vmin, vmax=vmax, legend=True, cax=cbaxes, legend_kwds={'orientation': 'horizontal'})
                    else:
                        test = cell.plot(column=var, cmap=colormap, ax=ax[i, j], vmin=vmin, vmax=vmax, legend=True, cax=cbaxes, legend_kwds={'orientation': 'horizontal'})
                    cbaxes.tick_params(labelsize=10)
                    cbaxes.text(legx, legy, 'Gridded Average of ' + var_labels[var], fontsize=12, rotation='horizontal', horizontalalignment=leg_align)
                else:
                    if var == 'max_dBZ_-10C':
                        test = cell.plot(column=var, norm=Normalize(-1, 80), vmin=vmin, vmax=vmax, cmap=colormap, ax=ax[i, j], legend=False, antialiased=True)#, legend_kwds={'orientation': 'horizontal'})
                    else:
                        test = cell.plot(column=var, vmin=vmin, vmax=vmax, cmap=colormap, ax=ax[i, j], legend=False)#, legend_kwds={'orientation': 'horizontal'})
                ax[i, j].set_title(labels[i, j])#, pad=0.05)
                ax[i,j].title.set_size(14)


    figname = outdir + figname
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')
    return

def get_leadtime(sub_df, figname):
    regions = ['northwest', 'west', 'west_north_central', 'southwest', 'east_north_central', 'south', 'southeast',
               'central', 'northeast', 'oconus', 'all']
    pred_thresholds = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
    mean_dict = {'dBZ':{}, 'glm':{}, 'eni':{}}
    median_dict = {'dBZ': {}, 'glm': {}, 'eni': {}}
    for region in regions:
        if region == 'all':
            region_df = sub_df
        else:
            region_df = sub_df[sub_df['initial_region'] == region]
        for k in mean_dict.keys():
            means = []
            medians = []
            for thresh in pred_thresholds:
                if k == 'dBZ':
                    col_name = 'lead_time_30dBZ_pred' + thresh
                if k == 'glm':
                    col_name = 'lead_time_glm_pred' + thresh
                if k == 'eni':
                    col_name = 'lead_time_eni_pred' + thresh
                means.append(np.nanmean(region_df[col_name]/60))
                medians.append(np.nanmedian(region_df[col_name]/60))
                del col_name
            mean_dict[k][region] = means
            median_dict[k][region] = medians

    ######MEDIAN
    ##Initial plot formatting
    fig_width_cm = 15
    fig_height_cm = 15
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]

    fig = plt.figure()
    fig.set_size_inches(fig_size)

    widths = [1.5]
    heights = [5, 5, 5]
    gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.2, width_ratios=widths, height_ratios=heights)
    axes = []
    num_axes = np.arange(0, 3, 1)
    regions = ['northwest', 'west', 'west_north_central', 'southwest', 'east_north_central', 'south', 'southeast',
               'central', 'northeast', 'oconus', 'all']
    legend_key = ['Northwest', 'West', 'West North Central', 'Southwest', 'East North Central', 'South', 'Southeast',
                  'Central', 'Northeast', 'OCONUS', 'All']
    colors = {'northwest': 'tab:red', 'west': 'tab:brown', 'west_north_central': 'tab:orange',
             'southwest': 'tab:green', 'east_north_central': 'tab:purple', 'south': 'tab:blue',
             'southeast': 'tab:pink', 'central': 'tab:olive', 'northeast': 'tab:cyan', 'oconus': 'tab:gray', 'all':'black'}
    plot_keys = list(mean_dict.keys())
    y_labels = ['LT to 30 dBZ [min]', 'LT to GLM [min]', 'LT to ENTLN [min]']
    #x_tick_labels = [str(int(x*100)) for x in np.round(np.arange(0, 1, 0.1), 1)]
    # plot_limits = {'dBZ': np.arange(.7, 1.05, .1), 'glm': np.arange(0, 1.02, 0.2),
    #                'eni': np.arange(0, .92, .2)}
    for i in range(len(num_axes)):
        plot_val = plot_keys[i]
        #ytick_vals = plot_limits[plot_val]
        print('Median', plot_val, median_dict[plot_val]['all'])
        for j in range(len(regions)):
            region = regions[j]
            x = [int(x) for x in pred_thresholds]
            y = median_dict[plot_val][region]
            color = colors[region]
            if j == 0:
                axes.append(fig.add_subplot(gs[num_axes[i]]))
            axes[-1].plot(x, y, color=color, marker='.', linestyle='solid', label=legend_key[j])
            axes[-1].set_ylabel(y_labels[i])
            if i == (len(num_axes) - 1):
                # axes[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
                # plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                axes[-1].set_xlabel('Probability Threshold [%]', labelpad=10)
            else:
                axes[-1].xaxis.set_ticklabels([])
                # if i == 0:
                #     axes[-1].set_title(title_string, pad=10)
                #     axes[-1].title.set_fontsize(14)
                # if i == 1:
                #     plt.axhline(convect_dbz, color='tab:brown', ls='--')
            # plt.xlim(datetime(2022, 7, 4, 15, 30), datetime(2022, 7, 4, 21, 30))
            # plt.axvline(lightning_time, color='tab:brown', ls='--')
        #plt.yticks(ytick_vals)
        #plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(1.2, .85, 0.23, 0.04), fancybox=True,
               ncol=1)  # left bottom width height
    figure_path = outdir + 'median_leadtimes.png'
    print(figure_path)
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)

    ##Initial plot formatting
    fig_width_cm = 15
    fig_height_cm = 15
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]

    fig = plt.figure()
    fig.set_size_inches(fig_size)

    widths = [1.5]
    heights = [5, 5, 5]
    gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.2, width_ratios=widths, height_ratios=heights)
    axes = []
    num_axes = np.arange(0, 3, 1)
    regions = ['northwest', 'west', 'west_north_central', 'southwest', 'east_north_central', 'south', 'southeast',
               'central', 'northeast', 'oconus', 'all']
    legend_key = ['Northwest', 'West', 'West North Central', 'Southwest', 'East North Central', 'South', 'Southeast',
                  'Central', 'Northeast', 'OCONUS', 'All']
    colors = {'northwest': 'tab:red', 'west': 'tab:brown', 'west_north_central': 'tab:orange',
              'southwest': 'tab:green', 'east_north_central': 'tab:purple', 'south': 'tab:blue',
              'southeast': 'tab:pink', 'central': 'tab:olive', 'northeast': 'tab:cyan', 'oconus': 'tab:gray',
              'all': 'black'}
    plot_keys = list(mean_dict.keys())
    y_labels = ['LT to 30 dBZ [min]', 'LT to GLM [min]', 'LT to ENTLN [min]']
    # x_tick_labels = [str(int(x*100)) for x in np.round(np.arange(0, 1, 0.1), 1)]
    # plot_limits = {'dBZ': np.arange(.7, 1.05, .1), 'glm': np.arange(0, 1.02, 0.2),
    #                'eni': np.arange(0, .92, .2)}
    for i in range(len(num_axes)):
        plot_val = plot_keys[i]
        # ytick_vals = plot_limits[plot_val]
        for j in range(len(regions)):
            region = regions[j]
            x = [int(x) for x in pred_thresholds]
            y = mean_dict[plot_val][region]
            color = colors[region]
            if j == 0:
                axes.append(fig.add_subplot(gs[num_axes[i]]))
            axes[-1].plot(x, y, color=color, marker='.', linestyle='solid', label=legend_key[j])
            axes[-1].set_ylabel(y_labels[i])
            if i == (len(num_axes) - 1):
                # axes[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
                # plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                axes[-1].set_xlabel('Probability Threshold [%]', labelpad=10)
            else:
                axes[-1].xaxis.set_ticklabels([])
                # if i == 0:
                #     axes[-1].set_title(title_string, pad=10)
                #     axes[-1].title.set_fontsize(14)
                # if i == 1:
                #     plt.axhline(convect_dbz, color='tab:brown', ls='--')
            # plt.xlim(datetime(2022, 7, 4, 15, 30), datetime(2022, 7, 4, 21, 30))
            # plt.axvline(lightning_time, color='tab:brown', ls='--')
        # plt.yticks(ytick_vals)
        # plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(1.2, .85, 0.23, 0.04), fancybox=True,
               ncol=1)  # left bottom width height
    figure_path = outdir + 'mean_leadtimes.png'
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)
    return

def dBZ_area_plot(sub_df, outdir):
    class_names = ['TPs w/ Lightning', 'TPs w/o Lightning']
    colors = ['tab:orange', 'tab:blue']
    categories = ['max_dBZ_area_30dBZ', 'max_dBZ_area_35dBZ', 'max_dBZ_area_40dBZ', 'max_dBZ_area_45dBZ', 'max_dBZ_area_50dBZ']
    x_labels = [r'Area $\geq$ 30 dBZ [km$^2$]', r'Area $\geq$ 35 dBZ [km$^2$]', r'Area $\geq$ 40 dBZ [km$^2$]', r'Area $\geq$ 45 dBZ [km$^2$]', r'Area $\geq$ 50 dBZ [km$^2$]']

    # colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:pink',
    #           'tab:blue', 'tab:olive', 'tab:brown', 'tab:cyan', 'tab:gray']

    numrows = 1
    numcols = 4
    fig, axes = plt.subplots(numrows, numcols, figsize=(15,5))
    bins = np.arange(0, 65, 5)

    for i in range(numrows):
        for j in range(numcols):
            x_vals = []
            for c in class_names:
                if c == 'TPs w/ Lightning':
                    cdf = sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 1) & (sub_df[categories[j]] > 0)]
                    denom = len(sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 1)].index)
                    tp_wl_perc = str(int(np.round(100*(len(cdf.index)/denom), decimals=0)))
                elif c == 'TPs w/o Lightning':
                    cdf = sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 0) & (sub_df[categories[j]] > 0)]
                    denom = len(sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 0)].index)
                    tp_wol_perc = str(int(np.round(100*(len(cdf.index)/denom), decimals=0)))
                else:
                    tp_wol_perc = 0
                    tp_wl_perc = 0
                x_vals.append(cdf[categories[j]].to_numpy())
            axes[j].hist(x_vals, bins, histtype='bar', color=colors, density=False, rwidth=0.9, label=class_names)
            axes[j].ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
            if i == (numrows - 1):
                axes[j].set_xlabel(x_labels[j], fontsize=12)
            if j == 0:
                axes[j].set_ylabel('# of Tracks', fontsize=12)
            axes[j].tick_params(labelsize=10)
            axes[j].yaxis.offsetText.set_fontsize(10)
            if len(tp_wol_perc) == 3:
                wol_x = 0.8
                wol_y = 0.93
            if len(tp_wol_perc) == 2:
                wol_x = 0.83
                wol_y = 0.93
            if len(tp_wol_perc) == 1:
                wol_x = 0.87
                wol_y = 0.93
            if len(tp_wl_perc) == 3:
                wl_x = 0.8
                wl_y = 0.84
            if len(tp_wl_perc) == 2:
                wl_x = 0.83
                wl_y = 0.84
            if len(tp_wl_perc) == 1:
                wl_x = 0.87
                wl_y = 0.84

            axes[j].text(wol_x, wol_y, str(tp_wol_perc) + '%', color='black', transform = axes[j].transAxes, bbox=dict(boxstyle='square', facecolor='white', edgecolor='tab:blue'))
            axes[j].text(wl_x, wl_y, str(tp_wl_perc) + '%', color='black', transform=axes[j].transAxes, bbox=dict(boxstyle='square', facecolor='white', edgecolor='tab:orange'))

    plt.legend(loc='center right',  bbox_to_anchor=(0.8, 1.1, 0.23, 0.04), fancybox=True, ncol=2, fontsize=12) # left bottom width height
    plt.subplots_adjust(top=0.75)
    figname = outdir + 'dBZ_area_all.png'
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close('all')

    plot_region_splits = np.array([['northwest', 'west_north_central', 'central', 'east_north_central', 'northeast'],
                                   ['west', 'southwest', 'south', 'southeast', 'oconus']])
    plot_region_labels_splits = np.array([['Northwest', 'West North Central', 'Central', 'East North Central', 'Northeast'],
                                         ['West', 'Southwest', 'South', 'Southeast', 'OCONUS']])

    for split in range(plot_region_splits.shape[0]):
        regions = plot_region_splits[split]
        region_labels = plot_region_labels_splits[split]

        numrows = 5
        numcols = 4
        fig, axes = plt.subplots(numrows, numcols, figsize=(15,15))
        bins = np.arange(0, 60, 5)

        for i in range(numrows):
            for j in range(numcols):
                x_vals = []
                for c in class_names:
                    if c == 'TPs w/ Lightning':
                        cdf = sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 1) & (sub_df[categories[j]] > 0) & (sub_df['initial_region'] == regions[i])]
                        denom = len(sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 1) & (sub_df['initial_region'] == regions[i])].index)
                        tp_wl_perc = str(int(np.round(100*(len(cdf.index)/denom), decimals=0)))
                    elif c == 'TPs w/o Lightning':
                        cdf = sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 0) & (sub_df[categories[j]] > 0) & (sub_df['initial_region'] == regions[i])]
                        denom = len(sub_df[(sub_df['class'] == 'tp') & (sub_df['lightning'] == 0) & (sub_df['initial_region'] == regions[i])].index)
                        tp_wol_perc = str(int(np.round(100*(len(cdf.index)/denom), decimals=0)))
                    else:
                        tp_wol_perc = 0
                        tp_wl_perc = 0
                    x_vals.append(cdf[categories[j]].to_numpy())
                axes[i, j].hist(x_vals, bins, histtype='bar', color=colors, density=False, rwidth=0.9, label=class_names)
                axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(1, 1), useMathText=True)
                if i == (numrows - 1):
                    axes[i, j].set_xlabel(x_labels[j], fontsize=12)
                if j == 0:
                    axes[i, j].set_ylabel(region_labels[i] + '\n' + '# of Tracks', fontsize=12)
                axes[i, j].tick_params(labelsize=10)
                axes[i, j].yaxis.offsetText.set_fontsize(10)
                if len(tp_wol_perc) == 3:
                    wol_x = 0.8
                    wol_y = 0.89
                if len(tp_wol_perc) == 2:
                    wol_x = 0.83
                    wol_y = 0.89
                if len(tp_wol_perc) == 1:
                    wol_x = 0.87
                    wol_y = 0.89
                if len(tp_wl_perc) == 3:
                    wl_x = 0.8
                    wl_y = 0.74
                if len(tp_wl_perc) == 2:
                    wl_x = 0.83
                    wl_y = 0.74
                if len(tp_wl_perc) == 1:
                    wl_x = 0.87
                    wl_y = 0.74

                axes[i, j].text(wol_x, wol_y, str(tp_wol_perc) + '%', color='black', transform = axes[i, j].transAxes, bbox=dict(boxstyle='square', facecolor='white', edgecolor='tab:blue'))
                axes[i, j].text(wl_x, wl_y, str(tp_wl_perc) + '%', color='black', transform=axes[i, j].transAxes, bbox=dict(boxstyle='square', facecolor='white', edgecolor='tab:orange'))
                if (i == 0) and (j == numcols-1):
                    axes[i,j].legend(loc='center right',  bbox_to_anchor=(0.8, 1.2, 0.23, 0.04), fancybox=True, ncol=2, fontsize=12) # left bottom width height

        #plt.legend(loc='center right',  bbox_to_anchor=(0.8, 1.1, 0.23, 0.04), fancybox=True, ncol=2, fontsize=12) # left bottom width height
        plt.subplots_adjust(hspace = 0.4, top=0.92)
        if split == 0:
            figname = outdir + 'dBZ_area_northern_regions.png'
        else:
            figname = outdir + 'dBZ_area_southern_regions.png'
        plt.savefig(figname, dpi=200, bbox_inches='tight')
        plt.close('all')
    return


if __name__ == "__main__":
    months = ['04', '05', '06', '07', '08', '09']
    #avoid = ['20220710', '20220717', '20220724', '20220806', '20220818', '20220915', '20220920', '20220926']

    rootobjdir = '/ships22/grain/sortland/objects'
    outdir = rootobjdir + '/plots/'
    os.makedirs(outdir, exist_ok=True)

    CVD_cmap_all = matplotlib.colormaps.get_cmap('pyart_HomeyerRainbow')
    colors1 = CVD_cmap_all(np.linspace(0, 0.25, 97))
    colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 159))
    cmap_colors = np.vstack((colors1, colors2))
    CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)

    dates = []
    obj_files = {}
    segmentation_file = None
    for m in months:
        date_dirs = glob(rootobjdir + '/2022/2022' + m + '*')
        for i in date_dirs:
            date = os.path.basename(i)
            # #####
            #if date in avoid: continue
            # if date == '20220724': continue
            # #####
            dt = datetime.strptime(date, '%Y%m%d')
            dates.append(date)
            obj_files[dt] = dt.strftime(rootobjdir + '/%Y/%Y%m%d/object_data.pickle')
            if not segmentation_file:
                segmentation_file = dt.strftime(rootobjdir + '/%Y/%Y%m%d/segments.netcdf')

    #dates = dates[1:]

    all_dfs = []
    for key in obj_files.keys():
        df = pd.read_pickle(obj_files[key])
        # print(df[df['track_num']==1949]['time_first_30dBZ'])
        all_dfs.append(df)

    df = pd.concat(all_dfs)
    # print('Columns available:')
    # for col in df.columns:
    #     print(col)

    # print(df[df['track_num'] == 469]['initial_region'])
    # sys.exit()

    ####limit the dataset to have radar converage and to be non-oceanic
    limited_df = df[(df['no_radar_cov'] == 0) & (df['ocean'] == 0)]
    fp_df = limited_df[limited_df['class']=='fp']
    tp_df = limited_df[limited_df['class']=='tp']

    ds_stats = {'Number of dates': len(dates),
                'Number of Tracks': len(df.index),
                'Tracks in the Ocean (Omitting)': len(df[df['ocean'] == 1].index),
                'Tracks w/ radar coverage and non-ocean (only these used for rest)': len(limited_df.index),
                'FP Tracks': len(fp_df.index),
                'FP Tracks w/ Lightning': len(fp_df[fp_df['lightning'] == 1].index),
                'FP Tracks w/o Lightning': len(fp_df[fp_df['lightning'] == 0].index),
                'TP Tracks': len(tp_df.index),
                'TP Tracks w/ Lightning': len(tp_df[tp_df['lightning'] == 1].index),
                'TP Tracks w/o Lightning': len(tp_df[tp_df['lightning'] == 0].index)}

    #We don't need any of the stuff we are omitting going forward
    df = limited_df
    print(ds_stats)

    ##Get histogram of number of cases in each month and region
    bars_month_region(df, outdir)
    ##Get histogram of number of cases in each month and region sorted by class
    bars_month_region_class(df, outdir)

    ##Get histogram of distribution across maximum radar
    gen_hist(df, outdir, 'max_dBZ_-10C', 12, 0, 60, np.arange(0, 65, 5), r'Maximum Radar at $-10^{\circ}$C', 'max_radar_hist.png', sciy=True)

    ##Get plot of C13 VS Max. Radar
    #sub_df = df[(df['max_dBZ_-10C'] > -99) & (df['min_C13'] > 0) & (df['sol_zen'] < 85)]
    sub_df = df
    #Get plot of dBZ areas
    dBZ_area_plot(sub_df, outdir)
    #Get plot of lead time
    figname = outdir + 'leadtimes.png'
    get_leadtime(sub_df, outdir)
    #Get plot of MLCAPE and EBS
    figname = outdir + '2Dhist_EBS_MLCAPE.png'
    get_2Dhist(sub_df, figname, 'max_MLCAPE180', 'max_EBS_magnitude', 0, 5500, 0, 55, 500, 5)
    # figname = outdir + '2Dhist_init_max_EBS_MLCAPE.png'
    # get_2Dhist(sub_df, figname, 'initial_max_MLCAPE180', 'initial_max_EBS_magnitude', 0, 5500, 0, 55, 500, 5)
    # figname = outdir + '2Dhist_init_avg_EBS_MLCAPE.png'
    # get_2Dhist(sub_df, figname, 'initial_avg_MLCAPE180', 'initial_avg_EBS_magnitude', 0, 5500, 0, 55, 500, 5)

    ##Get plot of C13 VS Max. Radar
    #sub_df = df[(df['max_dBZ_-10C'] > -99) & (df['min_C13'] > 0) & (df['sol_zen'] < 85)]
    sub_df = df[(df['max_dBZ_-10C'] > -99) & (df['min_C13'] > 0)]
    figname = outdir + '2Dhist_C13min_maxRadar.png'
    get_2Dhist(sub_df, figname, 'max_dBZ_-10C', 'min_C13', 0, 65, 190, 290, 5, 5)
    figname = outdir + '2Dhist_C11C14_C13min.png'
    get_2Dhist(sub_df, figname, 'min_C13', 'C11C14_BTD', 190, 290, -12, 10, 5, 1)
    figname = outdir + '2Dhist_C11C13BTD_maxRadar.png'
    get_2Dhist(sub_df, figname, 'max_dBZ_-10C', 'C11C13_BTD', 0, 65, -15, 5, 5, 1)
    figname = outdir + '2Dhist_C11C14BTD_maxRadar.png'
    get_2Dhist(sub_df, figname, 'max_dBZ_-10C', 'C11C14_BTD', 0, 65, -12, 10, 5, 1)
    figname = outdir + '2Dhist_C13C15BTD_maxRadar.png'
    get_2Dhist(sub_df, figname, 'max_dBZ_-10C', 'C13C15_BTD', 0, 65, -5, 15, 5, 1)

    sub_df = df[(df['max_dBZ_-10C'] > -99) & (df['min_C13'] > 0) & (df['sol_zen'] < 85)]
    figname = outdir + '2Dhist_C05_maxRadar.png'
    get_2Dhist(sub_df, figname, 'max_dBZ_-10C', 'C05', 0, 65, -5, 100, 5, 5)
    figname = outdir + '2Dhist_C02_maxRadar.png'
    get_2Dhist(sub_df, figname, 'max_dBZ_-10C', 'C02', 0, 65, 0, 110, 5, 5)
    figname = outdir + '2Dhist_C05_C13min.png'
    get_2Dhist(sub_df, figname, 'min_C13', 'C05', 190, 285, 0, 70, 10, 5)

    #Keep Record of Important Values
    f = open((outdir + 'dataset_stats.txt'), 'w')
    for key, val in ds_stats.items():
        f.write('%s:%s\n' % (key, val))
    f.close()