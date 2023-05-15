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

#python torchlightning_predict.py -DT 2022-08-27-15-51 -t 2 -lat 33.98 -lon -83.46.26
os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
os.environ['SATPY_CONFIG_PATH'] = '/home/sortland/ci_ltg/PPP_config/'

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

def format_inputs(scn, norm_vals, vars, center_x_idx, center_y_idx, hs1km):
    X = []
    for res_group in vars:
        if res_group == 'target':
            continue
        else:
            x = None
            center_x_ind = int(center_x_idx)
            center_y_ind = int(center_y_idx)
            hs = hs1km
            ymin_index = center_y_ind + hs
            ymax_index = center_y_ind - hs
            xmin_index = center_x_ind - hs
            xmax_index = center_x_ind + hs
            area_def = scn['C05'].attrs['area']
            x_coords, y_coords = area_def.get_proj_vectors()
            center_y = y_coords[center_y_ind]
            center_x = x_coords[center_x_ind]
            xmin = x_coords[xmin_index]
            xmax = x_coords[xmax_index]
            ymin = y_coords[ymin_index]
            ymax = y_coords[ymax_index]
            crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
            for variable in vars[res_group]:
                tmp = torch.tensor(crop_scn[variable].values).double()
                mean_std_dict = norm_vals[variable]
                tmp = (tmp - mean_std_dict['mean']) / mean_std_dict['std']
                tmp = tmp[None]
                if x is None:
                    x = tmp + 0.
                else:
                    x = torch.cat((x, tmp))
            X.append(x.to(dtype=torch.float))

    finest_size = X[0].shape[-1]
    print(X[0].shape[-2], X[0].shape[-1])
    upsamp = nn.Upsample(size=(X[0].shape[-2], X[0].shape[-1]), mode='nearest')  # size here should match the size of the images at 0.5km resolution
    inp = X[0]
    for i in range(len(X)):
        if i != 0:
            inp_temp = torch.unsqueeze(X[i], 0)
            inp_temp = upsamp(inp_temp)
            inp_temp = torch.squeeze(inp_temp, 0)
            inp = torch.cat((inp, inp_temp), dim=0)
        else:
            continue
    #X = torch.unsqueeze(inp,0) #need to have an extra dimensions since not coming from data loader [1, 4, 800, 800]
    X = inp
    return X

def load_pt(file, params):
    X = torch.load(file)
    base = os.path.basename(file)
    parts = file.split('/')
    target_folder = '/'
    for f in range(len(parts)):
        if f != 0 and f <= len(parts) - 3:
            target_folder = target_folder + parts[f] + '/'
    target_file = target_folder + 'target/' + base
    y = torch.load(target_file)
    return X, y

def get_radar_list(dts, rootradardir):
    radar_file_list = []
    for dt in dts:
        radardir = dt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
        ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
        rad_end = dt + timedelta(minutes=5)
        radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(radardir, '*.netcdf'))}
        five_min_radfilelist = [radfiles[x] for x in radfiles if dt <= x <= rad_end]
        radar_file_list.append(five_min_radfilelist[0])
    return radar_file_list

def get_radar(radar_file, sat_grid):
    dataset_rad = Dataset(radar_file, 'r')
    nlat = len(dataset_rad.dimensions['Lat'])
    nlon = len(dataset_rad.dimensions['Lon'])
    NW_lat = getattr(dataset_rad, 'Latitude')
    NW_lon = getattr(dataset_rad, 'Longitude')
    dy = getattr(dataset_rad, 'LonGridSpacing')
    dx = getattr(dataset_rad, 'LatGridSpacing')
    radardataset = dataset_rad.variables['Reflectivity_-10C'][:, :]
    dataset_rad.close()

    print('remapping radar')
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
    print('remap complete')
    return remapped_radar

def plot_maxradar(outdir, patch_file, date):
    df = Dataset(patch_file, 'r')
    topdt = datetime.strptime(date, '%Y-%m-%d-%H-%M')
    goesnc = Dataset('/home/sortland/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)
    print(gip.sweep_angle_axis, 'sweep')
    state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')
    CVD_cmap_all = matplotlib.cm.get_cmap('pyart_HomeyerRainbow')
    colors1 = CVD_cmap_all(np.linspace(0, 0.25, 98))
    colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 158))
    cmap_colors = np.vstack((colors1, colors2))
    CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)
    rad_group = df.groups['MRMS_one_km']
    RAD = rad_group.variables['reflectivity_60min_neg10C_max'][:,:]

    x1km = df.groups['goes16_one_km']['x']
    y1km = df.groups['goes16_one_km']['y']
    xx1km, yy1km = np.meshgrid(x1km, y1km)

    for i in range(RAD.shape[0]):
        for j in range(RAD.shape[1]):
            if RAD[i, j] < 0:
                RAD[i, j] = np.nan

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0, 0, 1, 1], projection=geoproj)
    xmin = np.min(x1km);
    xmax = np.max(x1km);
    ymin = np.min(y1km);
    ymax = np.max(y1km)
    print('max and mins:')
    print('x', xmin, xmax)
    print('y', ymin, ymax)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(state_borders, edgecolor='tan')
    max_radimg = ax.pcolormesh(xx1km, yy1km, ma.masked_less(RAD, 10), cmap=CVD_cmap, norm=Normalize(-1, 80), linewidth=0,
                               antialiased=True)
    cbaxes2 = fig.add_axes([0.31, -0.05, 0.5, 0.03])  # left bottom width height
    cbar2 = plt.colorbar(max_radimg, orientation='horizontal', pad=0.01, extend='max', drawedges=0, ax=ax,
                         ticks=np.arange(10, 80, 10), cax=cbaxes2)
    cbar2.set_label('Maximum Reflectivity at -10C [dBz]', fontsize=18)
    cbar2.ax.tick_params(labelsize=16)
    cbar2.set_alpha(1.0)
    cbar2.draw_all()

    timestamp = str(topdt)
    ax.set_title((' ').join(['Maximum Reflectivity for +1 hour ', timestamp, 'UTC']), fontsize=20)
    # ax.set_title((' ').join(['MRMS', 'CONUS', timestamp, 'UTC']), fontsize=16)
    filepath = os.path.join(outdir, topdt.strftime('max_radar_ex_%Y_%m_%d_%H_%M_%j.png'))
    print(filepath)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close('all')
    df.close()
    return

def make_figure(scn, preds, center_x_idx, center_y_idx, hs, radar_file, c, timestamp, figure_path, full_US):
    goesnc = Dataset('/home/sortland/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)
    print(gip.sweep_angle_axis, 'sweep')
    state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')
    NWS_cmap = ctables.registry.get_colortable('NWSStormClearReflectivity')
    NWS_cmap_sub = cmr.get_sub_cmap(NWS_cmap, 0.51, 0.8)

    CVD_cmap_all = matplotlib.cm.get_cmap('pyart_HomeyerRainbow')
    colors1 = CVD_cmap_all(np.linspace(0, 0.25, 98))
    colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 158))
    cmap_colors = np.vstack((colors1, colors2))
    CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)

    if full_US:
        cropped_scn = scn
    else:
        ymin_index = int(center_y_idx + hs)
        ymax_index = int(center_y_idx - hs)
        xmin_index = int(center_x_idx - hs)
        xmax_index = int(center_x_idx + hs)
        x_coords, y_coords = scn['C05'].attrs['area'].get_proj_vectors()
        xmin = x_coords[xmin_index]
        xmax = x_coords[xmax_index]
        ymin = y_coords[ymin_index]
        ymax = y_coords[ymax_index]
        cropped_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
    #####Get one km
    C05vals = cropped_scn['C05'].values
    C13vals = cropped_scn['C13'].values
    C02vals = cropped_scn['C02'].values

    x1km = cropped_scn['C05']['x']
    y1km = cropped_scn['C05']['y']
    xx1km, yy1km = np.meshgrid(x1km, y1km)
    xhalfkm = cropped_scn['C02']['x']
    yhalfkm = cropped_scn['C02']['y']
    xxhalfkm, yyhalfkm = np.meshgrid(xhalfkm, yhalfkm)

    print(C05vals.shape, len(xx1km), len(yy1km))
    sat_grid = cropped_scn['C05'].attrs['area']

    # #Get radar
    radar_data = get_radar(radar_file, sat_grid)
    print(radar_data.shape)

    #make figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0, 0, 1, 1], projection=geoproj)

    xmin = np.min(x1km); xmax = np.max(x1km); ymin = np.min(y1km); ymax = np.max(y1km)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(state_borders, edgecolor='plum')

    if c == 'C05':
        img = ax.pcolormesh(xx1km, yy1km, C05vals, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
        cbaxes = fig.add_axes([0.01, -0.05, 0.27, 0.03])  # left bottom width height
        cbar = plt.colorbar(img, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0, ax=ax,
                            ticks=np.arange(0, 100, 10), cax=cbaxes)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('ABI 1.6-$\mu$m reflectance [%]', fontsize=14)
        radimg_10C = ax.pcolormesh(xx1km, yy1km, ma.masked_less(radar_data, 10), cmap=CVD_cmap,
                                   norm=Normalize(-1, 80),
                                   linewidth=0, antialiased=True, alpha=0.7)
        cbaxes2 = fig.add_axes([0.32, -0.05, 0.27, 0.03])  # left bottom width height
        cbar2 = plt.colorbar(radimg_10C, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                             ax=ax, ticks=np.arange(10, 80, 10), cax=cbaxes2)
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label('Radar Reflectivity at -10C [dBz]', fontsize=14)
        cbar2.set_alpha(1.0)
        cbar2.draw_all()

        colors = ['#FF9494', '#FF5F5F', '#DC1C1C', '#920000']
        values = [0.1, 0.3, 0.5, 0.8]
        # linewidths = [1.6, 1.6, 1.7, 1.8, 2]
        linewidths = [2.2, 2.4, 2.6, 3]
        labels = [str(int(v * 100)) + '%' for v in values]
        contour_img = ax.contour(xx1km, yy1km, preds, values, colors=colors,
                                 linewidths=linewidths)  # , cmap=plt.get_cmap('cividis'))
        custom_lines = [Line2D([0], [0], color=c, lw=lw * 2) for c, lw in zip(colors, linewidths)]
        leg2 = ax.legend(custom_lines, labels, fontsize=12, loc='lower right', handlelength=2, handleheight=0.9,
                         ncol=len(labels), bbox_to_anchor=(1, -0.055))
        ctext = plt.text(0.66, -0.09, 'Probability of Convection in 60 min', fontsize=14,
                         transform=ax.transAxes)
    elif c == 'C13':
        img = ax.pcolormesh(xx1km, yy1km, zoom(C13vals, 2, order=0), vmin=200, vmax=300, cmap=plt.get_cmap('Greys'))
        cbaxes = fig.add_axes([0.01, -0.05, 0.27, 0.03])  # left bottom width height
        cbar = plt.colorbar(img, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0, ax=ax,
                            ticks=np.arange(200, 300, 20), cax=cbaxes)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('ABI 10.3-$\mu$m BT [%]', fontsize=14)
        radimg_10C = ax.pcolormesh(xx1km, yy1km, ma.masked_less(radar_data, 10), cmap=CVD_cmap,
                                   norm=Normalize(-1, 80),
                                   linewidth=0, antialiased=True, alpha=0.7)
        cbaxes2 = fig.add_axes([0.32, -0.05, 0.27, 0.03])  # left bottom width height
        cbar2 = plt.colorbar(radimg_10C, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                             ax=ax, ticks=np.arange(10, 80, 10), cax=cbaxes2)
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label('Radar Reflectivity at -10C [dBz]', fontsize=14)
        cbar2.set_alpha(1.0)
        cbar2.draw_all()

        if full_US:
            colors = ['#FF9494', '#FF5F5F', '#DC1C1C', '#920000']
            values = [0.1, 0.3, 0.5, 0.8]
            # linewidths = [1.6, 1.6, 1.7, 1.8, 2]
            linewidths = [0.5, 0.5, 0.8, 1]
        else:
            colors = ['#FF9494', '#FF5F5F', '#DC1C1C', '#920000']
            #colors = ['#920000', '#DC1C1C', '#FF5F5F', '#FF9494']
            values = [0.1, 0.3, 0.5, 0.8]
            # linewidths = [1.6, 1.6, 1.7, 1.8, 2]
            linewidths = [2.2, 2.4, 2.6, 3]
        labels = [str(int(v * 100)) + '%' for v in values]
        contour_img = ax.contour(xx1km, yy1km, preds, values, colors=colors,
                                 linewidths=linewidths)  # , cmap=plt.get_cmap('cividis'))
        custom_lines = [Line2D([0], [0], color=c, lw=lw * 2) for c, lw in zip(colors, linewidths)]
        leg2 = ax.legend(custom_lines, labels, fontsize=12, loc='lower right', handlelength=2, handleheight=0.9,
                         ncol=len(labels), bbox_to_anchor=(1, -0.055))
        ctext = plt.text(0.66, -0.09, 'Probability of Convection in 60 min', fontsize=14,
                         transform=ax.transAxes)
    else:
        img = ax.pcolormesh(xxhalfkm, yyhalfkm, C02vals, vmin=0, vmax=100, cmap=plt.get_cmap('Greys_r'))
        cbaxes = fig.add_axes([0.01, -0.05, 0.27, 0.03])  # left bottom width height
        cbar = plt.colorbar(img, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0, ax=ax,
                            ticks=np.arange(0, 100, 10), cax=cbaxes)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('ABI 0.64-$\mu$m reflectance [%]', fontsize=14)
        # radimg_10C = ax.pcolormesh(xxhalfkm, yyhalfkm, ma.masked_less(zoom(radar_data, 2, order=0), 10), cmap=NWS_cmap_sub,
        #                            norm=Normalize(-1, 80),
        #                            linewidth=0, antialiased=True, alpha=0.5)
        radimg_10C = ax.pcolormesh(xxhalfkm, yyhalfkm, ma.masked_less(zoom(radar_data, 2, order=0), 10), cmap=CVD_cmap,
                                   norm=Normalize(-1, 80),
                                   linewidth=0, antialiased=True, alpha=0.5)
        cbaxes2 = fig.add_axes([0.32, -0.05, 0.27, 0.03])  # left bottom width height
        cbar2 = plt.colorbar(radimg_10C, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                             ax=ax, ticks=np.arange(10, 80, 10), cax=cbaxes2)
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label('Radar Reflectivity at -10C [dBz]', fontsize=14)
        cbar2.set_alpha(1.0)
        cbar2.draw_all()

        #colors = ['#FF8300', '#9700FF', '#00FFFB', '#FF00EC', '#0015FF']
        colors = ['#FF9494', '#FF5F5F', '#DC1C1C', '#920000']
        values = [0.1, 0.3, 0.5, 0.8]
        linewidths = [1.6, 1.6, 1.7, 1.8, 2]
        #linewidths = [2.2, 2.4, 2.6, 3]
        labels = [str(int(v * 100)) + '%' for v in values]
        contour_img = ax.contour(xxhalfkm, yyhalfkm, zoom(preds, 2, order=0), values, colors=colors,
                                 linewidths=linewidths)  # , cmap=plt.get_cmap('cividis'))
        custom_lines = [Line2D([0], [0], color=c, lw=lw * 2) for c, lw in zip(colors, linewidths)]
        leg2 = ax.legend(custom_lines, labels, fontsize=12, loc='lower right', handlelength=2, handleheight=0.9,
                         ncol=len(labels), bbox_to_anchor=(1, -0.055))
        ctext = plt.text(0.66, -0.09, 'Probability of Convection in 60 min', fontsize=14,
                         transform=ax.transAxes)


    ax.set_title((' ').join(['GOES-East', 'CONUS', timestamp, 'UTC']), fontsize=16)
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)
    return

def plot_rgb(scn, VISfilepath, IRfilepath, center_x_idx, center_y_idx, hs, timestamp):
    ymin_index = int(center_y_idx + hs)
    ymax_index = int(center_y_idx - hs)
    xmin_index = int(center_x_idx - hs)
    xmax_index = int(center_x_idx + hs)
    x_coords, y_coords = scn['C05'].attrs['area'].get_proj_vectors()
    xmin = x_coords[xmin_index]
    xmax = x_coords[xmax_index]
    ymin = y_coords[ymin_index]
    ymax = y_coords[ymax_index]
    crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
    # Get plotting grid
    sat_grid = crop_scn['C05'].attrs['area']

    # have the scene load the data of interest in a way we want to display it (RGB composites defined in PPP_config)
    # IR images
    IRcrop_scn = crop_scn
    IRcrop_scn.load(['IRCloudPhaseFC'])
    IRresamp = IRcrop_scn.resample(resampler='native')

    # Visible channel images
    VIScrop_scn = crop_scn
    VIScrop_scn.load(['DayCloudPhaseFC'])
    VISresamp = VIScrop_scn.resample(resampler='native')

    # Plot IR False Color Image:
    fig = plt.figure(figsize=(12, 12))
    IRimg_data = get_enhanced_image(IRresamp['IRCloudPhaseFC']).data
    IRimg_data.plot.imshow(rgb='bands')
    plt.title((' ').join([timestamp, 'UTC']))
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    # If you want to save the RGB file:
    plt.savefig(IRfilepath, bbox_inches='tight')
    print('IR RGB images were created')
    plt.close(fig)
    # Uncomment code below to save the data as a netcdf:
    # IRresamp.save_dataset(writer='simple_image', dataset_id='IRCloudPhaseFC', filename=IRfilepath)

    # Visible False Color Image:
    fig = plt.figure(figsize=(12, 12))
    VISimg_data = get_enhanced_image(VISresamp['DayCloudPhaseFC']).data
    VISimg_data.plot.imshow(rgb='bands')
    # Uncomment code below to save the data as a netcdf:
    # VISresamp.save_dataset(writer='simple_image', dataset_id='DayCloudPhaseFC', filename=VISfilepath)
    plt.title((' ').join([timestamp, 'UTC']))
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(VISfilepath, bbox_inches='tight')
    print('VIS RGB images created.')
    plt.close('all')

    # If you don't unload the scenes, the memory blows up when looping through multiple datetimes.
    VIScrop_scn.unload()
    crop_scn.unload()
    scn.unload()
    IRcrop_scn.unload()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes a datetime and gets prediction images.')
    parser.add_argument('-a', '--abidir',
                        help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/',
                        default=['/arcdata/goes/grb/goes16'], type=str, nargs=1)
    parser.add_argument('-r', '--radardir', help='Root directory with radar netcdfs.',
                        default=['/apollo/grain/saved_ci_ltg_data/radar'],
                        nargs=1, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/predictions/'])
    parser.add_argument('-DT', '--DT', help='A datetime in %Y-%m-%d-%H-%M',
                        type=str, nargs=1)
    parser.add_argument('-t', '--deltat', help='Time from given date to have predictions through in minutes.',
                        type=int, nargs=1)
    parser.add_argument('-lat', '--center_lat', help='Center latitude for prediction',
                        type=str, nargs=1)
    parser.add_argument('-lon', '--center_lon', help='Center longitude for prediction',
                        type=str, nargs=1)
    parser.add_argument('-hs', '--hs', help='Half size for prediction area indices',
                        type=int, nargs=1, default=[127])
    parser.add_argument('-c', '--c', help='Channel to plot',
                        type=str, nargs=1, default=['C02'])
    parser.add_argument('-mrad', '--mrad', help='Plot maximum radar from existing saved data netcdf.',
                        action='store_true')
    parser.add_argument('-rad', '--rad', help='Plot instantaneous radar from radar files.',
                        action='store_true')
    parser.add_argument('-go', '--gen_outdir', help='general data plot output directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/general_data_plots/'])
    parser.add_argument('-patchfile', '--patchfile', help='existing patch nc for plotting', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/data/2019/06/26/south/positive_cases/2019_06_26_18_26_177_-9221_3597_708.nc'])
    parser.add_argument('-rgb', '--rgb', help='Plot RGB',
                        action='store_true')

    args = parser.parse_args()

    if args.mrad:
        rootoutdir = args.gen_outdir[0]
        patch_file = args.patchfile[0]
        date = args.DT[0]
        plot_maxradar(rootoutdir, patch_file, date)
    elif args.rgb:
        rootoutdir = args.gen_outdir[0]
        rootdatadir = args.abidir[0]
        rootradardir = args.radardir[0]
        time_delta = args.deltat[0]
        date = args.DT[0]
        center_lon = float(args.center_lon[0])
        center_lat = float(args.center_lat[0])
        hs1km = args.hs[0]
    else:
        rootoutdir = args.outdir[0]
        rootdatadir = args.abidir[0]
        rootradardir = args.radardir[0]
        time_delta = args.deltat[0]
        date = args.DT[0]
        center_lon = float(args.center_lon[0])
        center_lat = float(args.center_lat[0])
        hs1km = args.hs[0]
        model_name = model_checkpoint_path.split('/')[5]
        rootpreddir = rootoutdir + model_name + '/' + date[:10]
        c = args.c[0]
        mkdir_p(rootpreddir)

    t_start = time()
    print('Time at the start of the program:', datetime.now())
    norm_vals = {'C02': {'mean': 20.659235, 'std': 14.25904},
                 'C05': {'mean': 16.05063, 'std': 6.841584},
                 'C13': {"mean": 272.76743, "std": 20.392328},
                 'C11': {"mean": 270.29974, "std": 19.439615},
                 'C14': {'mean': 271.30692, 'std': 20.340834},
                 'C15': {'mean': 268.1801, 'std': 19.45848}
                 }
    vars = {'goes16_half_km': ['C02'], 'goes16_one_km': ['C05'], 'goes16_two_km': ['C13', 'C15'],
            'target': {'MRMS_one_km': ['reflectivity_60min_neg10C_max']}}  # in format group:[variable(s)]

    inputs = []
    for k in vars.keys():
        if k != 'target':
            inputs.append(vars[k])

    print('date', date)
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
    radar_file_list = get_radar_list(dts, rootradardir)
    # area_def, size_x, size_y, center_x_ind_1km, center_y_ind_1km = get_rep_files(abifile_lists, center_lon, center_lat)
    for k in range(len(abifile_lists)):
        abi_files = abifile_lists[k]
        dt = dts[k]
        timestamp = str(dt)
        print('Current timestamp:', timestamp)
        t_input_start = time()
        scn, bad_data = get_satpy_scene(abi_files)
        print(scn.available_composite_names())
        sys.exit()
        center_x_idx, center_y_idx = scn['C05'].attrs['area'].get_xy_from_lonlat(center_lon,
                                                                                 center_lat)  # get center lat/lon
        inputs = format_inputs(scn, norm_vals, vars, center_x_idx, center_y_idx, hs1km)
        inputs = [inputs]
        print('Time to format inputs:', time() - t_input_start)
        radar_file = radar_file_list[k]
        center_lat_nm = str(int(center_lat * 100))
        center_lon_nm = str(int(center_lon * 100))
        time_nm = timestamp[11:13] + '_' + timestamp[14:16]
        if args.rgb:
            IRoutdir = dt.strftime(os.path.join(rootoutdir, 'RGBimages/%Y/%Y_%m_%d_%j/IR'))
            VISoutdir = dt.strftime(os.path.join(rootoutdir, 'RGBimages/%Y/%Y_%m_%d_%j/VIS'))
            mkdir_p(IRoutdir)
            mkdir_p(VISoutdir)
            IRfilepath = os.path.join(IRoutdir, dt.strftime('GOES16_IR_%Y_%m_%d_%H_%M_%j.png'))
            VISfilepath = os.path.join(VISoutdir, dt.strftime('GOES16_VIS_%Y_%m_%d_%H_%M_%j.png'))
            plot_rgb(scn, VISfilepath, IRfilepath, center_x_idx, center_y_idx, hs1km, timestamp)
        # figure_path = rootpreddir + '/preds_' + time_nm + '_' + center_lat_nm + '_' + center_lon_nm + '_' + str(
        #     hs1km) + '.png'
        # make_figure(scn, preds, center_x_idx, center_y_idx, hs1km, radar_file, c, timestamp, figure_path, full_US)
    print('Predictions complete')