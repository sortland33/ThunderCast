#Author: Stephanie M. Ortland
import os, sys
from glob import glob
import torch
import psutil
import pathlib
from netCDF4 import Dataset
import torch.nn as nn
from datetime import datetime,timedelta
from time import time
from torch.utils.data import DataLoader
import errno
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import kornia.augmentation as K
from torchmetrics import Metric
from PL_Metrics import CriticalSuccessIndex as CSI
from PL_Metrics import POD, FAR, SR, Accuracy, Specificity, F1Score, Recall, Precision, NoRes, CEFs
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

#plot glm in paneled figure
# python plot_glm.py -DT 2021-08-25-19-31 -t 5 -lat 41.41 -lon -90.85 -hs 55
os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise

def get_abifile_lists(rootdatadir, dts): #git list of abi files
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

def get_abifile_dict(rootdatadir, dts): #get dictionary of abi files
    dict = {}
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
        dict[dts[i]] = filenames
    return dict

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

def get_radar_list(dts, rootradardir): #get list of radar files
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

def get_radar_dict(dts, rootradardir): #get dictionary of radar files
    radar_file_dict = {}
    for dt in dts:
        radardir = dt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
        ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
        rad_end = dt + timedelta(minutes=5)
        radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(radardir, '*.netcdf'))}
        five_min_radfilelist = [radfiles[x] for x in radfiles if dt <= x <= rad_end]
        radar_file_dict[dt] = five_min_radfilelist[0]
    return radar_file_dict

def get_radar(radar_file, sat_grid): #get radar from radar file
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

def get_eni_dict(rootenidir, dts): #get ENI file dictionary
    eni_file_dict = {}
    for dt in dts:
        enidir = dt.strftime(os.path.join(rootenidir, '%Y/%Y%m%d/ENI_LightningDensity/002_min/'))
        ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
        end = dt + timedelta(minutes=5)
        enifiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(enidir, '*.netcdf'))}
        five_min_enifilelist = [enifiles[x] for x in enifiles if dt <= x <= end]
        eni_file_dict[dt] = five_min_enifilelist[0]
    return eni_file_dict

def get_eni(eni_file, sat_grid): #get ENI data from ENI file
    ds = Dataset(eni_file, 'r')
    nlat = len(ds.dimensions['Lat'])
    nlon = len(ds.dimensions['Lon'])
    NW_lat = getattr(ds, 'Latitude')
    NW_lon = getattr(ds, 'Longitude')
    dy = getattr(ds, 'LonGridSpacing')
    dx = getattr(ds, 'LatGridSpacing')
    eni = ds.variables['ENI_LightningDensity'][:, :]
    ds.close()

    print('remapping eni')
    ###Remap radar data to same domain as satellite data
    eni_lons = np.zeros((nlat, nlon))
    eni_lats = np.zeros((nlat, nlon))
    for ii in range(nlat):
        eni_lats[ii, :] = NW_lat - ii * dy
    for ii in range(nlon):
        eni_lons[:, ii] = NW_lon + ii * dx
    eni_grid = pr.geometry.GridDefinition(lons=eni_lons, lats=eni_lats)

    remapped_eni = pr.kd_tree.resample_nearest(eni_grid, eni, sat_grid, radius_of_influence=10000, fill_value=-999)
    print('remap complete')
    return remapped_eni

def get_glm_dict(rootglmdir, dts): #get dictionary of glm files
    glm_file_dict = {}
    for dt in dts:
        glmdir = dt.strftime(os.path.join(rootglmdir, '%Y/%Y%m%d/GLM/goes_east/agg/'))
        ###Grab glm files within 5 minutes of the ABI file and choose the first one to resample and plot.
        end = dt + timedelta(minutes=2)
        start = dt - timedelta(minutes=2)
        glmfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(glmdir, '*.netcdf'))}
        five_min_glmfilelist = [glmfiles[x] for x in glmfiles if start <= x <= end]
        glm_file_dict[dt] = five_min_glmfilelist[0]
    return glm_file_dict

def get_fed(glmfile,
            glmvar='flash_extent_density',
            glm_area_def=None,
            target_area_def=None,
            remap_cache=None,
            radius_of_influence=4000,
            fill_value=-1): #get flash extent density from glm file

    glmdata = Dataset(glmfile, 'r')
    fed = glmdata[glmvar][:]

    remapped_fed = pr.kd_tree.get_sample_from_neighbour_info('nn', remap_cache['target_def'].shape, fed,
                                                             remap_cache['input_index'],
                                                             remap_cache['output_index'],
                                                             remap_cache['index_array'])
    print(np.min(remapped_fed), np.max(remapped_fed))
    return remapped_fed

def get_remap_cache(input_def, target_def, radius_of_influence=4000):
  input_index, output_index, index_array, distance_array = \
               pr.kd_tree.get_neighbour_info(input_def, target_def, radius_of_influence, neighbours=1)
  remap_cache = {'input_index':input_index, 'output_index':output_index,
                 'index_array':index_array, 'target_def':target_def}
  return remap_cache


def get_area_definition(projection_file, outny=-1, outnx=-1, return_latlons=False, return_xy=False):
    from pyproj import Proj
    import pyresample as pr
    import numpy.ma as ma

    nc = Dataset(projection_file)
    gip = nc.variables['goes_imager_projection']
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]

    x *= gip.perspective_point_height
    y *= gip.perspective_point_height
    ny, nx = len(y), len(x)  # np.shape(xx)
    extents = (x.min(), y.min(), x.max(), y.max())
    if (outny > 0):
        # Increase or decrease spatial resolution. Note that outny and outnx should be
        # a whole-number multiple of ny and nx.
        # assert(ny % outny == 0 and nx % outnx == 0)
        ny = outny;
        nx = outnx

    newgrid = pr.geometry.AreaDefinition('geos', 'goes_conus', 'goes', \
                                         {'proj': 'geos', 'h': gip.perspective_point_height, \
                                          'lon_0': gip.longitude_of_projection_origin, \
                                          'a': gip.semi_major_axis, 'b': gip.semi_minor_axis,
                                          'sweep': gip.sweep_angle_axis}, nx, ny, extents)
    nc.close()

    if (return_latlons):
        return newgrid, ny, nx, newlons, newlats
    if (return_xy):
        return newgrid, ny, nx, x, y
    else:
        return newgrid, ny, nx

def make_subplot_glm(scn, center_x_idx, center_y_idx, hs, glmfile, c, timestamp, i, j, fig, axes, geoproj):
    state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')
    CVD_cmap_all = matplotlib.colormaps.get_cmap('pyart_HomeyerRainbow')
    colors1 = CVD_cmap_all(np.linspace(0, 0.25, 98))
    colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 158))
    cmap_colors = np.vstack((colors1, colors2))
    CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)

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

    # Get GLM
    glm_area_def, _, _ = get_area_definition('/home/sortland/ci_ltg/proj_files/GOES_East.nc', return_latlons=False)
    remap_cache = get_remap_cache(glm_area_def, sat_grid)
    glmgrid = get_fed(glmfile, glmvar='flash_extent_density', remap_cache=remap_cache)
    print('glm max', np.nanmax(glmgrid))

    #make figure
    xmin = np.min(x1km);
    xmax = np.max(x1km);
    ymin = np.min(y1km);
    ymax = np.max(y1km)
    axes[i, j].set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    axes[i, j].add_feature(cfeature.BORDERS)
    axes[i, j].add_feature(state_borders, edgecolor='plum')
    axes[i, j].set_aspect('equal')

    if c == 'C05':
        img = axes[i, j].pcolormesh(xx1km, yy1km, C05vals, vmin=0, vmax=100, cmap=matplotlib.colormaps['Greys_r'])
        glmimg = axes[i, j].pcolormesh(xx1km, yy1km, ma.masked_less(glmgrid, 0.1), vmin=1.0, vmax=90,
                                           cmap=matplotlib.colormaps['viridis'], linewidth=0, antialiased=True)  # , alpha=0.9)
    elif c == 'C13':
        img = axes[i, j].pcolormesh(xx1km, yy1km, zoom(C13vals, 2, order=0), vmin=200, vmax=300,
                                    cmap=matplotlib.colormaps['Greys'])
        glmimg = axes[i, j].pcolormesh(xx1km, yy1km, ma.masked_less(glmgrid, 0.1), vmin=1.0, vmax=90,
                                           cmap=matplotlib.colormaps['viridis'], linewidth=0, antialiased=True)  # , alpha=0.9)
    else:
        img = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, C02vals, vmin=0, vmax=100, cmap=matplotlib.colormaps['Greys_r'])
        glmimg = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, ma.masked_less(zoom(glmgrid, 2, order=0), 0.1), vmin=1.0, vmax=20,
                                           cmap=matplotlib.colormaps['viridis'], linewidth=0, antialiased=True , alpha=0.7)
        # ax[i, j].imshow(glmgrid, transform=crs, extent=crs.bounds, cmap==matplotlib.colormaps['plasma'], vmin=0.1, vmax=50)
    axes[i, j].set_title((' ').join([timestamp, 'UTC']), fontsize=16)
    return img, glmimg

def make_subplot_eni(scn, center_x_idx, center_y_idx, hs, enifile, c, timestamp, i, j, fig, axes, geoproj):
    state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')

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

    # Get ENI
    eni_data = get_eni(enifile, sat_grid)
    print(np.min(eni_data), np.max(eni_data))

    #make figure
    xmin = np.min(x1km);
    xmax = np.max(x1km);
    ymin = np.min(y1km);
    ymax = np.max(y1km)
    axes[i, j].set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    axes[i, j].add_feature(cfeature.BORDERS)
    axes[i, j].add_feature(state_borders, edgecolor='plum')
    axes[i, j].set_aspect('equal')

    if c == 'C05':
        img = axes[i, j].pcolormesh(xx1km, yy1km, C05vals, vmin=0, vmax=100, cmap=matplotlib.colormaps['Greys_r'])
        eniimg = axes[i, j].pcolormesh(xx1km, yy1km, ma.masked_less(eni_data, 0.1), vmin=0, vmax=2, cmap=matplotlib.colormaps['viridis'], linewidth=0, antialiased=True)  # , alpha=0.9)
    elif c == 'C13':
        img = axes[i, j].pcolormesh(xx1km, yy1km, zoom(C13vals, 2, order=0), vmin=200, vmax=300,
                                    cmap=matplotlib.colormaps['Greys'])
        eniimg = axes[i, j].pcolormesh(xx1km, yy1km, ma.masked_less(eni_data, 0.1), vmin=0, vmax=2, cmap=matplotlib.colormaps['viridis'], linewidth=0, antialiased=True)  # , alpha=0.9)
    else:
        img = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, C02vals, vmin=0, vmax=100, cmap=matplotlib.colormaps['Greys_r'])
        eniimg = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, ma.masked_less(zoom(eni_data, 2, order=0), 0.1), vmin=0, vmax=2, cmap=matplotlib.colormaps['viridis'], linewidth=0, antialiased=True , alpha=0.7)
        # ax[i, j].imshow(glmgrid, transform=crs, extent=crs.bounds, cmap==matplotlib.colormaps['plasma'], vmin=0.1, vmax=50)
    axes[i, j].set_title((' ').join([timestamp, 'UTC']), fontsize=16)
    return img, eniimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes a datetime and gets prediction images.')
    parser.add_argument('-a', '--abidir',
                        help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/',
                        default=['/arcdata/goes/grb/goes16'], type=str, nargs=1)
    parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/predictions/'])
    parser.add_argument('-DTstart', '--DT', help='A datetime in %Y-%m-%d-%H-%M',
                        type=str, nargs=1)
    parser.add_argument('-t', '--deltat', help='Time between subplots.',
                        type=int, nargs=1)
    parser.add_argument('-np', '--numplots', help='Number of subplots subplots. (4 or 6)',
                        type=int, nargs=1, default=[6])
    parser.add_argument('-lat', '--center_lat', help='Center latitude for prediction',
                        type=str, nargs=1)
    parser.add_argument('-lon', '--center_lon', help='Center longitude for prediction',
                        type=str, nargs=1)
    parser.add_argument('-hs', '--hs', help='Half size for prediction area indices',
                        type=int, nargs=1, default=[127])
    parser.add_argument('-c', '--c', help='Channel to plot',
                        type=str, nargs=1, default=['C02'])

    args = parser.parse_args()

    rootoutdir = args.outdir[0]
    rootdatadir = args.abidir[0]
    rootglmdir = '/ships19/grain/probsevere/lightning'
    #rootglmdir = '/apollo/grain/saved_probsevere_data/lightning'
    rootenidir = rootglmdir
    time_delta = args.deltat[0]
    date = args.DT[0]
    center_lon = float(args.center_lon[0])
    center_lat = float(args.center_lat[0])
    hs1km = args.hs[0]
    numplots = args.numplots[0]
    if numplots % 2 != 0:
        if numplots % 3 == 0:
            numcols = int(numplots/3)
            numrows = int(numplots/numcols)
        print('Number of subplots should be 4 or 6')
    else:
        numcols = int(numplots/2)
        numrows = int(numplots/numcols)
    print(numcols, numrows)
    c = args.c[0]
    rootsavedir = '/ships19/grain/convective_init/random_imgs/'
    mkdir_p(rootsavedir)

    t_start = time()

    topdt = datetime.strptime(date, '%Y-%m-%d-%H-%M')
    dts = []
    for i in range(numplots):
        min = i*time_delta
        dt = topdt + timedelta(minutes=min)
        dts.append(dt)
    print(np.array(dts))
    hrs = [x.hour for x in dts]
    hr_min = {}
    for hr in hrs:
        mins = []
        for x in dts:
            if x.hour == hr:
                mins.append(x.minute)
        hr_min[str(hr)] = mins
    satdir = topdt.strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
    netcdfpattern = satdir + '*C02*'
    satfiles = {datetime.strptime(os.path.basename(x).split('_')[3][1:-1], '%Y%j%H%M%S'): x for x in
                glob(netcdfpattern)}
    sat_netcdfs = []
    new_dts = []
    for x in satfiles:
        if str(x.hour) in hr_min.keys():
            mins = hr_min[str(x.hour)]
            if x.minute in mins:
                sat_netcdfs.append(satfiles[x])
                if x not in new_dts:
                    new_dts.append(x)

    abifile_dict = get_abifile_dict(rootdatadir, new_dts)
    glmfile_dict = get_glm_dict(rootglmdir, new_dts)
    enifile_dict = get_eni_dict(rootenidir, new_dts)
    dts = np.array(np.sort(new_dts)).reshape((numrows, numcols))
    goesnc = Dataset('/home/sortland/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)

    if numplots == 9:
        fig, axes = plt.subplots(numrows, numcols, figsize=(15, 15), subplot_kw={'projection': geoproj})
    else:
        fig, axes = plt.subplots(numrows, numcols, figsize=(15, 10), subplot_kw={'projection': geoproj})

    print(numcols, numrows)
    for i in range(numrows):
        for j in range(numcols):
            print('i, j:', i, j)
            dt = dts[i,j]
            timestamp = str(dt)
            abi_files = abifile_dict[dt]
            glmfile = glmfile_dict[dt]
            enifile = enifile_dict[dt]
            scn, bad_data = get_satpy_scene(abi_files)
            center_x_idx, center_y_idx = scn['C05'].attrs['area'].get_array_indices_from_lonlat(center_lon, center_lat)  # get center lat/lon
            img, lightningimg = make_subplot_glm(scn, center_x_idx, center_y_idx, hs1km, glmfile, c, timestamp, i, j, fig, axes, geoproj)
            #img, lightningimg = make_subplot_eni(scn, center_x_idx, center_y_idx, hs1km, enifile, c, timestamp, i, j, fig, axes, geoproj)


    cbaxes = fig.add_axes([0.125, .06, 0.23, 0.03])  # left bottom width height
    cbar = plt.colorbar(img, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0,
                        ticks=np.arange(0, 100, 10), cax=cbaxes)
    cbar.ax.tick_params(labelsize=12)
    if c == 'C05':
        cbar.set_label('ABI 0.64-$\mu$m reflectance [%]', fontsize=14)
    elif c == 'C13':
        cbar.set_label('ABI 10.3-$\mu$m BT [%]', fontsize=14)
    else:
        cbar.set_label('ABI 0.64-$\mu$m reflectance [%]', fontsize=14)
    cbaxes2 = fig.add_axes([0.375, .06, 0.23, 0.03])  # left bottom width height
    # cbar2 = plt.colorbar(lightningimg, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
    #                      ticks=np.arange(0, 2, 0.5), cax=cbaxes2)
    cbar2 = plt.colorbar(lightningimg, orientation='horizontal', pad=0.01, extend='max', drawedges=0,
                         ticks=np.arange(5, 21, 5), cax=cbaxes2)
    #cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label(r'GLM Flash Extent Density [$\frac{flashes}{5\ mins}$]', fontsize=14)
    #cbar2.set_label(r'ENI Lightning Density [$\frac{1}{min*km^2}$]', fontsize=14)
    cbar2.set_alpha(1.0)
    cbar2.draw_all()
    time_nm = timestamp[11:13] + '_' + timestamp[14:16]
    center_lat_nm = str(int(center_lat * 100))
    center_lon_nm = str(int(center_lon * 100))
    figure_path = rootsavedir + 'glm_' + time_nm + '_' + center_lat_nm + '_' + center_lon_nm + '_' + str(hs1km) + '.png'
    #figure_path = rootsavedir + 'eni_' + time_nm + '_' + center_lat_nm + '_' + center_lon_nm + '_' + str(hs1km) + '.png'
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)
    print(figure_path)
    print('Predictions complete')