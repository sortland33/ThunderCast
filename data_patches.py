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

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['SATPY_CONFIG_PATH'] = '/home/sortland/ci_ltg/PPP_config/'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'

sys.path.insert(0, '/home/sortland/ci_ltg')
PWD = os.environ['PWD'] + '/'

# Run this file example: python data_patches.py -DTfile LATLONDT_patch.config
# python data_patches.py -DTfile LATLONDT.config -repfile

# Define functions
def mkdir_p(path): #create a new pathway
    try:
        os.makedirs(path)
    except (OSError, IOError) as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def current_memory(): #get current memory usage
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    return mem

def get_dates(DTfilelist):  # Gets info from DT configuration list with dates of interest
    file1 = open(DTfilelist, 'r')
    is_latlon = None
    dates = []
    for line in file1:
        if is_latlon is None:
            is_latlon = line[0] == 'L'
            continue
        if line[0] == '#': continue
        if is_latlon:
            parts = line.split()
            dates.append(parts[0])
    return dates

def get_datetimes(rootdatadir, dates): #get datetimes for dates of interest
    dts = []
    for n in range(len(dates)):
        # dts_per_date = []
        topdt = datetime.strptime(dates[n], '%Y-%m-%d')
        enddt = topdt + timedelta(hours=25)
        satdir = topdt.strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        netcdfpattern_C02 = topdt.strftime(satdir + 'OR_ABI-L1b-RadC-M*C02_G16_s%Y*')
        sat_netcdfs = np.sort(glob(netcdfpattern_C02))
        for i in range(len(sat_netcdfs)):
            # First get information for the datetimes
            sat_netcdf = sat_netcdfs[i]
            basefile = os.path.basename(sat_netcdf)
            parts = basefile.split('_')
            separator = '_'
            year_day_time = parts[3][1:-1]
            dt_start = datetime.strptime(year_day_time, '%Y%j%H%M%S')
            dts.append(dt_start)
            # dts_per_date.append(dt_start)
    # get every other datetime instead of every one. Alter time step you take
    rand_int_gen = random.randint(0, 2)
    dts = dts[rand_int_gen::3]
    return dts

def get_abifile_lists(rootdatadir, dts): #get all abifiles (channels) for each datatime
    list = []
    abi_channels = ['02', '05', '11', '13', '14', '15']  # for this application we do not need all the channels
    for i in range(len(dts)):
        satdir = dts[i].strftime(rootdatadir + '%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        filenames = []
        for ch in abi_channels:
            netcdfpattern = dts[i].strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
            try:
                filenames.append(glob(netcdfpattern)[0])
            except IndexError:
                continue
        list.append(filenames)
    return list

def get_rep_scene(file_list): #get a representative satpy scene
    files = file_list[0]
    rep_scene = Scene(reader='abi_l1b', filenames=files)
    rep_scene.load(['C13', 'C15', 'C11', 'C14', 'C02', 'C05'])
    return (rep_scene)

def get_center_inds(size_x, size_y, hs, overlap): #get center indices of each patch
    step = hs * 2 - overlap
    centers = []
    y_end = size_y - ((size_y - hs) % step)
    x_end = size_x - ((size_x - hs) % step)
    y_centers = np.arange(hs, y_end, step)
    x_centers = np.arange(hs, x_end, step)
    xx_centers, yy_centers = np.meshgrid(x_centers, y_centers)  # xx and yy centers should be the same shape
    for i in range(xx_centers.shape[0]):
        for j in range(xx_centers.shape[1]):
            center = (xx_centers[i, j], yy_centers[i, j])
            centers.append(center)
    return centers

def get_info_for_patches(centers, rep_scn, hs):  #get all the information for the patches. This should stay the same between ABI files.
    center_x_ind = centers[0]
    center_y_ind = centers[1]
    ymin_index = center_y_ind + hs
    ymax_index = center_y_ind - hs
    xmin_index = center_x_ind - hs
    xmax_index = center_x_ind + hs

    area_def = rep_scn['C05'].attrs['area']
    x_coords, y_coords = area_def.get_proj_vectors()

    center_y = y_coords[center_y_ind]
    center_x = x_coords[center_x_ind]
    xmin = x_coords[xmin_index]
    xmax = x_coords[xmax_index]
    ymin = y_coords[ymin_index]
    ymax = y_coords[ymax_index]

    center_lon, center_lat = area_def.get_lonlat_from_projection_coordinates(center_x, center_y)
    if center_lon == np.inf or center_lat == np.inf:
        return

    cropped = rep_scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
    crop_area = cropped['C05'].attrs['area']

    # Determine which state the center lon and lat are in for saving
    shpfilename = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces')
    reader = shpreader.Reader(shpfilename)
    states = reader.records()
    point = sgeom.Point(center_lon, center_lat)
    center_state = None
    for state in states:
        attribute = 'name'
        name = state.attributes[attribute]
        geom = state.geometry
        if geom.contains(point):
            center_state = state.attributes[attribute].rstrip('\x00')

    conus_lat_min = 33
    conus_lat_max = 41
    conus_lon_min = -117
    conus_lon_max = -82

    # Determine patch output directory from center_state and U.S. Climate regions
    northwest = ['Washington', 'Oregon', 'Idaho']
    west = ['California', 'Nevada']
    west_north_central = ['Montana', 'Nebraska', 'North Dakota', 'South Dakota',
                          'Wyoming']  # northern rockies and plains
    southwest = ['Arizona', 'Colorado', 'New Mexico', 'Utah']
    east_north_central = ['Minnesota', 'Michigan', 'Iowa', 'Wisconsin']  # upper midwest
    south = ['Arkansas', 'Kansas', 'Louisiana', 'Mississippi', 'Oklahoma', 'Texas']
    southeast = ['Alabama', 'Florida', 'Georgia', 'North Carolina', 'South Carolina', 'Virginia']
    central = ['Illinois', 'Indiana', 'Kentucky', 'Missouri', 'Ohio', 'Tennessee', 'West Virginia']  # Ohio Valley
    northeast = ['Connecticut', 'Delaware', 'Maine', 'Massachusetts', 'New Hampshire', 'New Jersey', 'New York',
                 'Pennsylvania', 'Rhode Island', 'Vermont', 'Maryland']

    if (center_state in northwest):
        region = 'northwest'
    elif (center_state in west):
        region = 'west'
    elif (center_state in west_north_central):
        region = 'west_north_central'
    elif (center_state in southwest):
        region = 'southwest'
    elif (center_state in east_north_central):
        region = 'east_north_central'
    elif (center_state in south):
        region = 'south'
    elif (center_state in southeast):
        region = 'southeast'
    elif (center_state in central):
        region = 'central'
    elif (center_state in northeast):
        region = 'northeast'
    else:
        if center_lon < conus_lon_min:
            region = 'oconus_west'
        elif center_lon > conus_lon_max:
            region = 'oconus_east'
        elif center_lat > conus_lat_max:
            region = 'oconus_north'
        elif center_lat < conus_lat_min:
            region = 'oconus_south'
        else:
            region = 'oconus_other'

    patch_info = [center_lon, center_lat, region, center_x_ind, center_y_ind, crop_area]
    return patch_info

def get_rep_radar_grid(dts, rootoutdir): #get a representative radar grid- since these values will stay consistent
    try:
        dt = dts[0]
        radar_file = dt.strftime(os.path.join(rootoutdir, 'MRMS_targets/%Y/%m/%d/%Y_%m_%d_%H_%M_%j.nc'))
        dataset_rad = Dataset(radar_file, 'r')
    except:
        mid_ind = int(len(dts)/2)
        dt = dts[mid_ind]
        radar_file = dt.strftime(os.path.join(rootoutdir, 'MRMS_targets/%Y/%m/%d/%Y_%m_%d_%H_%M_%j.nc'))
        dataset_rad = Dataset(radar_file, 'r')
    nlat = getattr(dataset_rad, 'nlat')
    nlon = getattr(dataset_rad, 'nlon')
    NW_lat = getattr(dataset_rad, 'NW_lat')
    NW_lon = getattr(dataset_rad, 'NW_lon')
    dy = getattr(dataset_rad, 'dy')
    dx = getattr(dataset_rad, 'dx')

    ###Remap radar data to same domain as satellite data
    radar_lons = np.zeros((nlat, nlon))
    radar_lats = np.zeros((nlat, nlon))
    for ii in range(nlat):
        radar_lats[ii, :] = NW_lat - ii * dy
    for ii in range(nlon):
        radar_lons[:, ii] = NW_lon + ii * dx
    radar_grid = pr.geometry.GridDefinition(lons=radar_lons, lats=radar_lats)
    dataset_rad.close()
    return radar_grid

def get_patch_percents(p_info, rootoutdir, single_dt, radar_grid): #get the data patches percent of the area with radar at -10C
    center_lon = p_info[0]
    center_lat = p_info[1]
    region = p_info[2]
    center_x_ind = p_info[3]
    center_y_ind = p_info[4]
    sat_grid = p_info[5]
    try:
        max_radar_file = single_dt.strftime(os.path.join(rootoutdir, 'MRMS_targets/%Y/%m/%d/%Y_%m_%d_%H_%M_%j.nc'))
        data = Dataset(max_radar_file, 'r+')
        max_radar_10C = data.groups['MRMS_one_km'].variables['reflectivity_60min_neg10C_max'][:,:]
        remapped_radar_10C = pr.kd_tree.resample_nearest(radar_grid, max_radar_10C, sat_grid, radius_of_influence=10000, fill_value=-999)
        data.close()
        # Calculate percent of radar = -999
        ind_no_cov = np.where(remapped_radar_10C == -999)
        fraction_no_cov = len(ind_no_cov[0]) / (remapped_radar_10C.shape[0] * remapped_radar_10C.shape[1])
        percent_no_cov = fraction_no_cov * 100
        del ind_no_cov, fraction_no_cov

        # Calculate percent of radar >= 30 dBZ
        ind_10C = np.where(remapped_radar_10C >= 30)
        fraction_10C = len(ind_10C[0]) / (remapped_radar_10C.shape[0] * remapped_radar_10C.shape[1])
        percent_10C = fraction_10C * 100
        del ind_10C, fraction_10C

    except:
        print('Error in calculation')
        percent_10C = 0
        percent_no_cov = 100
        max_radar_file = None
        max_radar_10C = None
        remapped_radar_10C = None

    p_info = [center_lon, center_lat, region, center_x_ind, center_y_ind, percent_10C, percent_no_cov]
    del center_lon, center_lat, region, center_x_ind, center_y_ind, sat_grid, max_radar_file, max_radar_10C, remapped_radar_10C
    del percent_no_cov, percent_10C
    return p_info


def save_rep_satpy_scene(abi_files, repfile):
    scn = Scene(reader='abi_l1b', filenames=abi_files)  # get satpy scene to be cropped to patches
    CHs = ['C13', 'C15', 'C11', 'C14', 'C02', 'C05']
    scn.load(CHs)  # Load the scene and check for data problems
    for c in CHs:
        scn[c] = scn[c].astype(np.float32)
        scn[c].encoding = {'zlib': True, 'dtype': np.float32}  # only save the data as floats instead of doubles

    scn.save_datasets(writer='cf', datasets=['C13', 'C15', 'C11', 'C14', 'C02', 'C05'],
                      filename=repfile, groups={'goes16_half_km': ['C02'], 'goes16_one_km': ['C05'],
                                                'goes16_two_km': ['C11', 'C13', 'C14', 'C15']},
                      include_lonlats=False)  # , exclude_attrs=['raw_metadata'])
    del scn
    return


def delete_no_cov_patches(patch_info_dicts, dates): #we don't want to save patches with too many no coverage pixels to avoid distorting the dataset.
    keys = list(patch_info_dicts.keys())
    for j in range(len(dates)):
        date = datetime.strptime(dates[j], '%Y-%m-%d')
        keys_in_date = []  # we need to get a list of all the keys in this particular date
        for k in keys:
            key_datetime = datetime.strptime(k, '%Y%m%d%H%M%S')
            if (key_datetime.year == date.year and key_datetime.month == date.month and key_datetime.day == date.day):
                keys_in_date.append(k)
        for k in keys_in_date:
            new_patches = []
            for i in range(len(patch_info_dicts[k])):
                pat = patch_info_dicts[k][i]
                if pat[-1] > 1.0:
                    continue
                else:
                    new_patches.append(patch_info_dicts[k][i])
            patch_info_dicts[k] = new_patches
    return patch_info_dicts


def delete_random_zero_radar_patches(patch_info_dicts, dates): #We don't need to save all the patches without radar at -10C- if we did it would take up a lot of memory
    keys = list(patch_info_dicts.keys())
    for j in range(len(dates)):
        date = datetime.strptime(dates[j], '%Y-%m-%d')
        keys_in_date = []  # we need to get a list of all the keys in this particular date
        for k in keys:
            key_datetime = datetime.strptime(k, '%Y%m%d%H%M%S')
            if (key_datetime.year == date.year and key_datetime.month == date.month and key_datetime.day == date.day):
                keys_in_date.append(k)

        zero_patches = []
        non_zero_patches = []
        for k in keys_in_date:  # get all the zero radar patch cases- collect [dictionary key, index, center_region]
            for i in range(len(patch_info_dicts[k])):
                pat = patch_info_dicts[k][i]
                cr = pat[2]
                pat_info = [k, i, cr]
                if pat[-2] < 1.0:
                    zero_patches.append(pat_info)
                else:
                    non_zero_patches.append(pat_info)

        all_regions = ['northwest', 'west', 'west_north_central', 'southwest', 'east_north_central', 'south',
                       'southeast',
                       'central', 'northeast', 'oconus_west', 'oconus_east', 'oconus_north', 'oconus_south',
                       'oconus_other']

        desired_patches = []
        for k in range(len(all_regions)):
            center_region = all_regions[k]
            neg_cases = [zero_patches[p] for p in range(len(zero_patches)) if zero_patches[p][-1] == center_region]
            pos_cases = [non_zero_patches[p] for p in range(len(non_zero_patches)) if
                         non_zero_patches[p][-1] == center_region]
            if len(neg_cases) > len(pos_cases) and len(
                    pos_cases) != 0:  # only keep random zero cases equal to number of positives
                neg_cases = random.sample(neg_cases, len(pos_cases))
            elif len(pos_cases) == 0:
                neg_cases = []
            else:
                neg_cases = neg_cases
                pos_cases = pos_cases
            desired_patches = desired_patches + pos_cases + neg_cases

        for k in keys_in_date:
            new_patches = []
            sub_desired_patches = [desired_patches[p] for p in range(len(desired_patches)) if
                                   desired_patches[p][0] == k]
            # select only the patches for each date
            for n in range(len(sub_desired_patches)):
                ind = sub_desired_patches[n][1]
                pat = patch_info_dicts[k][ind]
                new_patches.append(pat)
            patch_info_dicts[k] = new_patches

    return patch_info_dicts


def get_satpy_scene(abi_files, dt):  # get satpy scene to be cropped to patches
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


def solar_angles(dt,lon,lat): #get solar angles
  '''
  ---------------------------------------------------------------------
   Compute solar angles

   input:
         dt   = python datetime object
         ###jday = julian day
         ###tu   = time of day - fractional hours
         lon  = latitude in degrees
         lat  = latitude in degrees

    output:
         asol = solar zenith angle in degrees
         phis = solar azimuth angle in degrees

  ----------------------------------------------------------------------
  '''

  if(not(isinstance(dt,list)) and not(isinstance(dt,np.ndarray))): dt = [dt]
  lon = np.array(lon)
  lat = np.array(lat)

  hour = np.array([_.hour for _ in dt])
  minute = np.array([_.minute for _ in dt])
  jday = np.array([int(_.strftime('%j')) for _ in dt])

  dtor = np.pi/180.

  tu = hour + minute/60.0

  #      mean solar time
  tsm = tu + lon/15.0
  xlo = lon*dtor
  xla = lat*dtor
  xj = np.copy(jday)

  #      time equation (mn.dec)
  a1 = (1.00554*xj - 6.28306)  * dtor
  a2 = (1.93946*xj + 23.35089) * dtor
  et = -7.67825*np.sin(a1) - 10.09176*np.sin(a2)

  #      true solar time
  tsv = tsm + et/60.0
  tsv = tsv - 12.0

  #      hour angle
  ah = tsv*15.0*dtor

  #      solar declination (in radian)
  a3 = (0.9683*xj - 78.00878) * dtor
  delta = 23.4856*np.sin(a3)*dtor

  #     elevation, azimuth
  cos_delta = np.cos(delta)
  sin_delta = np.sin(delta)
  cos_ah = np.cos(ah)
  sin_xla = np.sin(xla)
  cos_xla = np.cos(xla)

  amuzero = sin_xla*sin_delta + cos_xla*cos_delta*cos_ah
  elev = np.arcsin(amuzero)
  cos_elev = np.cos(elev)
  az = cos_delta*np.sin(ah)/cos_elev
  caz = (-cos_xla*sin_delta + sin_xla*cos_delta*cos_ah) / cos_elev
  azim=0.*az
  index=np.where(az >= 1.0)
  if np.size(index) > 0 : azim[index] = np.arcsin(1.0)
  index=np.where(az <= -1.0)
  if np.size(index) > 0 : azim[index] = np.arcsin(-1.0)
  index=np.where((az > -1.0) & (az < 1.0))
  if np.size(index) > 0 : azim[index] = np.arcsin(az[index])

  index=np.where(caz <= 0.0)
  if np.size(index) > 0 : azim[index] = np.pi - azim[index]

  index=np.where((caz > 0.0) & (az <= 0.0))
  if np.size(index) > 0 : azim[index] = 2 * np.pi + azim[index]
  azim = azim + np.pi
  pi2 = 2 * np.pi
  index=np.where(azim > pi2)
  if np.size(index) > 0 : azim[index] = azim[index] - pi2

  #     conversion in degrees
  elev = elev / dtor
  asol = 90.0 - elev
  phis = azim / dtor
  sol_zenith=asol
  sol_azimuth=phis

  return sol_zenith, sol_azimuth

def get_patch(patch_info, scn, dt, rootoutdir, hs1km, max_radar_10C, radar_grid): #get 320x320 km data patches
    center_lon = patch_info[0]
    center_lat = patch_info[1]
    region = patch_info[2]
    center_x_ind = int(patch_info[3])
    center_y_ind = int(patch_info[4])
    percent_10C = patch_info[5]
    percent_no_cov = patch_info[6]

    NCoutdir = dt.strftime(os.path.join(rootoutdir, ('data/%Y/%m/%d/' + region)))
    if percent_10C >= 1.0:
        NCoutdir = NCoutdir + '/positive_cases'
    else:
        NCoutdir = NCoutdir + '/negative_cases'
    mkdir_p(NCoutdir)
    center_lon_nm = str(round(center_lon * 100))
    center_lat_nm = str(round(center_lat * 100))
    percent_10C_nm = str(round(percent_10C * 100))
    NCfilepath = os.path.join(NCoutdir,
                              (dt.strftime('%Y_%m_%d_%H_%M_%j_') + center_lon_nm + '_' + center_lat_nm + '_' + percent_10C_nm + '.nc'))

    CHs = ['C13', 'C15', 'C11', 'C14', 'C02', 'C05']
    nans_present = False
    ymin_index = int(center_y_ind + hs1km)
    ymax_index = int(center_y_ind - hs1km)
    xmin_index = int(center_x_ind - hs1km)
    xmax_index = int(center_x_ind + hs1km)
    x_coords, y_coords = scn['C05'].attrs['area'].get_proj_vectors()
    xmin = x_coords[xmin_index]
    xmax = x_coords[xmax_index]
    ymin = y_coords[ymin_index]
    ymax = y_coords[ymax_index]
    crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])

    try:
        for c in CHs:
            CHvals = crop_scn[c].values
            if (np.isnan(np.min(CHvals)) or np.isnan(np.max(CHvals))):
                print('Skipping datetime for bad data.')
                if os.path.exists(NCfilepath):
                    print('removing file because path exists.')
                    os.remove(NCfilepath)
                NCfilepath = None
                return NCfilepath
            del CHvals
    except KeyError:
        if os.path.exists(NCfilepath):
            print('removing file because path exists.')
            os.remove(NCfilepath)
        NCfilepath = None
        return NCfilepath

    for c in CHs:
        crop_scn[c] = crop_scn[c].astype(np.float32)
        crop_scn[c].encoding = {'dtype': np.float32}

    #before saving check for nans or fill values in the radar
    sat_grid = crop_scn['C05'].attrs['area']
    remapped_radar_10C = pr.kd_tree.resample_nearest(radar_grid, max_radar_10C, sat_grid, radius_of_influence=10000,
                                                     fill_value=-999)
    lons, lats = crop_scn['C05'].attrs['area'].get_lonlats()
    sol_zen, sol_azi = solar_angles(dt, lons, lats)
    rad_nan_test = np.isnan(remapped_radar_10C)
    zen_nan_test = np.isnan(sol_zen)
    azi_nan_test = np.isnan(sol_azi)
    if np.any(rad_nan_test) or np.any(zen_nan_test) or np.any(azi_nan_test):
        print('Skipping datetime for bad data.')
        if os.path.exists(NCfilepath):
            print('removing file because path exists.')
            os.remove(NCfilepath)
        NCfilepath = None
        return NCfilepath
    if np.any(np.isin(remapped_radar_10C, [9.96921e+36])) or np.any(np.isin(sol_zen, [9.96921e+36])) or np.any(np.isin(sol_azi, [9.96921e+36])):
        print('Skipping datetime for bad data.')
        if os.path.exists(NCfilepath):
            print('removing file because path exists.')
            os.remove(NCfilepath)
        NCfilepath = None
        return NCfilepath

    crop_scn.save_datasets(writer='cf', datasets=['C13', 'C15', 'C11', 'C14', 'C02', 'C05'],
                           filename=NCfilepath, groups={'goes16_half_km': ['C02'], 'goes16_one_km': ['C05'],
                                                        'goes16_two_km': ['C11', 'C13', 'C14', 'C15']}, include_lonlats=False)

    root = Dataset(NCfilepath, 'r+')
    #Add radar
    root.description = 'Data Patch for Convective Initiation Model (CONUS)'
    root.center_latitude = center_lat
    root.center_longitude = center_lon
    root.center_region = region
    root.percent_60max_neg10C_greater30dBZ = percent_10C
    root.percent_60max_neg10C_noCoverage = percent_no_cov

    radargroup = root.createGroup('MRMS_one_km')
    radargroup.createDimension('y', remapped_radar_10C.shape[0])
    radargroup.createDimension('x', remapped_radar_10C.shape[1])
    reflectivity_max = radargroup.createVariable('reflectivity_60min_neg10C_max', 'float32',
                                                 ('y', 'x'))
    reflectivity_max[:, :] = remapped_radar_10C
    #Add solar angles
    group_1km = root.groups['goes16_one_km']
    solar_zenith = group_1km.createVariable('solar_zenith', 'float32', ('y', 'x'))
    solar_zenith[:,:] = sol_zen
    solar_azimuth = group_1km.createVariable('solar_azimuth', 'float32', ('y', 'x'))
    solar_azimuth[:,:] = sol_azi
    del crop_scn
    root.close()
    return NCfilepath


# Main program
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Takes a list of datetimes and creates netcdfs of data patches for CI model.')
    parser.add_argument('-a', '--abidir',
                        help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/',
                        default=['/arcdata/goes/grb/goes16/'], type=str, nargs=1)
    parser.add_argument('-r', '--radardir', help='Root directory with radar netcdfs.',
                        default=['/apollo/grain/saved_ci_ltg_data/radar'],
                        nargs=1, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/'])
    parser.add_argument('-DTfile', '--DTfilelist', help='A list of lats/lons/datetimes.',
                        type=str, nargs=1)
    # parser.add_argument('-RepDir', '--repdir', help='Representative file output directory.',type=str, nargs=1, default=['/ships19/grain/convective_init'])
    # parser.add_argument('-repfile', '--add_rep_file', help='If you want to save a representative satpy file (if you do not have one already).',
    #                     action='store_true')
    args = parser.parse_args()

    print('Versions for Dask:')
    print(dask.__version__)
    # print(dask.distributed.__version__)

    t_start = time.time()
    ###information from inputs
    rootoutdir = args.outdir[0]
    rootdatadir = args.abidir[0]
    rootradardir = args.radardir[0]
    # rootrepdir = args.repdir[0]
    # utils.mkdir_p(rootoutdir)  # make sure output folder exists
    mkdir_p(rootoutdir)  # make sure output folder exists

    print('Getting dates and file lists.')
    dates = get_dates(args.DTfilelist[0])  # get dates from input file
    dts = get_datetimes(rootdatadir, dates)  # list of datetimes
    abifile_lists = get_abifile_lists(rootdatadir, dts)  # list of list of files per datetime

    client = Client(n_workers=5, threads_per_worker=1, memory_limit='15GB') #Call Dask Client
    # client = Client(n_workers=8, threads_per_worker=1, memory_limit='10GB') #Call Dask Client
    #client = Client("tcp://128.104.109.126:8786")

    rep_scene = get_rep_scene(abifile_lists)  # Representative scene for 1km area def and sizes
    radar_grid = get_rep_radar_grid(dts, rootoutdir)
    hs1km = 159
    overlap1km = 32
    size_x = rep_scene['C05'].sizes['x']
    size_y = rep_scene['C05'].sizes['y']
    print('Getting center inds')
    centers = get_center_inds(size_x, size_y, hs1km, overlap1km)
    print('num total centers', len(centers))
    #centers = centers[:500]
    print('Getting info for patches')
    res = client.map(get_info_for_patches, centers, rep_scn=rep_scene, hs=hs1km)  # center_lon, center_lat, xmin, ymin, xmax, ymax, region, area_def
    info = client.gather(res)
    patch_info = [x for x in info if x is not None]  # get rid of those pesky space pixels.
    #client.restart()
    del (size_x, size_y, overlap1km, centers, res, info)
    #gc.collect()
    client.close()
    print('getting dictionary of patches per datetime.')
    patch_info_dicts = {}
    for l in range(len(dts)):
        print('date ', l, ' of ', len(dts))
        single_dt = dts[l]
        info = []
        counter = 0
        for p in patch_info:
            info.append(get_patch_percents(p, rootoutdir, single_dt, radar_grid))
            counter = counter + 1
        #res = client.map(get_patch_percents, patch_info, rootoutdir=rootoutdir, single_dt=single_dt, radar_grid=radar_grid)
        #info = client.gather(res)
        dict_key = single_dt.strftime('%Y%m%d%H%M%S')
        patch_info_dicts[dict_key] = info
        #client.restart()
        #del (single_dt, dict_key)
        #gc.collect()
    # by the end of this we should have a list of dictionaries. Each dictionary key is a datetime and contains all the patches
    # in the particular datetime.
    print('deleting no coverage')
    patch_info_dicts = delete_no_cov_patches(patch_info_dicts, dates)
    print('deleting 0% radars randomly')
    patch_info_dicts = delete_random_zero_radar_patches(patch_info_dicts, dates)
    gc.collect()

    prev_mem = current_memory()
    print('Grabbing Patches')
    print(f"Initial memory: {prev_mem:>0.05f}")
    t_loop = time.time()
    for i in range(len(abifile_lists)):  # Call Dask Client
        dt = dts[i]  # should have one list of files per datetime so the dt list should line up with the abifile_lists
        # Get patch info for datetime from dictionary
        dict_key = dt.strftime('%Y%m%d%H%M%S')
        try:
            patch_info = patch_info_dicts[dict_key]
        except:
            continue

        print('number of patches in dt', len(patch_info))
        # Get maximum radar file from datetime
        try:
            max_radar_file = dt.strftime(os.path.join(rootoutdir, 'MRMS_targets/%Y/%m/%d/%Y_%m_%d_%H_%M_%j.nc'))
            data = Dataset(max_radar_file, 'r')
            max_radar_10C = data.groups['MRMS_one_km'].variables['reflectivity_60min_neg10C_max'][:, :]
            #file_test.close()
        except:
            continue

        #####ABI Patch Cropping and Scene Saving
        abi_files = abifile_lists[i]
        scn, bad_data = get_satpy_scene(abi_files, dt)
        if bad_data:  #move to the next timestamp
            del scn
            continue

        for patches in patch_info:
            patch_res = get_patch(patches, scn, dt, rootoutdir, hs1km, max_radar_10C, radar_grid)
            #radar_res = add_radar_to_nc(patch_res, max_radar_file, hs1km)
            del patch_res#, radar_res
        curr_mem = current_memory()
        print(f"| Count: {i:>03d} | Num Patches:{len(patch_info)} | Date Processing: {dt:%Y-%m-%d %H%M%S} | Current memory usage: {current_memory():>0.05f} | Memory delta: {curr_mem - prev_mem:>0.05f} |")
        del (dict_key, patch_info, max_radar_file, abi_files, scn, bad_data)
        gc.collect()

        #can use dask here instead of a loop.
        # patch_res = client.map(get_patch, patch_info, scn=scn, dt=dt, rootoutdir=rootoutdir, hs1km=hs1km, max_radar_10C=max_radar_10C, radar_grid=radar_grid)
        # final_res = client.gather(patch_res)
        # curr_mem = current_memory()
        # print(
        #     f"| Count: {i:>03d} | Num Patches:{len(patch_info)} | Date Processing: {dt:%Y-%m-%d %H%M%S} | Current memory usage: {current_memory():>0.05f} | Memory delta: {curr_mem - prev_mem:>0.05f} |")
        # client.restart()
        # data.close()
        # del (dict_key, patch_info, max_radar_file, abi_files, scn, bad_data, patch_res, final_res)
        # gc.collect()
        # print('After client restart:')
        # curr_mem = current_memory()
        # print(f"| Count: {i:>03d} | Date Processing: {dt:%Y-%m-%d %H%M%S} | Current memory usage: {current_memory():>0.05f} | Memory delta: {curr_mem - prev_mem:>0.05f} |")
        # current_process = psutil.Process(os.getpid())
        # for child in current_process.children(recursive=True):
        #     print(f'|Child: {child} | Memory: {child.memory_percent():>0.05f}')
        prev_mem = curr_mem
    print('Time to complete: ', time.time() - t_start)