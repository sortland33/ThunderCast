#Author Stephanie M. Ortland
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
import utils
import traceback
from dask import delayed
import gc

os.environ['HDF5_USE_FILE_LOCKING']="FALSE" #allows access to files on server

sys.path.insert(0,'/home/sortland/ci_ltg') #specify current working directory- modify as needed
PWD = os.environ['PWD']+'/'

#Define functions
def get_dates(DTfilelist): # Gets info from DT configuration list with dates of interest
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

def get_datetimes(rootdatadir, dates): #get all datetimes for each date
    dts = []
    for n in range(len(dates)):
        dts_per_date = []
        topdt = datetime.strptime(dates[n], '%Y-%m-%d')
        enddt = topdt + timedelta(hours=25)
        satdir = topdt.strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        netcdfpattern_C02 = topdt.strftime(satdir + 'OR_ABI-L1b-RadC-M*C02_G16_s%Y*')
        sat_netcdfs = np.sort(glob(netcdfpattern_C02))
        for i in range(len(sat_netcdfs)):
            sat_netcdf = sat_netcdfs[i]
            basefile = os.path.basename(sat_netcdf)
            parts = basefile.split('_')
            separator = '_'
            year_day_time = parts[3][1:-1]
            dt_start = datetime.strptime(year_day_time, '%Y%j%H%M%S')
            dts.append(dt_start)
            dts_per_date.append(dt_start)
    return dts

def get_abifile_lists(rootdatadir, dts): #get list of advanced baseline imager files (each channel) for each datetime
    list = []
    abi_channels = ['02', '05', '11', '13', '14', '15'] #for this application we do not need all the channels
    for i in range(len(dts)):
        satdir = dts[i].strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        filenames = []
        bad_data = False
        for ch in abi_channels:
            netcdfpattern = dts[i].strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
            try:
                filenames.append(glob(netcdfpattern)[0])
            except IndexError:
                print('Index error for ', netcdfpattern)
                bad_data = True
        if bad_data:
            filenames = []
        list.append(filenames)
    return list

def get_rep_radar_info(dt, rootradardir): #get information about radar- considering this representative of all radar files
    #get information from file
    radardir = dt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
    files = glob(os.path.join(radardir, '*.netcdf'))
    dataset_10C = Dataset(files[0], 'r')
    nlat_10C = len(dataset_10C.dimensions['Lat'])
    nlon_10C = len(dataset_10C.dimensions['Lon'])
    NW_lat_10C = getattr(dataset_10C, 'Latitude')
    NW_lon_10C = getattr(dataset_10C, 'Longitude')
    dy_10C = getattr(dataset_10C, 'LonGridSpacing')
    dx_10C = getattr(dataset_10C, 'LatGridSpacing')
    dataset_10C.close()

    # Get radar grid
    radar_lons_10C = np.zeros((nlat_10C, nlon_10C))
    radar_lats_10C = np.zeros((nlat_10C, nlon_10C))
    for ii in range(nlat_10C):
        radar_lats_10C[ii, :] = NW_lat_10C - ii * dy_10C
    for ii in range(nlon_10C):
        radar_lons_10C[:, ii] = NW_lon_10C + ii * dx_10C
    radar_grid_vals= [radar_lons_10C, radar_lats_10C]
    rad_info = [nlat_10C, nlon_10C, NW_lat_10C, NW_lon_10C, dy_10C, dx_10C]
    return radar_grid_vals, rad_info

def get_rep_files(file_list): #use satpy to get a represtative scene (area definition and sizes of x and y)
    files = file_list[0]
    rep_scene = Scene(reader='abi_l1b', filenames=files)
    try:
        rep_scene.load(['C13', 'C15', 'C11', 'C14', 'C02', 'C05'])
    except:
        try:
            files = file_list[1]
            rep_scene = Scene(reader='abi_l1b', filenames=files)
            rep_scene.load(['C13', 'C15', 'C11', 'C14', 'C02', 'C05'])
        except:
            print('Representative scene would not load.')
            sys.exit()

    area_def = rep_scene['C05'].attrs['area']
    size_x = rep_scene['C05'].sizes['x']
    size_y = rep_scene['C05'].sizes['y']
    print('size_x', size_x)
    print('size_y', size_y)
    del rep_scene
    return(area_def, size_x, size_y)

def get_radar_files(dt, rootradardir): #get the radar files for the datetime + 1 hour
    # Radar files were saved specificially for this project and can be used without copying first
    enddt = dt + timedelta(hours=1)
    radardir_top = dt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
    radardir_end = enddt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
    if radardir_top == radardir_end:
        radfiles_10C = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in glob(os.path.join(radardir_top, '*.netcdf'))}
        radfilelist_10C_orig = [radfiles_10C[x] for x in radfiles_10C if dt < x < enddt]
        radfilelist_10C = np.sort(radfilelist_10C_orig)
    else:
        iso_ncs_ct = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                      glob(os.path.join(radardir_top, '*.netcdf'))}
        iso_ncs_top = [iso_ncs_ct[x] for x in iso_ncs_ct if dt <= x <= enddt]
        iso_ncs_nextday = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                           glob(os.path.join(radardir_end, '*.netcdf'))}
        nextdt = datetime(enddt.year, enddt.month, enddt.day) #we want to avoid an extra file saved in this folder for WDSSII work so we can't just use the end and beginning date
        iso_ncs_end = [iso_ncs_nextday[x] for x in iso_ncs_nextday if nextdt <= x <= enddt]
        radfilelist_10C = np.sort(np.concatenate((iso_ncs_top, iso_ncs_end)))
    return radfilelist_10C

def get_max_radar(rad_info, radfilelist_10C): #calculate the maximum radar for all the radar files (for a datetime + 1 hour)
    nlat_10C = rad_info[0]
    nlon_10C = rad_info[1]
    nfile_10C = len(radfilelist_10C)
    radardataset_10C = np.zeros((nfile_10C, nlat_10C, nlon_10C))
    # Put radar in multi-dim array for easier use later.
    for k in range(nfile_10C):
        radardata = Dataset(radfilelist_10C[k], 'r')
        radardataset_10C[k, :, :] = radardata.variables['Reflectivity_-10C'][:, :]
        radardata.close()
    ###Get maxiumums for time period
    max_radar_10C = np.nanmax(radardataset_10C, axis=0)
    return max_radar_10C

def save_max_nc(max_radar_10C, rootoutdir, dt, rad_info): #save the maximum radar in a netcdf
    NCoutdir = dt.strftime(os.path.join(rootoutdir, 'MRMS_targets/%Y/%m/%d'))
    utils.mkdir_p(NCoutdir)
    NCfilepath = os.path.join(NCoutdir, dt.strftime('%Y_%m_%d_%H_%M_%j.nc'))
    print('NCfilepath:', NCfilepath)
    root = Dataset(NCfilepath, 'w', format='NETCDF4')
    root.description = 'MRMS Radar Target Datasets (CONUS)'
    root.nlat = rad_info[0] #rad_info = [nlat_10C, nlon_10C, NW_lat_10C, NW_lon_10C, dy_10C, dx_10C]
    root.nlon = rad_info[1]
    root.NW_lat = rad_info[2]
    root.NW_lon = rad_info[3]
    root.dy = rad_info[4]
    root.dx = rad_info[5]
    radargroup = root.createGroup('MRMS_one_km')
    print('shapes:', max_radar_10C.shape[0], max_radar_10C.shape[1])
    radargroup.createDimension('y', max_radar_10C.shape[0])
    radargroup.createDimension('x', max_radar_10C.shape[1])
    reflectivity_max = radargroup.createVariable('reflectivity_60min_neg10C_max', 'float32',
                                                 ('y', 'x'))
    reflectivity_max[:, :] = max_radar_10C
    root.close()
    return


#Main program
if __name__ == "__main__":

    #user inputs for program, can run python save_max_radar.py -DTfile DTfile.config as an example
    parser = argparse.ArgumentParser(description='Takes a list of datetimes and creates netcdfs of data patches for CI model.')
    parser.add_argument('-a','--abidir',help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/',
                        default=['/arcdata/goes/grb/goes16/'],type=str,nargs=1)
    parser.add_argument('-r', '--radardir', help='Root directory with radar netcdfs.', default=['/apollo/grain/saved_ci_ltg_data/radar'],
                        nargs=1, type=str)
    parser.add_argument('-l', '--local_dir', help='Local directory for copying files.', type=str, nargs=1, default=['/apollo/grain/sbradshaw/CopiedData/'])
    parser.add_argument('-o','--outdir',help='Output directory.',type=str, nargs=1, default=['/ships19/grain/convective_init/'])
    parser.add_argument('-DTfile', '--DTfilelist', help='A list of lats/lons/datetimes.', type=str, nargs=1)
    parser.add_argument('-RepDir', '--repdir', help='Representative file output directory.',type=str, nargs=1, default=['/ships19/grain/convective_init'])
    parser.add_argument('-repfile', '--add_rep_file', help='If you want to save a representative satpy file (if you do not have one already).',
                        action='store_true')
    args = parser.parse_args()

    ###get information from user inputs
    rootoutdir = args.outdir[0]
    rootdatadir = args.abidir[0]
    rootradardir = args.radardir[0]
    rootrepdir = args.repdir[0]
    utils.mkdir_p(rootoutdir)  # make sure output folder exists
    fileloc = args.local_dir[0]

    dates = get_dates(args.DTfilelist[0]) #get dates from input file
    print(dates)
    dts = get_datetimes(rootdatadir, dates) #list of datetimes
    abifile_lists = get_abifile_lists(rootdatadir, dts) #list of list of files per datetime
    radfile_lists = [get_radar_files(dt, rootradardir) for dt in dts] #get radar file lists

    #To speed this up, dask can be used with the commented out code below
    # client = Client(n_workers=8, threads_per_worker=1, memory_limit='5GB') #Call Dask Client
    # print(client)
    # area_def, size_x, size_y = get_rep_files(abifile_lists) #Representative scene for 2km area def and sizes.

    radar_grid_vals, rad_info = get_rep_radar_info(dts[0], rootradardir)

    indices = list(range(len(abifile_lists)))
    print('ABI files for test:', len(abifile_lists))
    # res = client.map(collect_max_radar, indices, abifile_lists=abifile_lists, dts=dts,
    #                  radfile_lists=radfile_lists, area_def=area_def, radar_grid_vals=radar_grid_vals, rootoutdir=rootoutdir, rad_info=rad_info)
    # client.close()

    for i in indices:
        print('i:', i)
        dt = dts[i]  # should have one list of files per datetime so the dt list should line up with the abifile_lists
        rad_files = radfile_lists[i]
        abi_files = abifile_lists[i]
        if len(abi_files) == 0:
            continue
        #####Add radar to nc files
        if len(rad_files) == 0:
            continue  # if there are no radar files within the hour, skip this timestep
        print('getting max radar')
        max_radar_10C = get_max_radar(rad_info, rad_files)
        print('remapping radar')
        #remapped_radar_10C = resample_max_radar(max_radar_10C, area_def, radar_grid_vals)
        print('saving file')
        save_max_nc(max_radar_10C, rootoutdir, dt, rad_info)
        del max_radar_10C