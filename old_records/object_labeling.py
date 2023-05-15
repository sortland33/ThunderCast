from pyproj import Proj
import sys,os
sys.path.insert(0,'/home/sbradshaw/ci_ltg')
os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
import utils
import numpy as np
import numpy.ma as ma
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import zarr
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime,timedelta
import pyresample as pr
from glob import glob
import utils
import shutil
import pygrib as pg
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

PWD = os.environ['PWD']+'/'

#####################
<<<<<<< HEAD
dates = [datetime(2018, 5, 31), datetime(2019, 5, 31)]#, datetime(2019, 5, 29), datetime(2019, 5, 30)]#,
         #datetime(2019, 5, 24), datetime(2019, 5, 25), datetime(2019, 5, 26)]
=======
dates = [datetime(2018, 8, 12), datetime(2018, 8, 13), datetime(2018, 8, 14), datetime(2018, 8, 15),
         datetime(2018, 8, 16), datetime(2018, 8, 17)]
>>>>>>> 01c66bd740d09102763e257c497829d59112eb98
toverall = time.time()
needwdss = []
for n in range(len(dates)):
    #Define data directories
    strdate = str(dates[n])
    firstdt = datetime.strptime(strdate, '%Y-%m-%d %H:%M:%S')
    print('Date: ', firstdt)
    lastdt = firstdt + timedelta(hours=25)
    rootradardir = '/apollo/grain/saved_ci_ltg_data/radar/'
    radardir_top = firstdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
    radardir_end = lastdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
    wdssdir_top = firstdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/wdss2_objs/ClusterID/scale_0/'))
    wdssdir_end = lastdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/wdss2_objs/ClusterID/scale_0/'))
    outdir = firstdt.strftime('/apollo/grain/saved_ci_ltg_data/LabeledGrids/1hour/%Y/%Y%m%d/')
    utils.mkdir_p(outdir)

    if not os.path.isdir(wdssdir_top):
        needwdss.append(wdssdir_top)
        continue
    if not os.path.isdir(wdssdir_end):
        needwdss.append(wdssdir_end)
        continue

    #Create list of all WDSS files for the day plus an hour in the next day
    wdss_ncs_top = np.sort(glob(os.path.join(wdssdir_top, '*.netcdf')))
    wdss_ncs_nextday = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                        glob(os.path.join(wdssdir_end, '*.netcdf'))}
    wdss_ncs_end = [wdss_ncs_nextday[x] for x in wdss_ncs_nextday if firstdt <= x <= lastdt]
    wdss_netcdfs_all = np.sort(np.concatenate((wdss_ncs_top, wdss_ncs_end)))

    #Create list of all Radar files for the day plus an hour in the next day
    #Recall here that we have a radar file from the previous day saved in the current day because of WDSS2
    iso_ncs_ct = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                  glob(os.path.join(radardir_top, '*.netcdf'))}
    iso_ncs_top = [iso_ncs_ct[x] for x in iso_ncs_ct if firstdt <= x <= lastdt]
    iso_ncs_nextday = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                       glob(os.path.join(radardir_end, '*.netcdf'))}
    nextdt = lastdt - timedelta(hours = 1)
    iso_ncs_end = [iso_ncs_nextday[x] for x in iso_ncs_nextday if nextdt <= x <= lastdt]
    iso_ncs = np.sort(np.concatenate((iso_ncs_top, iso_ncs_end)))

    #Get general WDSSII info
    wdssobjs = Dataset(wdss_netcdfs_all[0], 'r')
    nlat = len(wdssobjs.dimensions['Lat'])
    nlon = len(wdssobjs.dimensions['Lon'])
    NW_lat = getattr(wdssobjs, 'Latitude')
    NW_lon = getattr(wdssobjs, 'Longitude')
    dy = getattr(wdssobjs, 'LonGridSpacing')
    dx = getattr(wdssobjs, 'LatGridSpacing')
    wdssobjs.close()

    #Get general Radar info
    dataset_10C = Dataset(iso_ncs[0], 'r')
    nlat_10C = len(dataset_10C.dimensions['Lat'])
    nlon_10C = len(dataset_10C.dimensions['Lon'])
    NW_lat_10C = getattr(dataset_10C, 'Latitude')
    NW_lon_10C = getattr(dataset_10C, 'Longitude')
    dy_10C = getattr(dataset_10C, 'LonGridSpacing')
    dx_10C = getattr(dataset_10C, 'LatGridSpacing')
    dataset_10C.close()

    ## List of all dts for a date
    dts_all = []
    for m in range(len(wdss_netcdfs_all)):
        #Create array of start dts
        basefile = os.path.basename(wdss_netcdfs_all[m])
        parts = basefile.split('-')
        year_day = parts[0][:]
        time1 = parts[1][0:6]
        timestamp_str = str(year_day) + '-' + str(time1).zfill(4)
        dts_all.append(timestamp_str)

    #Trim list of datetimes because we only need to do the ones for a particular day (i.e. 6/26/2019)
    dts_date = {datetime.strptime(x, '%Y%m%d-%H%M%S'): x for x in dts_all}
    day_end_dt = firstdt + timedelta(hours=24)
    dts = np.sort([dts_date[x] for x in dts_date if firstdt <= x <= day_end_dt])

    #For each timestep in the day, we want to create a labeled grid of radar objects to 1 hour in the future.
    for k in range(len(dts)):
        print('Loop number: ', k, ' of ', len(dts))
        print('The datetime is: ', dts[k])

        #Instantiate label grid to be output later
        label_grid = np.zeros((nlat, nlon))  # should overwrite existing grid with zeros
        label_lat_shape, label_lon_shape = label_grid.shape

        #Grab WDSS object files for an hour out from the timestep
        topdt = datetime.strptime(dts[k], '%Y%m%d-%H%M%S')
        enddt = topdt + timedelta(minutes = 60)
        wdssfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x
                    for x in wdss_netcdfs_all}
        wdss_netcdfs_1hour = np.sort([wdssfiles[x] for x in wdssfiles if topdt <= x <= enddt])
        print(len(wdss_netcdfs_1hour))

        #Grab Radar files for an hour out from the timestep
        iso_ncs_hour = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                      iso_ncs}
        iso_ncs_1hour = np.sort([iso_ncs_hour[x] for x in iso_ncs_hour if topdt <= x <= enddt])
        print(len(iso_ncs_1hour))

        #Get datetime list for one hour
        dts_to_1hour = np.sort([dts_date[x] for x in dts_date if topdt <= x <= enddt])
        print(len(dts_to_1hour))

        #Create or update a dictionary structure of WDSS and radar data
        print('Creating/Updating data structure....')
        t_for_struct = time.time()
        if k ==0: #if this is the first time step we need to first create the structure
            # Create a dictionary with dates, WDSS2 ids, and radar data for each date
            data_struct = {}
            ## Set-up dictionary for each time
            for kk in range(len(dts_to_1hour)):
                t = dts_to_1hour[kk]
                # Get WDSSII info
                wdssobjs = Dataset(wdss_netcdfs_1hour[kk], 'r')
                clusterID = wdssobjs.variables['ClusterID'][:]
                # Get matching radar reflectivity at -10C
                radardata = Dataset(iso_ncs_1hour[kk], 'r')
                radardataset_10C = radardata.variables['Reflectivity_-10C'][:, :]
                data_struct[t] = {'WDSS_IDS': {}, 'Radar_Data': radardataset_10C}
                # print(data_struct[t]['Radar_Data'])
                ids = np.unique(clusterID)
                ids = ids[ids > 0]  # avoids missing pixels
                for uniq_id in ids:
                    inds = np.where(clusterID == uniq_id)
                    data_struct[t]['WDSS_IDS'][uniq_id] = inds
                wdssobjs.close()
                radardata.close()
            print('Time to create data structure: ', time.time() - t_for_struct)
        else: #if this is not the first time step we can update the structure
            #We delete the entry we no longer need (out of range of the hour)
            dict_keys = list(data_struct.keys())
            for j in range(len(dict_keys)):
                if dict_keys[j] not in dts_to_1hour:
                    print('Removing datetime from structure: ', dict_keys[j])
                    del data_struct[dict_keys[j]]
            #We add entries for any new data for the new time step included in the hour
            dict_keys = list(data_struct.keys())
            for j in range(len(dts_to_1hour)):
                if dts_to_1hour[j] not in dict_keys:
                    t = dts_to_1hour[j]
                    new_dt = datetime.strptime(dts_to_1hour[j], '%Y%m%d-%H%M%S')
                    wdssfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x
                                 for x in wdss_netcdfs_all}
                    new_wdss_netcdf = [wdssfiles[x] for x in wdssfiles if x == new_dt]
                    wdssobjs = Dataset(new_wdss_netcdf[0], 'r')
                    clusterID = wdssobjs.variables['ClusterID'][:]
                    isofiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in iso_ncs}
                    new_iso_nc = [isofiles[x] for x in isofiles if x == new_dt]
                    radardata = Dataset(new_iso_nc[0], 'r')
                    radardataset_10C = radardata.variables['Reflectivity_-10C'][:, :]
                    data_struct[t] = {'WDSS_IDS': {}, 'Radar_Data': radardataset_10C}
                    ids = np.unique(clusterID)
                    ids = ids[ids > 0]  # avoids missing pixels
                    for uniq_id in ids:
                        inds = np.where(clusterID == uniq_id)
                        data_struct[t]['WDSS_IDS'][uniq_id] = inds
                    wdssobjs.close()
                    radardata.close()
            print('Time to update data structure: ', time.time() - t_for_struct)

        #For each datetime, determine object reflectivity (CI algorithm) and label the grid accordingly
        print('Determining labeled grid....')
        t0 = time.time()
        for i in range(len(dts_to_1hour)):
            z = dts_to_1hour[i]
            for ii in data_struct[z]['WDSS_IDS']:
                indices = data_struct[z]['WDSS_IDS'][ii]
                radardataset_10C = data_struct[z]['Radar_Data']
                maxref10 = np.max(radardataset_10C[indices]) #determine the maximum reflectivity value in an object
                label_grid_max = np.max(label_grid[indices]) #determine the maximum existing value on the labeled grid
                if maxref10 <= 38: #If the object's maximum reflectivity is in CI range or below it
                    if label_grid_max <= maxref10: #AND if the existing label is less than the maximum
                        label_grid[indices] = maxref10 #Label the grid object with the maximum (does not erase existing CI labels)
                    else: #Otherwise the existing label is not less than the maximum, so we keep the label the same (avoid storm decay)
                        label_grid[indices] = label_grid[indices]
                else: #If the object's maximum reflectivity is greater than 38, it is mature
                    if np.any(np.logical_and(label_grid[indices] > 0, label_grid[indices] <= 38)):  # if the label grid has CI range already or below
                        label_grid[indices] = label_grid[indices] #don't label the object (avoid overlap)
                    else: #If the grid does not have any CI range already
                        #If this is not the first time step and there are not any previously labeled objects in the area,
                        #this object would have initiated in the time step so we need to label it CI:
                        if i != 0 and label_grid_max == 0:
                            label_grid[indices] = 38
                        else: #Otherwise this is an existing mature area or it is our first timestep so we should label the grid with the maximum
                            label_grid[indices] = maxref10

        print('Time to create labeled grid: ', time.time()-t0)
        print('Saving overall truth grid')
        #Save the labeled grid as a netcdf
        output_file = os.path.join(outdir, ('_').join(['Labeled', 'Grid', dts[k] + '.netcdf']))
        print(output_file)
        rootgrp = Dataset(output_file, 'w')
        labelgroup = rootgrp.createGroup('Labeled_Grids')
        labelgroup.createDimension('Lat', label_grid.shape[0])
        labelgroup.createDimension('Lon', label_grid.shape[1])
        labeled_grid_data = labelgroup.createVariable('Labeled_Ref_-10C_grid', 'float32', ('Lat', 'Lon'))
        labeled_grid_data[:] = label_grid
        labelgroup.units = 'dBZ'
        rootgrp.TypeName = 'Reflectivity_-10C'
        rootgrp.DataType = 'LatLonGrid'
        rootgrp.NWLatitude = NW_lat_10C
        rootgrp.NWlongtitude = NW_lon_10C
        rootgrp.LatGridSpacing = dx_10C
        rootgrp.LonGridSpacing = dy_10C
        rootgrp.Height = 500
        rootgrp.close()
print('Time to run date: ', time.time() - toverall)
print('Need WDSS for:', needwdss)
