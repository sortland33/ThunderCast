from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
os.environ['SATPY_CONFIG_PATH'] = '/home/sortland/thundercast/PPP_config/'
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from pyproj import Proj
import time
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import errno
import numpy.ma as ma
import random
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.spatial import ConvexHull
import psutil

os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"
#print('using tobac version', str(tobac.__version__))

def extract_date(ds):
    for var in ds.variables:
        ds_time = datetime.fromtimestamp(ds.attrs['Time'])
        return ds.assign(time=ds_time)

def get_radar_info(radfiles, start, end, test, time_first_30dBZ, lead_time_30dBZ, ti, no_cov_rad, obj_rads, obj_rad_areas, count, time_first_35dBZ, time_first_40dBZ, time_first_45dBZ, loaded_rad):
    radar_subset = np.sort([radfiles[x] for x in radfiles if start <= x < end])
    radar_times = np.sort([x for x in radfiles if start <= x < end])
    for f in range(len(radar_subset)):
        current_time = radar_times[f]
        if current_time in loaded_rad.keys():
            rad_test = loaded_rad[current_time]
        else:
            data = Dataset(radar_subset[f], 'r')
            rad_test = data.variables['Reflectivity_-10C'][:, :]
            loaded_rad[current_time] = rad_test
            data.close()
        obj_rad_vals = np.array(rad_test[np.squeeze(test)])
        obj_rads[current_time] = obj_rad_vals
        obj_rad_areas['25'][current_time] = (obj_rad_vals >= 25).sum()
        obj_rad_areas['30'][current_time] = (obj_rad_vals >= 30).sum()
        obj_rad_areas['35'][current_time] = (obj_rad_vals >= 35).sum()
        obj_rad_areas['40'][current_time] = (obj_rad_vals >= 40).sum()
        obj_rad_areas['45'][current_time] = (obj_rad_vals >= 45).sum()
        obj_rad_areas['50'][current_time] = (obj_rad_vals >= 50).sum()

        if np.isnan(obj_rad_vals).all():
            obj_max_rad = 0
        else:
            if np.nanmin(obj_rad_vals) < -99:
                no_cov_rad = True
            obj_max_rad = np.nanmax(obj_rad_vals)
        if obj_max_rad >= 30:
            if not time_first_30dBZ:
                time_first_30dBZ = current_time
                lead_time_30dBZ = current_time - ti
        if obj_max_rad >= 35:
            if not time_first_35dBZ:
                time_first_35dBZ = current_time
        if obj_max_rad >= 40:
            if not time_first_40dBZ:
                time_first_40dBZ = current_time
        if obj_max_rad >= 45:
            if not time_first_45dBZ:
                time_first_45dBZ = current_time
    return time_first_30dBZ, lead_time_30dBZ, no_cov_rad, obj_rads, obj_rad_areas, time_first_35dBZ, time_first_40dBZ, time_first_45dBZ, loaded_rad

def get_qi_info(qifiles, start, end, test, count, obj_qis, loaded_qi):
    subset = np.sort([qifiles[x] for x in qifiles if start <= x < end])
    qi_times = np.sort([x for x in qifiles if start <= x < end])
    for f in range(len(subset)):
        current_time = qi_times[f]
        if current_time in loaded_qi.keys():
            qi_test = loaded_qi[current_time]
        else:
            data = Dataset(subset[f], 'r')
            qi_test = data.variables['Reflectivity_QI'][:,:]
            loaded_qi[current_time] = qi_test
            data.close()
        obj_qi_vals = np.array(qi_test[np.squeeze(test)])
        obj_qis[current_time] = obj_qi_vals
    return obj_qis, loaded_qi

def get_glm_info(glmfiles, start, end, time_first_glm, lead_time_glm, ti, seg_poly, poly_extent, loaded_glm):
    glm_subset = np.sort([glmfiles[x] for x in glmfiles if start <= x < end])
    glm_times = np.sort([x for x in glmfiles if start <= x < end])
    lon_max, lon_min, lat_max, lat_min = poly_extent
    for f_idx in range(len(glm_subset)):
        f = glm_subset[f_idx]
        t_glm = glm_times[f_idx]
        if time_first_glm:
            break
        if t_glm in loaded_glm.keys():
            glm = loaded_glm[t_glm]
        else:
            ds_glm = xr.open_dataset(f, engine='netcdf4')
            d = {'event_id': ds_glm.variables['event_id'], 'longitude': ds_glm.variables['event_lon'], 'latitude': ds_glm.variables['event_lat']}
            glm = pd.DataFrame(data=d)
            loaded_glm[t_glm] = glm
            ds_glm.close()
        #limit the scope so it doesn't take as long to process
        glm = glm[glm['latitude'] >= lat_min]
        glm = glm[glm['latitude'] <= lat_max]
        glm = glm[glm['longitude'] >= lon_min]
        glm = glm[glm['longitude'] <= lon_max]
        #print(glm)
        if glm.empty:
            #print('glm empty dataframe')
            del glm
            continue
        else:
            glm = glm.drop_duplicates(subset=['longitude', 'latitude'])
            for idx in range(glm.shape[0]): #shape of 0 gives num rows
                grid_point = sgeom.Point(glm['longitude'].values[idx], glm['latitude'].values[idx])
                if grid_point.within(seg_poly):
                    time_first_glm = t_glm
                    lead_time_glm = time_first_glm - ti
                    del glm
                    break
    return time_first_glm, lead_time_glm, loaded_glm

def get_eni_info(enifiles, start, end, time_first_eni, lead_time_eni, ti, seg_poly, poly_extent, loaded_eni):
    eni_subset = np.sort([enifiles[x] for x in enifiles if start <= x < end])
    eni_times = np.sort([x for x in enifiles if start <= x < end])
    lon_max, lon_min, lat_max, lat_min = poly_extent
    for f_idx in range(len(eni_subset)):
        if time_first_eni:
            break
        f = eni_subset[f_idx]
        t_eni = eni_times[f_idx]
        if t_eni in loaded_eni.keys():
            eni = loaded_eni[t_eni]
        else:
            time_w = lambda x: (x.replace('time:', ''))
            type_w = lambda x: (x.replace('type:', ''))
            lat_w = lambda x: (x.replace('latitude:', ''))
            lon_w = lambda x: (x.replace('longitude:', ''))
            pC_w = lambda x: (x.replace('peakCurrent:', ''))
            ic_w = lambda x: (x.replace('icHeight:', ''))
            ns_w = lambda x: (x.replace('numSensors:', ''))
            icM_w = lambda x: (x.replace('icMultiplicity:', ''))
            cgM_w = lambda x: (x.replace('cgMultiplicity:', ''))
            eni = pd.read_csv(f, names=['time', 'type', 'latitude', 'longitude', 'peakCurrent', 'icHeight',
                                        'numSensors', 'icMultiplicity', 'cgMultiplicity'],
                              converters={'time': time_w, 'type': type_w, 'latitude': lat_w, 'longitude': lon_w,
                                          'peakCurrent': pC_w, 'icHeight': ic_w, 'numSensors': ns_w,
                                          'icMultiplicity': icM_w, 'cgMultiplicity': cgM_w})
            eni['latitude'] = eni['latitude'].astype(float)
            eni['longitude'] = eni['longitude'].astype(float)
            eni['time'] = eni['time'].astype('string')
            eni['time'] = eni['time'].str[1:-12]
            eni['time'] = pd.to_datetime(eni['time'], yearfirst=True, format='mixed')
            loaded_eni[t_eni] = eni
        #limit the extent to the min/max lat/lon to reduce computation time when steping through events
        eni = eni[eni['latitude'] >= lat_min]
        eni = eni[eni['latitude'] <= lat_max]
        eni = eni[eni['longitude'] >= lon_min]
        eni = eni[eni['longitude'] <= lon_max]
        if eni.empty:
            del eni
            continue
        else:
            eni = eni.drop_duplicates(subset=['longitude', 'latitude'])
            for idx in range(eni.shape[0]): #num rows
                grid_point = sgeom.Point(eni['longitude'].values[idx], eni['latitude'].values[idx])
                if grid_point.within(seg_poly):
                    time_first_eni = pd.to_datetime(eni['time'].values[idx])
                    lead_time_eni = time_first_eni - ti
                    del eni
                    break
    return time_first_eni, lead_time_eni, loaded_eni

def get_pred_info(predfiles, frame_time, test, obj_preds, count, obj_pred_max_vals, loaded_pred):
    file = predfiles[frame_time]
    if frame_time in loaded_pred.keys():
        pred_test = loaded_pred[frame_time]
    else:
        data = Dataset(file, 'r')
        pred_test = data.variables['thundercast_preds'][:,:]
        data.close()
    obj_pred_vals = np.array(pred_test[np.squeeze(test)])
    obj_preds[count] = obj_pred_vals
    obj_pred_max_vals[frame_time] = np.nanmax(obj_pred_vals)
    return obj_preds, obj_pred_max_vals, loaded_pred

def get_abi_info(abifiles, frame_time, test, count, obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13,
                 obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi, loaded_abi):
    file = abifiles[frame_time]
    if frame_time in loaded_abi.keys():
        load_dict = loaded_abi[frame_time]
        C13_test = load_dict['C13']
        C02_test = load_dict['C02']
        C05_test = load_dict['C05']
        C11_test = load_dict['C11']
        C14_test = load_dict['C14']
        C15_test = load_dict['C15']
        sol_zen_array = load_dict['sol_zen']
        sol_azi_array = load_dict['sol_azi']
        sat_zen_array = load_dict['sat_zen']
    else:
        data = Dataset(file, 'r')
        load_dict = {}
        C13_test = data.variables['C13'][:, :]
        C02_test = data.variables['C02'][:, :]
        C05_test = data.variables['C05'][:, :]
        C11_test = data.variables['C11'][:, :]
        C14_test = data.variables['C14'][:, :]
        C15_test = data.variables['C15'][:, :]
        sol_zen_array = data.variables['sol_zen'][:, :]
        sol_azi_array = data.variables['sol_azi'][:, :]
        sat_zen_array = data.variables['sat_zen'][:, :]
        load_dict['C13'] = C13_test
        load_dict['C02'] = C02_test
        load_dict['C05'] = C05_test
        load_dict['C11'] = C11_test
        load_dict['C14'] = C14_test
        load_dict['C15'] = C15_test
        load_dict['sol_zen'] = sol_zen_array
        load_dict['sol_azi'] = sol_azi_array
        load_dict['sat_zen'] = sat_zen_array
        data.close()
        loaded_abi[frame_time] = load_dict

    C13_array = np.array(C13_test[np.squeeze(test)])
    min_ind = np.argmin(C13_array)
    obj_abi_C13[count] = C13_array[min_ind] #the minimum BT for C13

    C02_array = np.array(C02_test[np.squeeze(test)])
    obj_abi_C02[count] = C02_array[min_ind] #C02 reflectance at C13 min.

    C05_array = np.array(C05_test[np.squeeze(test)])
    obj_abi_C05[count] = C05_array[min_ind] #C05 reflectance at C13 min.

    C11_array = np.array(C11_test[np.squeeze(test)])
    # obj_abi_C11[count] = C11_array[min_ind]

    C14_array = np.array(C14_test[np.squeeze(test)])
    # obj_abi_C14[count] = C14_array[min_ind]

    C15_array = np.array(C15_test[np.squeeze(test)])
    # obj_abi_C15[count] = C15_array[min_ind]

    obj_abi_diffC11C13[count] = C11_array[min_ind] - C13_array[min_ind]
    obj_abi_diffC11C14[count] = C11_array[min_ind] - C14_array[min_ind]
    obj_abi_diffC13C15[count] = C13_array[min_ind] - C15_array[min_ind]

    sol_zen_array = np.array(sol_zen_array[np.squeeze(test)])
    sol_zen[count] = sol_zen_array[min_ind]

    sol_azi_array = np.array(sol_azi_array[np.squeeze(test)])
    sol_azi[count] = sol_azi_array[min_ind]

    sat_zen_array = np.array(sat_zen_array[np.squeeze(test)])
    sat_zen[count] = sat_zen_array[min_ind]

    return obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13, obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi, loaded_abi

def get_region(lon, lat, center_state=None):
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
    shpfilename = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces')
    reader = shpreader.Reader(shpfilename)
    states = reader.records()
    point = sgeom.Point(lon, lat)

    if not center_state:
        for state in states:
            name = state.attributes['name']
            geom = state.geometry
            if geom.contains(point):
                center_state = state.attributes['name'].rstrip('\x00')
                break
    #print('center_state', center_state)

    shpfilename = shpreader.natural_earth(resolution='50m', category='physical', name='land')
    land_geom = unary_union(list(shpreader.Reader(shpfilename).geometries()))
    land = prep(land_geom)
    def is_land(point):
        return (land.contains(point))

    if is_land(point):
        ocean = 0
    else:
        ocean = 1

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
        region = 'oconus'

    return region, ocean

def get_hrrr_info(dt, all_masks, hrrrfiles, loaded_hrrr):
    all_hrs = []
    for key in all_masks.keys():
        if key.minute > 30:
            all_hrs.append(key.hour + 1)
        else:
            all_hrs.append(key.hour)
    EBS= {}
    EBS_theta = {}
    MUCAPE = {}
    MLCAPE = {}
    MUCIN = {}
    MLCIN = {}
    hrrr_variables = ['EBS_magnitude', 'EBS_theta', 'MUCAPE255', 'MLCAPE180', 'MUCIN255', 'MLCIN180']
    for hr in np.unique(all_hrs):
        new_dt = dt + timedelta(hours=int(hr))
        new_start = new_dt - timedelta(minutes=30)
        new_end = new_dt + timedelta(minutes=30)
        keys_for_hr = [x for x in list(all_masks.keys()) if new_start < x <= new_end]
        new_mask = np.full(all_masks[keys_for_hr[0]].shape, False)
        for key in keys_for_hr:
            new_mask = ma.mask_or(new_mask, all_masks[key])
        if new_dt in loaded_hrrr.keys():
            load_dict = loaded_hrrr[new_dt]
        else:
            hrrr_file = hrrrfiles[new_dt]
            data = Dataset(hrrr_file, 'r')
            load_dict = {}
            for var in hrrr_variables:
                load_dict[var] = data.variables[var][:, :]
            loaded_hrrr[new_dt] = load_dict
            data.close()
        for var in hrrr_variables:
            hrrr_test = load_dict[var]
            holder = np.array(hrrr_test[new_mask])
            if var == 'EBS_magnitude':
                EBS[hr] =  np.where(holder == -999, np.nan, holder)
            if var == 'EBS_theta':
                EBS_theta[hr] = np.where(holder == -999, np.nan, holder)
            if var == 'MUCAPE255':
                MUCAPE[hr] = np.where(holder == -999, np.nan, holder)
            if var == 'MLCAPE180':
                MLCAPE[hr] = np.where(holder == -999, np.nan, holder)
            if var == 'MUCIN255':
                MUCIN[hr] = np.where(holder == -999, np.nan, holder)
            if var == 'MLCIN180':
                MLCIN[hr] = np.where(holder == -999, np.nan, holder)
    return EBS, EBS_theta, MUCAPE, MLCAPE, MUCIN, MLCIN, loaded_hrrr

def add_to_dict(add_dict, value_dict, name, get_max=True, get_min=True, get_avg=True, get_std=True, get_initial_max=True,
                get_initial_avg=True, get_initial_min=True, get_initial_std=True, get_end_max=True, get_end_min=True,
                get_end_avg=True, get_end_std=True):
    all_obj_vals = np.concatenate(list(value_dict.values()), axis=None)
    initial_obj_vals = list(value_dict.values())[0]
    end_obj_vals = list(value_dict.values())[-1]
    if not np.isnan(all_obj_vals).all():
        if get_max:
            add_dict['max_'+name] = np.nanmax(all_obj_vals)
        if get_min:
            add_dict['min_'+name] = np.nanmin(all_obj_vals)
        if get_avg:
            add_dict['avg_'+name] = np.nanmean(all_obj_vals)
        if get_std:
            add_dict['std_'+name] = np.nanstd(all_obj_vals)
        if get_initial_max:
            add_dict['initial_max_'+name] = np.nanmax(initial_obj_vals)
        if get_initial_min:
            add_dict['initial_min_'+name] = np.nanmin(initial_obj_vals)
        if get_initial_avg:
            add_dict['initial_avg_' + name] = np.nanmean(initial_obj_vals)
        if get_initial_std:
            add_dict['initial_std_' + name] = np.nanstd(initial_obj_vals)
        if get_end_max:
            add_dict['end_max_'+name] = np.nanmax(end_obj_vals)
        if get_end_min:
            add_dict['end_min_'+name] = np.nanmin(end_obj_vals)
        if get_end_avg:
            add_dict['end_avg_' + name] = np.nanmean(end_obj_vals)
        if get_end_std:
            add_dict['end_std_' + name] = np.nanstd(end_obj_vals)
    return add_dict

def add_abi_to_dict(add_dict, obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13, obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi):
    C13 = list(obj_abi_C13.values())
    min_ind = np.argmin(C13)
    C02 = list(obj_abi_C02.values())
    C05 = list(obj_abi_C05.values())
    C11C13_BTD = list(obj_abi_diffC11C13.values())
    C11C14_BTD = list(obj_abi_diffC11C14.values())
    C13C15_BTD = list(obj_abi_diffC13C15.values())
    sat_zen = list(sat_zen.values())
    sol_zen = list(sol_zen.values())
    sol_azi = list(sol_azi.values())
    add_dict['min_C13'] = C13[min_ind]
    add_dict['C02'] = C02[min_ind]
    add_dict['C05'] = C05[min_ind]
    add_dict['C11C13_BTD'] = C11C13_BTD[min_ind]
    add_dict['C11C14_BTD'] = C11C14_BTD[min_ind]
    add_dict['C13C15_BTD'] = C13C15_BTD[min_ind]
    add_dict['sat_zen'] = sat_zen[min_ind]
    add_dict['sol_zen'] = sol_zen[min_ind]
    add_dict['sol_azi'] = sol_azi[min_ind]
    return add_dict

def get_polygon(segment_at_time, test):
    segment_lats = segment_at_time.coords['latitude'].values
    masked_lats = segment_lats[np.squeeze(test)]
    segment_lons = segment_at_time.coords['longitude'].values
    masked_lons = segment_lons[np.squeeze(test)]
    extent = [np.nanmax(masked_lons), np.nanmin(masked_lons), np.nanmax(masked_lats), np.nanmin(masked_lats)]
    list_points = []
    for i in range(len(masked_lons)):
        list_points.append([masked_lons[i], masked_lats[i]])
    if len(list_points) < 3:
        seg_poly = None
        extent = None
        return seg_poly, extent
    list_points = np.array(list_points)
    hull = ConvexHull(list_points)
    seg_poly = sgeom.Polygon(list(zip(list_points[hull.vertices, 0], list_points[hull.vertices, 1])))
    return seg_poly, extent

def current_memory(): #get current memory usage
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    return mem


if __name__ == "__main__":
    rootpreddir = '/ships22/grain/sortland/thundercast_ncs'
    rootobjdir = '/ships22/grain/sortland/objects'
    date_dirs = glob(rootobjdir + '/2022/202209*')
    dates = []
    for i in date_dirs:
        dates.append(os.path.basename(i))
    dates = np.sort(dates)
    for date in np.sort(dates):
        date_start = time.time()
        dt = datetime.strptime(date, '%Y%m%d')
        eni_directory = dt.strftime('/ships19/grain/lightning/%Y/%Y_%m_%d_%j/')
        main_directory = outdir = dt.strftime('/ships22/grain/sortland/objects/%Y/%Y%m%d/')
        remapped_directory = dt.strftime('/ships22/grain/sortland/remapped_obj_data/%Y/%Y%m%d/')
        pred_directory = dt.strftime('/ships22/grain/sortland/thundercast_ncs/%Y/%Y%m%d/')
        glm_directory = dt.strftime('/arcdata/goes/grb/goes16/%Y/%Y_%m_%d_%j/glm/L2/LCFA/')
        output_file = outdir + 'object_data.pickle'
        csv_test = outdir + 'object_data.csv'

        # print('getting lists of all the files...')
        radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    np.sort(glob(remapped_directory + 'radar_-10C/*'))}
        loaded_rad = {}
        qifiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    np.sort(glob(remapped_directory + 'radar_qi/*'))}
        loaded_qi = {}
        hrrrfiles = {datetime.strptime(x.split('/')[-1], '%Y-%m-%d-%H.netcdf'): x for x in
                    np.sort(glob(remapped_directory + 'hrrr/*'))}
        loaded_hrrr = {}
        abifiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    np.sort(glob(remapped_directory + 'abi_pc/*'))}
        loaded_abi = {}
        glmfiles = {datetime.strptime(os.path.basename(x).split('_')[3], 's%Y%j%H%M%S%f'): x for x in
                    np.sort(glob(glm_directory + '*'))}
        loaded_glm = {}

        eni_hr_dirs = np.sort(glob(eni_directory + '*'))
        all_eni_txt = []
        for dir in eni_hr_dirs:
            to_add = [x for x in glob(dir + '/*') if '.current' not in x]
            all_eni_txt = all_eni_txt + to_add
        enifiles = {datetime.strptime(x.split('/')[-1], 'entln_%Y%j_%H%M%S.txt'): x for x in all_eni_txt}
        loaded_eni = {}
        predfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    np.sort(glob(pred_directory + '*'))}
        loaded_pred = {}

        #print('opening object files')
        segmentation_file = main_directory + 'segments.netcdf'
        segment_info = xr.open_dataset(segmentation_file, engine='netcdf4')
        feature_file = main_directory + 'features_filtered.pickle'
        feature_info = pd.read_pickle(feature_file)
        all_features = feature_info['feature']
        # for col in feature_info:
        #     print(col)

        all_tracks = list(np.unique(feature_info['ms_track_id'].values))
        num_all_tracks = len(all_tracks)
        #print('This date has ', num_all_tracks, ' tracks.')

        prev_mem = current_memory()
        print(f"Initial memory: {prev_mem:>0.05f}")

        for main_count, track_num in enumerate(all_tracks): #look at each tracked cell in the dataset
            # track_num = 9279
            add_dict = {}
            add_dict['track_num'] = track_num
            track_info = feature_info[feature_info['ms_track_id'] == track_num].reset_index(drop=True)
            max_frame = np.max(track_info['frame'])
            min_frame = np.min(track_info['frame'])
            initial_timestr = track_info.loc[track_info['frame']==min_frame, 'timestr'].values[0]
            second_timestr = track_info.loc[track_info['frame']==(min_frame + 1), 'timestr'].values[0]
            final_timestr = track_info.loc[track_info['frame']==max_frame, 'timestr'].values[0]
            ti = datetime.strptime(initial_timestr, '%Y-%m-%d %H:%M:%S')
            t2 = datetime.strptime(second_timestr, '%Y-%m-%d %H:%M:%S')
            tf = datetime.strptime(final_timestr, '%Y-%m-%d %H:%M:%S')
            duration = tf - ti
            add_dict['object_start_time'] = ti
            add_dict['object_end_time'] = tf
            add_dict['object_duration'] = duration.total_seconds()
            time_between_frames = t2 - ti
            t_half = time_between_frames/2 #this is a timedelta- use this to get the closest files to the time of the predictions

            ###Clear up loaded data
            loaded_rad = {k: loaded_rad[k] for k in loaded_rad if k >= ti}
            loaded_qi = {k: loaded_qi[k] for k in loaded_qi if k >= ti}
            loaded_hrrr = {k: loaded_hrrr[k] for k in loaded_hrrr if k >= ti}
            loaded_abi = {k: loaded_abi[k] for k in loaded_abi if k >= ti}
            loaded_glm = {k: loaded_glm[k] for k in loaded_glm if k >= ti}
            loaded_eni = {k: loaded_eni[k] for k in loaded_eni if k >= ti}
            loaded_pred = {k: loaded_pred[k] for k in loaded_pred if k >= ti}

            skip=False
            time_first_30dBZ = None
            lead_time_30dBZ = None
            time_first_35dBZ = None
            time_first_40dBZ = None
            time_first_45dBZ = None
            time_first_glm = None
            lead_time_glm = None
            lead_time_eni = None
            time_first_eni = None
            classification = None
            no_cov_rad = False
            obj_preds = {}
            obj_rads = {}
            obj_rad_areas = {'25':{}, '30':{}, '35':{}, '40':{}, '45':{}, '50':{}}
            obj_qis = {}
            obj_abi = {}
            #get_records = True
            all_masks = {}
            obj_abi_diffC11C13 = {}
            obj_abi_diffC13C15 = {}
            obj_abi_C02 = {}
            obj_abi_C05 = {}
            obj_abi_C13 = {}
            obj_abi_diffC11C14 = {}
            sat_zen = {}
            sol_zen = {}
            sol_azi = {}
            obj_pred_max = {}

            t_start = time.time()
            #print('num in track', len(np.unique(track_info['frame'])))

            #below count is really the index but we reset those so it should just count up from 0
            for count, sub_track in track_info.iterrows(): #The cells exist for multiple frames so here we can cycle through
                ###memory check- reset if too many are loaded to avoid crashing machines
                mem_check = current_memory()/1000
                if mem_check >= 20:
                    loaded_rad = {}
                    loaded_qi = {}
                    loaded_hrrr = {}
                    loaded_abi = {}
                    loaded_glm = {}
                    loaded_eni = {}
                    loaded_pred = {}
                frame = sub_track['frame']
                frame_time = datetime.strptime(sub_track['timestr'], '%Y-%m-%d %H:%M:%S')

                if count == 0: #determine region of object's first appearance
                    feat_lon = sub_track['longitude']
                    feat_lat = sub_track['latitude']
                    add_dict['initial_lat'] = feat_lat
                    add_dict['initial_lon'] = feat_lon
                    add_dict['initial_region'], add_dict['ocean'] = get_region(feat_lon, feat_lat, center_state=None)

                start = frame_time - t_half
                if start < dt:
                    start = dt
                end = frame_time + t_half

                segment_at_time = segment_info.where(segment_info.coords['time'] == segment_info.coords['time'][frame],
                                                     drop=True)  # gets a subset of dataset for timestamp
                features = [sub_track['feature']]
                test = np.isin(segment_at_time['segmentation_mask'], features)
                if not np.unique(test).any():
                    print('Segmentation mask for time frame does not contain the track features.')
                    skip = True
                    break

                seg_poly, poly_extent = get_polygon(segment_at_time, test) #get polygon for testing point data
                if not seg_poly:
                    print('There were not enough points to make a polygon.')
                    skip = True
                    break

                if time_first_glm and time_first_eni:
                    if time_first_30dBZ:
                        break
                    else:
                        time_first_30dBZ, lead_time_30dBZ, no_cov_rad, obj_rads, obj_rad_areas, time_first_35dBZ, time_first_40dBZ, time_first_45dBZ, loaded_rad = get_radar_info(radfiles, start, end, test, time_first_30dBZ, lead_time_30dBZ, ti, no_cov_rad, obj_rads, obj_rad_areas, count, time_first_35dBZ, time_first_40dBZ, time_first_45dBZ, loaded_rad)
                else:
                    all_masks[frame_time] = np.squeeze(test)
                    obj_preds, obj_pred_max, loaded_pred = get_pred_info(predfiles, frame_time, test, obj_preds, count, obj_pred_max, loaded_pred)
                    time_first_30dBZ, lead_time_30dBZ, no_cov_rad, obj_rads, obj_rad_areas, time_first_35dBZ, time_first_40dBZ, time_first_45dBZ, loaded_rad = get_radar_info(radfiles, start, end, test, time_first_30dBZ, lead_time_30dBZ, ti, no_cov_rad, obj_rads, obj_rad_areas, count, time_first_35dBZ, time_first_40dBZ, time_first_45dBZ, loaded_rad)
                    obj_qis, loaded_qi = get_qi_info(qifiles, start, end, test, count, obj_qis, loaded_qi)
                    obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13, obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi, loaded_abi = get_abi_info(abifiles, frame_time, test, count, obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13, obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi, loaded_abi)
                    if not time_first_glm:
                        time_first_glm, lead_time_glm, loaded_glm = get_glm_info(glmfiles, start, end, time_first_glm, lead_time_glm, ti, seg_poly, poly_extent, loaded_glm)
                    if not time_first_eni:
                        time_first_eni, lead_time_eni, loaded_eni = get_eni_info(enifiles, start, end, time_first_eni, lead_time_eni, ti, seg_poly, poly_extent, loaded_eni)
                    if time_first_glm and time_first_eni and time_first_30dBZ:
                        break

            if skip:
                continue

            #SORT
            if time_first_30dBZ:
                add_dict['class'] = 'tp'
                add_dict['time_first_30dBZ'] = time_first_30dBZ
                #add_dict['lead_time_30dBZ'] = lead_time_30dBZ.total_seconds()
            else:
                add_dict['class'] = 'fp'

            if time_first_35dBZ:
                add_dict['time_first_35dBZ'] = time_first_35dBZ
            if time_first_40dBZ:
                add_dict['time_first_40dBZ'] = time_first_40dBZ
            if time_first_45dBZ:
                add_dict['time_first_45dBZ'] = time_first_45dBZ

            if time_first_eni or time_first_glm:
                add_dict['lightning'] = 1
                if time_first_eni:
                    add_dict['time_first_eni'] = time_first_eni
                if time_first_glm:
                    add_dict['time_first_glm'] = time_first_glm
            else:
                add_dict['lightning'] = 0

            if no_cov_rad:
                add_dict['no_radar_cov'] = 1
            else:
                add_dict['no_radar_cov'] = 0

            #Save Prediction Information
            obj_pred_max_vals = np.array(list(obj_pred_max.values()))
            obj_pred_max_keys = list(obj_pred_max.keys())
            pred_thresh = np.round(np.arange(0, 1, 0.1), 1)
            names = ['pred'+str(int(x*100)) for x in pred_thresh]
            time_thresh = None
            for thresh_count, thresh in enumerate(pred_thresh):
                if thresh == 0:
                    if any(x > thresh for x in obj_pred_max_vals):
                        time_thresh = obj_pred_max_keys[np.argmax(obj_pred_max_vals > thresh)]
                else:
                    if any(x >= thresh for x in obj_pred_max_vals):
                        time_thresh = obj_pred_max_keys[np.argmax(obj_pred_max_vals >= thresh)]
                if time_thresh:
                    if time_first_30dBZ:
                        add_name = 'lead_time_30dBZ_' + names[thresh_count]
                        add_dict[add_name] = (time_first_30dBZ - time_thresh).total_seconds()
                        #print('30dBZ', (time_first_30dBZ - time_thresh).total_seconds())
                    if time_first_glm:
                        add_name = 'lead_time_glm_' + names[thresh_count]
                        add_dict[add_name] = (time_first_glm - time_thresh).total_seconds()
                        #print('glm', (time_first_glm - time_thresh).total_seconds())
                    if time_first_eni:
                        add_name = 'lead_time_eni_' + names[thresh_count]
                        add_dict[add_name] = (time_first_eni - time_thresh).total_seconds()
                        #print('eni', (time_first_eni - time_thresh).total_seconds())

            all_preds = np.concatenate(list(obj_preds.values()), axis=None)
            initial_preds = obj_preds[0] #count=0 so first object frame
            if np.isnan(all_preds).all():
                add_dict['max_pred'] = 0
                add_dict['initial_max_pred'] = 0
                add_dict['avg_pred'] = 0
                add_dict['std_pred'] = 0
            else:
                add_dict['max_pred'] = np.nanmax(all_preds)
                add_dict['initial_max_pred'] = np.nanmax(initial_preds)
                add_dict['avg_pred'] = np.nanmean(all_preds)
                add_dict['std_pred'] = np.nanstd(all_preds)
            del all_preds, initial_preds, obj_preds

            #Save Radar -10C Information
            initial_rads = list(obj_rads.values())[0]
            if not np.isnan(initial_rads).all():
                add_dict['initial_max_dBZ_-10C'] = np.nanmax(initial_rads)
            var_name = 'dBZ_-10C'
            add_dict = add_to_dict(add_dict, obj_rads, var_name, get_min=False, get_avg=False, get_std=False)
            del initial_rads, obj_rads

            #Save Radar Area Information
            obj_rad_areas_25 = np.concatenate(list(obj_rad_areas['25'].values()), axis=None)
            add_dict['max_dBZ_area_25dBZ'] = np.nanmax(obj_rad_areas_25)
            obj_rad_areas_30 = np.concatenate(list(obj_rad_areas['30'].values()), axis=None)
            add_dict['max_dBZ_area_30dBZ'] = np.nanmax(obj_rad_areas_30)
            obj_rad_areas_35 = np.concatenate(list(obj_rad_areas['35'].values()), axis=None)
            add_dict['max_dBZ_area_35dBZ'] = np.nanmax(obj_rad_areas_35)
            obj_rad_areas_40 = np.concatenate(list(obj_rad_areas['40'].values()), axis=None)
            add_dict['max_dBZ_area_40dBZ'] = np.nanmax(obj_rad_areas_40)
            obj_rad_areas_45 = np.concatenate(list(obj_rad_areas['45'].values()), axis=None)
            add_dict['max_dBZ_area_45dBZ'] = np.nanmax(obj_rad_areas_45)
            obj_rad_areas_50 = np.concatenate(list(obj_rad_areas['50'].values()), axis=None)
            add_dict['max_dBZ_area_50dBZ'] = np.nanmax(obj_rad_areas_50)
            del obj_rad_areas

            #Save Radar QI Information
            var_name = 'qi'
            add_dict = add_to_dict(add_dict, obj_qis, var_name, get_initial_max=False, get_initial_avg=False, get_initial_min=False, get_initial_std=False, get_end_max=False, get_end_min=False, get_end_avg=False, get_end_std=False)
            del obj_qis

            #Save ABI
            add_dict = add_abi_to_dict(add_dict, obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13,
                                       obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi)
            del obj_abi_C02, obj_abi_C05, obj_abi_C13, obj_abi_diffC11C13, obj_abi_diffC11C14, obj_abi_diffC13C15, sat_zen, sol_zen, sol_azi

            # Save HRRR Information
            EBS, EBS_theta, MUCAPE, MLCAPE, MUCIN, MLCIN, loaded_hrrr = get_hrrr_info(dt, all_masks, hrrrfiles, loaded_hrrr)
            var_name = 'EBS_magnitude'
            add_dict = add_to_dict(add_dict, EBS, var_name, get_min=False, get_initial_min=False)
            var_name = 'EBS_theta'
            add_dict = add_to_dict(add_dict, EBS_theta, var_name, get_min=False, get_initial_min=False)
            var_name = 'MUCAPE255'
            add_dict = add_to_dict(add_dict, MUCAPE, var_name, get_min=False, get_initial_min=False)
            var_name = 'MLCAPE180'
            add_dict = add_to_dict(add_dict, MLCAPE, var_name, get_min=False, get_initial_min=False)
            var_name = 'MUCIN255'
            add_dict = add_to_dict(add_dict, MUCIN, var_name, get_max=False, get_initial_max=False)
            var_name = 'MLCIN180'
            add_dict = add_to_dict(add_dict, MLCIN, var_name, get_max=False, get_initial_max=False)
            del EBS, EBS_theta, MUCAPE, MLCAPE, MUCIN, MLCIN, all_masks

            if main_count == 0:
                df = pd.DataFrame([add_dict])
            else:
                add_df = pd.DataFrame([add_dict])
                df = pd.concat([df, add_df])
                del add_df
            del add_dict
            curr_mem = current_memory()
            t_complete = time.time()-t_start
            print(f"| Date Processing: {dt:%Y-%m-%d} | Loop: {main_count}/{num_all_tracks}| Track_num: {track_num} | Time to Complete (sec):{t_complete:0.2f} | Current memory usage: {current_memory():>0.05f} | Memory delta: {curr_mem - prev_mem:>0.05f} |")
            prev_mem = curr_mem
        df.to_pickle(output_file)
        df.to_csv(csv_test)
        print('Time to complete ', date, ' ', time.time() - date_start)