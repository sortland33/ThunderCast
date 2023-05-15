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

#command for segmotion
#w2segmotionll -i code_index.xml -o test01 -T MergedReflectivityQCComposite -d "30 43 3 -1" -t "0" -p "25,70,100:0:0,0,0" -k percent:50:2:0:2 -X /data/probsevere//src/static/clusterTable.xml

###Let's first copy all the radar data we want for the time period we specified above.
#date = datetime(2019, 6, 19)
#strdate = str(date)
#topdt = datetime.strptime(strdate, '%Y-%m-%d %H:%M:%S')

#rootradardir='/apollo/grain/saved_probsevere_data/radar'
#radardir = topdt.strftime(os.path.join(rootradardir, '%Y%m%d/MergedReflectivityQCComposite/00.50/'))

state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='none')
datadir = '/data/sbradshaw/segmotion_out2/test20190626_-10C/ClusterID/scale_0/'
radardatadir = '/data/sbradshaw/segmotion_out2/'
netcdfs = glob(os.path.join(datadir, '*.netcdf'))
print(netcdfs)
radardir = os.path.join(radardatadir, 'MergedReflectivityQCComposite/00.50/')
isodir = os.path.join(radardatadir, 'Reflectivity_-10C/00.50/')
outdir = '/data/sbradshaw/segmotion_out2/visualization/test_20190626/scale_0/'
isooutdir = '/data/sbradshaw/segmotion_out2/visualization/test_20190626/Obj_Plot_MergedRef/'

goesnc = Dataset('/home/sbradshaw/ci_ltg/proj_files/GOES_East.nc', 'r')
gip = goesnc.variables['goes_imager_projection']

geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                             sweep_axis=gip.sweep_angle_axis, \
                             satellite_height=gip.perspective_point_height)

goesvis = Dataset('/apollo/grain/saved_ci_ltg_data/netcdf/2019/2019_06_05_156/goes16_2019_06_05_11_41_156.nc', 'r')
xv = goesvis.groups['half_km']['x']
yv = goesvis.groups['half_km']['y']
xxv,yyv = np.meshgrid(xv,yv)
x1km = goesvis.groups['one_km']['x']
y1km = goesvis.groups['one_km']['y']
xx1km, yy1km = np.meshgrid(x1km, y1km)
x = goesvis.groups['two_km']['x']
y = goesvis.groups['two_km']['y']
xmin = np.min(x); xmax = np.max(x); ymin = np.min(y); ymax = np.max(y)

sat_group = goesvis.groups['one_km']
sat_lons = sat_group.variables['longitude'][:]
sat_lats = sat_group.variables['latitude'][:]
sat_grid = pr.geometry.GridDefinition(lons=sat_lons, lats=sat_lats)

for i in range(len(netcdfs)):

    wdssobjs = Dataset(netcdfs[i], 'r')
    clusterID = wdssobjs.variables['ClusterID'][:]
    nlat = len(wdssobjs.dimensions['Lat'])
    nlon = len(wdssobjs.dimensions['Lon'])
    NW_lat = getattr(wdssobjs, 'Latitude')
    NW_lon = getattr(wdssobjs, 'Longitude')
    dy = getattr(wdssobjs, 'LonGridSpacing')
    dx = getattr(wdssobjs, 'LatGridSpacing')

    basefile = os.path.basename(netcdfs[i])
    parts = basefile.split('-')
    print(parts)
    year_day = parts[0][:]
    time = parts[1][0:6]

    timestamp = str(year_day) + '-' + str(time).zfill(4)
    print(timestamp)

    ##For Plotting Regular Radar Reflectivity with the objects (right hand plot)
    ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
    dt_start = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
    dt_end = dt_start + timedelta(minutes=3)
    print(dt_start, dt_end)

    #radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
    #            glob(os.path.join(radardir, '*.netcdf'))}
    #radfilelist = [radfiles[x] for x in radfiles if dt_start <= x < dt_end]

    #print(radfilelist)

    ###Get general information from the radar files
    #dataset_rad = Dataset(radfilelist[0], 'r')
    #nlat = len(dataset_rad.dimensions['Lat'])
    #nlon = len(dataset_rad.dimensions['Lon'])
    #NW_lat = getattr(dataset_rad, 'Latitude')
    #NW_lon = getattr(dataset_rad, 'Longitude')
    #dy = getattr(dataset_rad, 'LonGridSpacing')
    #dx = getattr(dataset_rad, 'LatGridSpacing')
    #radardataset = dataset_rad.variables['MergedReflectivityQCComposite'][:, :]
    #dataset_rad.close()

    ###Remap radar data to same domain as satellite data
    #radar_lons = np.zeros((nlat, nlon))
    #radar_lats = np.zeros((nlat, nlon))
    #for ii in range(nlat):
    #    radar_lats[ii, :] = NW_lat - ii * dy
    #for ii in range(nlon):
    #    radar_lons[:, ii] = NW_lon + ii * dx
    #radar_grid = pr.geometry.GridDefinition(lons=radar_lons, lats=radar_lats)

    #remapped_radar = pr.kd_tree.resample_nearest(radar_grid, radardataset, sat_grid,
    #                                             radius_of_influence=10000, fill_value=-999)
    ##

    ##For plotting Reflectivity at -10C with the objects (right hand plot)

    # Get -10C isotherm radar files within time frame of the datetime we are interested in.
    radfiles_10C = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(isodir, '*.netcdf'))}
    radfilelist_10C_orig = [radfiles_10C[x] for x in radfiles_10C if dt_start <= x <= dt_end]
    radfilelist_10C = np.sort(radfilelist_10C_orig)

    print(radfilelist_10C)
    # We need to check the isotherm data for reflectivity values greater than 35 dBz. If the threshold is met,
    # we want to collect this data.

    ###Get general information from the -10C isotherm files
    #dx_10C = 0.01
    #dy_10C = 0.01
    #NW_lat_10C = 55
    #NW_lon_10C = -130
    #grb = pg.open(radfilelist_10C[0])
    #msg = grb[1]
    #data_10C = msg.values
    #grb.close()
    #nlat_10C, nlon_10C = np.shape(data_10C)
    #nfile_10C = len(radfilelist_10C)
    #radardataset_10C = np.zeros((nfile_10C, nlat_10C, nlon_10C))

    ###Get general information from the radar files
    dataset_iso = Dataset(radfilelist_10C[0], 'r')
    nlat_10C = len(dataset_iso.dimensions['Lat'])
    nlon_10C = len(dataset_iso.dimensions['Lon'])
    NW_lat_10C = getattr(dataset_iso, 'Latitude')
    NW_lon_10C = getattr(dataset_iso, 'Longitude')
    dy_10C = getattr(dataset_iso, 'LonGridSpacing')
    dx_10C = getattr(dataset_iso, 'LatGridSpacing')
    data_10C= dataset_iso.variables['Reflectivity_-10C'][:, :]
    dataset_iso.close()

    ###Remap -10C isotherm to same domain as satellite data
    radar_lons_10C = np.zeros((nlat_10C, nlon_10C))
    radar_lats_10C = np.zeros((nlat_10C, nlon_10C))
    for ii in range(nlat_10C):
        radar_lats_10C[ii, :] = NW_lat_10C - ii * dy_10C
    for ii in range(nlon_10C):
        radar_lons_10C[:, ii] = NW_lon_10C + ii * dx_10C
    radar_grid_10C = pr.geometry.GridDefinition(lons=radar_lons_10C, lats=radar_lats_10C)
    remapped_radar_10C = pr.kd_tree.resample_nearest(radar_grid_10C, data_10C, sat_grid,
                                                     radius_of_influence=10000, fill_value=-999)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection':geoproj}, figsize = (10,10))
    #fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': geoproj})

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_axes([0, 0, 1, 1], projection=geoproj)
    midx = xmin + (xmax - xmin)/2
    midy = ymin + (ymax - ymin)/2
    xend = xmax
    xbeg = midx + (xmax - xmin)/10
    ybeg = ymin + (ymax - ymin)/6
    yend = midy - (ymax - ymin)/8
    ax1.set_extent([xbeg, xend, midy, ymax], crs=geoproj)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(state_borders, edgecolor='tan')
    ax2.set_extent([xbeg, xend, midy, ymax], crs=geoproj)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(state_borders, edgecolor='tan')

    ###Remap to same domain as satellite data
    wdss_lons = np.zeros((nlat, nlon))
    wdss_lats = np.zeros((nlat, nlon))
    for ii in range(nlat):
        wdss_lats[ii, :] = NW_lat - ii * dy
    for ii in range(nlon):
        wdss_lons[:, ii] = NW_lon + ii * dx
    wdss_grid = pr.geometry.GridDefinition(lons=wdss_lons, lats=wdss_lats)

    remapped_wdss = pr.kd_tree.resample_nearest(wdss_grid, clusterID, sat_grid,
                                                 radius_of_influence=10000, fill_value=-999)

    maxclust = np.max(clusterID)
    minclust = np.min(clusterID)
    half = (maxclust + minclust)/2
    objimg = ax1.pcolormesh(xx1km,yy1km, remapped_wdss, vmin=minclust, vmax=maxclust, cmap='viridis', linewidth=0)
    #cbaxes = fig.add_axes([0.01,-0.05,0.27,0.03])  # left bottom width height
    cbar = plt.colorbar(objimg, orientation='horizontal', pad=0.01, extendfrac=0, drawedges=0, ax=ax1,
                        ticks=np.arange(minclust, maxclust, half))
    cbar.set_label('KMeans []', fontsize=10)
    cbar.draw_all()

    # plotting radar data instead- need to change variables and colormaps.
    # autumn_r
    #radimg = ax2.pcolormesh(xxv, yyv, ma.masked_less(zoom(remapped_radar, 2, order=0), 10), vmin=30, vmax=45,
    #                       cmap=plt.get_cmap('viridis'), linewidth=0)
    radimg = ax2.pcolormesh(xxv, yyv, ma.masked_less(zoom(remapped_radar_10C, 2, order=0), 10), vmin=25, vmax=40,
                           cmap=plt.get_cmap('viridis'), linewidth=0)
    #cbaxes2 = fig.add_axes([0.31, -0.05, 0.27, 0.03])  # left bottom width height
    cbar2 = plt.colorbar(radimg, orientation='horizontal', pad=0.01, extend='max', drawedges=0, ax=ax2,
                         ticks=np.arange(25, 45, 5))
    cbar2.set_label('Radar Reflectivity at -10C [dBz]', fontsize=10)
    cbar2.set_alpha(1.0)
    cbar2.draw_all()

    ax1.set_title((' ').join([timestamp, 'UTC']), fontsize=12)
    #fig.suptitle((' ').join([timestamp, 'UTC']), fontsize=16)

    #plt.savefig(os.path.join(outdir,('_').join(['ClusterID','scale0', timestamp+'.png'])),bbox_inches='tight')
    plt.savefig(os.path.join(isooutdir, ('_').join(['ClusterID', 'scale0', 'Ref-10C', timestamp + '.png'])), bbox_inches='tight')
    plt.close(fig)
    #sys.exit()