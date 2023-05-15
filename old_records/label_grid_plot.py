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
isodir = '/apollo/grain/saved_ci_ltg_data/radar/2019/20190626/Reflectivity_-10C/00.50/'
labeldir = '/apollo/grain/saved_ci_ltg_data/LabeledGrids/1hour/2019/20190626/'
outdir = '/home/sbradshaw/ci_ltg/CopiedData_2/'

#netcdfs = glob(os.path.join(labeldir, '*.nc'))
#Just for plotting how it changes:
netcdfs = np.sort(glob(os.path.join(labeldir, '*.netcdf')))
print(netcdfs)

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
xmin = np.nanmin(x); xmax = np.nanmax(x); ymin = np.nanmin(y); ymax = np.nanmax(y)

sat_group = goesvis.groups['one_km']
sat_lons = sat_group.variables['longitude'][:]
sat_lats = sat_group.variables['latitude'][:]
sat_grid = pr.geometry.GridDefinition(lons=sat_lons, lats=sat_lats)

for i in range(len(netcdfs)):

    basefile = os.path.basename(netcdfs[i])
    parts = basefile.split('-')
    year_day = parts[0][13:]
    time = parts[1][0:6]

    timestamp = str(year_day) + '-' + str(time).zfill(4)
    print(timestamp)

    ##For Plotting Regular Radar Reflectivity with the objects (right hand plot)
    ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
    dt_start = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
    dt_end = dt_start + timedelta(minutes=3)
    print(dt_start, dt_end)

    ##For plotting Reflectivity at -10C with the objects

    # Get -10C isotherm radar files within time frame of the datetime we are interested in.
    radfiles_10C = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x
                    for x in glob(os.path.join(isodir, '*.netcdf'))}
    radfilelist_10C_orig = [radfiles_10C[x] for x in radfiles_10C if dt_start <= x <= dt_end]
    radfilelist_10C = np.sort(radfilelist_10C_orig)

    # We need to check the isotherm data for reflectivity values greater than 35 dBz. If the threshold is met,
    # we want to collect this data.

    ###Get general information from the -10C isotherm files
    dataset_10C = Dataset(radfilelist_10C[0], 'r')
    nlat_10C = len(dataset_10C.dimensions['Lat'])
    nlon_10C = len(dataset_10C.dimensions['Lon'])
    NW_lat_10C = getattr(dataset_10C, 'Latitude')
    NW_lon_10C = getattr(dataset_10C, 'Longitude')
    dy_10C = getattr(dataset_10C, 'LonGridSpacing')
    dx_10C = getattr(dataset_10C, 'LatGridSpacing')
    data_10C = dataset_10C.variables['Reflectivity_-10C'][:, :]
    dataset_10C.close()

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

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection':geoproj}, figsize = (15,15))

    #fig = plt.figure(figsize=(10, 10))
    #ax1 = fig.add_axes([0, 0, 1, 1], projection=geoproj)
    midx = xmin + (xmax - xmin)/2
    midy = ymin + (ymax - ymin)/2
    xend = xmax
    xbeg = midx + (xmax - xmin)/10
    ybeg = ymin + (ymax - ymin)/6
    yend = midy - (ymax - ymin)/8
    #Reflectivity at -10C
    ax1.set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(state_borders, edgecolor='tan')
    #Objects labeled and changing at each step
    ax2.set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(state_borders, edgecolor='tan')
    #Truth Data
    #ax3.set_extent([midx, xmax, midy, ymax], crs=geoproj)
    #ax3.add_feature(cfeature.BORDERS)
    #ax3.add_feature(state_borders, edgecolor='tan')

##Reflectivity at -10C
    radimg = ax1.pcolormesh(xxv, yyv, ma.masked_less(zoom(remapped_radar_10C, 2, order=0), 10), vmin=30, vmax=40,
                            cmap=plt.get_cmap('viridis'), linewidth=0)
    #cbaxes2 = fig.add_axes([0.31, -0.05, 0.27, 0.03])  # left bottom width height
    cbar = plt.colorbar(radimg, orientation='horizontal', pad=0.01, extend='both', drawedges=0, ax=ax1,
                        ticks=np.arange(30, 42, 2))
    cbar.set_label('Radar Reflectivity at -10C [dBz]', fontsize=10)
    cbar.set_alpha(1.0)
    cbar.draw_all()
    #ax2.set_title('Objects created with Merged Reflectivity', fontsize=12)
    ax1.set_title((' ').join([timestamp, 'UTC']), fontsize=12)

##Plot labeled plots as they change
    # Get labeled grids and resample to same domain as satellite data
    labeled_nc = Dataset(netcdfs[i], 'r')
    labeled_group = labeled_nc['Labeled_Grids']
    labeled_objs = labeled_group.variables['Labeled_Ref_-10C_grid'][:]
    nlat = len(labeled_group.dimensions['Lat'])
    nlon = len(labeled_group.dimensions['Lon'])
    NW_lat = 51
    NW_lon = -125
    dy = 0.01
    dx = 0.01

    ###Remap to same domain as satellite data
    labeled_lons = np.zeros((nlat, nlon))
    labeled_lats = np.zeros((nlat, nlon))
    for ii in range(nlat):
        labeled_lats[ii, :] = NW_lat - ii * dy
    for ii in range(nlon):
        labeled_lons[:, ii] = NW_lon + ii * dx
    grid_labels = pr.geometry.GridDefinition(lons=labeled_lons, lats=labeled_lats)
    remapped_labeled_objs_step = pr.kd_tree.resample_nearest(grid_labels, labeled_objs, sat_grid,
                                                             radius_of_influence=10000, fill_value=-999)

    remapped_labeled_objs_step[remapped_labeled_objs_step == 0] = 'nan'
    objimg_step = ax2.pcolormesh(xx1km, yy1km, remapped_labeled_objs_step, vmin=30, vmax=40, cmap='viridis', linewidth=0)
    # cbaxes = fig.add_axes([0.01,-0.05,0.27,0.03])  # left bottom width height
    cbar = plt.colorbar(objimg_step, orientation='horizontal', pad=0.01, extend='both', drawedges=0, ax=ax2,
                        ticks=np.arange(30, 42, 2))
    cbar.set_label('Label from Radar Reflectivity at -10C [dBZ]', fontsize=10)
    cbar.draw_all()

    # plotting radar data instead- need to change variables and colormaps.
    # autumn_r
    # radimg = ax2.pcolormesh(xxv, yyv, ma.masked_less(zoom(remapped_radar, 2, order=0), 10), vmin=30, vmax=45,
    #                       cmap=plt.get_cmap('viridis'), linewidth=0)

    ax2.set_title('Labeled Grid', fontsize=12)

##Plot truth data
    #Get labeled grids and resample to same domain as satellite data
    #truth = Dataset(netcdfs_truth[0], 'r')
    #truth_group = truth['Labeled_Grids']
    #truth_objs = truth_group.variables['Labeled_Ref_-10C_grid'][:]
    #nlat_truth = len(truth_group.dimensions['dim1'])
    #nlon_truth = len(truth_group.dimensions['dim2'])
    #NW_lat = 51
    #NW_lon = -125
    #dy = 0.01
    #dx = 0.01

    ###Remap -10C isotherm to same domain as satellite data
    #truth_lons = np.zeros((nlat_truth, nlon_truth))
    #truth_lats = np.zeros((nlat_truth, nlon_truth))
    #for ii in range(nlat):
    #    truth_lats[ii, :] = NW_lat - ii * dy
    #for ii in range(nlon):
    #    truth_lons[:, ii] = NW_lon + ii * dx
    #grid_labels = pr.geometry.GridDefinition(lons=truth_lons, lats=truth_lats)
    #remapped_truth_objs = pr.kd_tree.resample_nearest(grid_labels, truth_objs, sat_grid,
    #                                                    radius_of_influence=10000, fill_value=-999)

    #remapped_truth_objs[remapped_truth_objs==0] = 'nan'
    #truth_objimg = ax3.pcolormesh(xx1km,yy1km, remapped_truth_objs, vmin=30, vmax=40, cmap='viridis', linewidth=0)
    #cbaxes = fig.add_axes([0.01,-0.05,0.27,0.03])  # left bottom width height
    #cbar = plt.colorbar(truth_objimg, orientation='horizontal', pad=0.01, extend='both', drawedges=0, ax=ax3,
    #                    ticks=np.arange(30, 42, 2))
    #cbar.set_label('Labeled Truth Grid [dBZ]', fontsize=10)
    #cbar.draw_all()

    # plotting radar data instead- need to change variables and colormaps.
    # autumn_r
    #radimg = ax2.pcolormesh(xxv, yyv, ma.masked_less(zoom(remapped_radar, 2, order=0), 10), vmin=30, vmax=45,
    #                       cmap=plt.get_cmap('viridis'), linewidth=0)

    #ax3.set_title('Truth Data from Reflectivity -10C', fontsize=12)

    plt.savefig(os.path.join(outdir, ('_').join(['Objs', 'created', 'with', 'Ref-10C', timestamp + '.png'])), bbox_inches='tight')
    plt.close(fig)
    #sys.exit()