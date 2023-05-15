## Created by John Cintineo
import logging
from logging.config import dictConfig
from logging_def import *
from subprocess import Popen,PIPE,call
from datetime import datetime,timedelta
import time
import os,sys,re
import smtplib
import getpass
from datetime import datetime
from netCDF4 import Dataset
import errno
import numpy as np
#from pyhdf.SD import SD, SDC
#####################################################################################################
def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise
#####################################################################################################
def read_data(file, var_name, atts=None, global_atts=None,nogzip=False):
  FUNCTION_NAME = 'READ_DATA'
  logging.info(FUNCTION_NAME + ': Process started. Received ' + file)
  
  gzip=0
  if(file[-3:] == '.gz'):
    if(not(nogzip)): gzip=1
    unzipped_file = file[0:len(file)-3]
    the_file = unzipped_file
    s = call(['gunzip','-f',file])
  else:
    gzip=0
    the_file = file
  
  try:
    openFile = Dataset(the_file,'r')
  except (OSError,IOError) as err:
    logging.error(FUNCTION_NAME + ': ' + str(err))
    return -1

  data = openFile.variables[var_name][:]
  #get variable attributes
  if(isinstance(atts,dict)):
    for att in atts:
      if(att in openFile.variables[var_name].ncattrs()):
        atts[att] = getattr(openFile.variables[var_name],att) #atts is modified and returned
  #get global attributes
  if(isinstance(global_atts,dict)):
    for gatt in global_atts:
      if(gatt in openFile.ncattrs()):
        global_atts[gatt] = getattr(openFile,gatt) #global_atts is modified and returned
  openFile.close()
  
  #--gzip file after closing?
  if (gzip):
    s = Popen(['gzip','-f',unzipped_file]) #==will run in background since it doesn't need output from command.
    #s = call(['gzip','-f',unzipped_file])

  logging.info(FUNCTION_NAME + ': Got new ' + var_name + '. Process ended.')
  return data
#####################################################################################################
def write_netcdf(output_file,datasets,dims,atts={},gzip=False,wait=False,**kwargs):
  FUNCTION_NAME = 'WRITE_NETCDF'
  logging.info(FUNCTION_NAME + ': Process started')
  
  try:
    mkdir_p(os.path.dirname(output_file))
  except (OSError,IOError) as err:
    logging.error(FUNCTION_NAME + ': ' + str(err))
    return False
  else:
    ncfile = Dataset(output_file,'w',format='NETCDF3_CLASSIC') #can comment out the classic option
  #dimensions
  for dim in dims:
    ncfile.createDimension(dim,dims[dim])
  #variables
  
  for varname in datasets:
    if(isinstance(datasets[varname]['data'],np.ndarray)):
      dtype = str((datasets[varname]['data']).dtype)
    elif(isinstance(datasets[varname]['data'],int) or isinstance(datasets[varname]['data'],np.int32) or isinstance(datasets[varname]['data'],np.int16)):
      dtype = 'i'
    elif(isinstance(datasets[varname]['data'],float) or isinstance(datasets[varname]['data'],np.float32) or isinstance(datasets[varname]['data'],np.float16)):
      dtype = 'f'   
    
    fill_value = datasets[varname]['atts']['_FillValue'] if('_FillValue' in datasets[varname]['atts']) else None
    chunksizes = datasets[varname]['chunksizes'] if('chunksizes' in datasets[varname]) else None
    dat = ncfile.createVariable(varname,dtype,datasets[varname]['dims'],fill_value=fill_value,chunksizes=chunksizes)
    dat[:] = datasets[varname]['data']
    #variable attributes
    if('atts' in datasets[varname]):
      for att in datasets[varname]['atts']:
        if(att != '_FillValue'): dat.__setattr__(att,datasets[varname]['atts'][att]) #_FillValue is made in 'createVariable'
  #global attributes
  for key in atts:
    ncfile.__setattr__(key,atts[key])
  ncfile.close()
  logging.info(FUNCTION_NAME + ': Wrote out ' + output_file)

  if(gzip):
    if(wait):
      s = call(['gzip','-f',output_file]) #==wait to finish
    else:
      s = Popen(['gzip','-f',output_file]) #==will run in background

  logging.info(FUNCTION_NAME + ': Process ended')
  return True
#####################################################################################################
#def read_hdf_sd(file,varnames,att_names=[],FillValue=-999,unscale=True):
#  FUNCTION_NAME = 'READ_HDF_SD'
#  logging.info(FUNCTION_NAME + ': Process started')

#  data = {}; atts = {}
#  if(not(isinstance(varnames,list))):
#    varnames = [varnames]
#  if(not(os.path.isfile(file))):
#    logging.error(FUNCTION_NAME + ': ' + file + ' does not exist')
#    return data,atts

#  hdf = SD(file, SDC.READ)
#
#  for var in varnames:
#   if(var in hdf.datasets().keys()):
#      sd = hdf.select(var)
#      data[var] = {}
#      data[var]['data'] = sd[:]
#      data[var]['atts'] = sd.attributes()
#      data[var]['isValid'] = True
#      sd.endaccess() #close dataset
#    else:
#      logging.warning(FUNCTION_NAME + ': ' + var + ' is not present in ' + file)
#      data[var] = {}
#      data[var]['data'] = -1
#      data[var]['atts'] = {}
#      data[var]['isValid'] = False
#
#
#    if("_FillValue" in data[var]['atts']):
#      bad_ind = (data[var]['data'] == data[var]['atts']['_FillValue'])
#    if('scale_factor' in data[var]['atts'] and 'add_offset' in data[var]['atts'] and unscale):
#      data[var]['data'] = data[var]['data'] * data[var]['atts']['scale_factor'] + data[var]['atts']['add_offset']
#    if("_FillValue" in data[var]['atts']):
#      data[var]['data'][bad_ind] = FillValue
#
#  for att in att_names:
#    if(att in hdf.attributes()):
#      atts[att] = hdf.attributes()[att]
#
#  hdf.end() #close hdf file
#
#  logging.info(FUNCTION_NAME + ': Process ended')
#
#  return data,atts
#####################################################################################################
def restart_w2alg_proc(group,proc_name,email=False):
  FUNCTION_NAME = 'RESTART_W2ALG_PROC'
  logging.info(FUNCTION_NAME + ': Process started')

  machine=os.uname()[1]

  logging.info(FUNCTION_NAME + ': executing `w2alg stop ' + group + ' ' + proc_name + '` on ' + machine)

  FNULL = open(os.devnull,'w') #write output to /dev/null

  s = Popen(['w2alg','stop',group,proc_name],stdout=FNULL,stderr=FNULL)
  
  time.sleep(2) #make sure proc_name completely shuts down

  logging.info(FUNCTION_NAME + ': executing `w2alg start ' + group + ' ' + proc_name + '` on ' + machine)

  s = Popen(['w2alg','start',group,proc_name],stdout=FNULL,stderr=FNULL)

  if(email):
    sub = 'ProbSevere processing notification on ' + machine
    msg = '`w2alg stop/start ' + group + ' ' + proc_name + '` has occurred on ' + machine \
          + '. This may be a routine restart because objectID# got too large, '\
          + 'or because there was too large a time gap between scans.'
    to = os.environ['ADDRESS']
    send_email(sub,msg,to)

  logging.info(FUNCTION_NAME + ': Process ended')

#####################################################################################################
def send_email(subject,message,to):
  FUNCTION_NAME = 'SEND_EMAIL'

  logging.info(FUNCTION_NAME + ': Process started')

  user = getpass.getuser()
  hostname = os.uname()[1]
  FROM = 'john.cintineo@ssec.wisc.edu'
  TO = to.split(',')

  now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
  theEmail = "Subject: %s\n\n%s\n%s" % (subject,message,now)

  s = smtplib.SMTP('localhost')
  s.sendmail(FROM,TO, theEmail)
  s.quit()

  logging.info(FUNCTION_NAME + ': Process ended')
#####################################################################################################
def read_gpd_file(gpd_file):
  FUNCTION_NAME = 'READ_GPD_FILE'
  logging.info(FUNCTION_NAME + ': Process started')
  gpd_info = {}

  try:
    f = open(gpd_file,'r')
  except (OSError,IOError) as err:
    logging.error(FUNCTION_NAME + ': ' + str(err))
    return gpd_info
  for line in f:
    parts = re.split(r'\s{2,}',line)
    if(parts[0] == 'Map Projection:'):
      gpd_info['projection'] = parts[1].rstrip('\n')
    elif(parts[0] == 'Map Reference Latitude:'):
      gpd_info['reflat'] = float(parts[1])
    elif(parts[0] == 'Map Reference Longitude:'):
      gpd_info['reflon'] = float(parts[1])
    elif(parts[0] == 'Map Scale:'):
      gpd_info['map_scale'] = float(parts[1])
    elif(parts[0] == 'Map Origin Latitude:'):
      gpd_info['NW_lat'] = float(parts[1])
    elif(parts[0] == 'Map Origin Longitude:'):
      gpd_info['NW_lon'] = float(parts[1])
    elif(parts[0] == 'Map Equatorial Radius:'):
      gpd_info['equatorial_radius'] = float(parts[1])
    elif(parts[0] == 'Grid Width:'):
      gpd_info['nlon'] = int(parts[1])
    elif(parts[0] == 'Grid Height:'):
      gpd_info['nlat'] = int(parts[1])
    elif(parts[0] == 'Grid Map Units per Cell:'):
      gpd_info['dy'] = gpd_info['dx'] = int(parts[1])/3600.

  logging.info(FUNCTION_NAME + ': Read ' + gpd_file + '. Process ended.')

  return gpd_info
#####################################################################################################
def get_area_definition(projection_file,return_latlons=False):
  from pyproj import Proj
  import pyresample as pr
  import numpy.ma as ma
 
  nc = Dataset(projection_file)
  gip = nc.variables['goes_imager_projection']
  x = nc.variables['x'][:]
  y = nc.variables['y'][:]

  p = Proj('+proj=geos +lon_0='+str(gip.longitude_of_projection_origin)+\
           ' +h='+str(gip.perspective_point_height)+' +x_0=0.0 +y_0=0.0 +a=' + \
           str(gip.semi_major_axis) + ' +b=' + str(gip.semi_minor_axis))
	     
  x *= gip.perspective_point_height
  y *= gip.perspective_point_height

  xx,yy = np.meshgrid(x,y)
  ny,nx = np.shape(xx)
  newlons, newlats = p(xx, yy, inverse=True)
  mlats = ma.masked_where((newlats < -90) | (newlats > 90),newlats)
  mlons = ma.masked_where((newlons < -180) | (newlons > 180),newlons)
  newx,newy = p(mlons,mlats)
  max_x = newx.max(); min_x = newx.min(); max_y = newy.max(); min_y = newy.min()
  half_x = (max_x - min_x) / newlons.shape[1] / 2.
  half_y = (max_y - min_y) / newlats.shape[0] / 2.
  extents = (min_x - half_x, min_y - half_y, max_x + half_x, max_y + half_y)

  newgrid = pr.geometry.AreaDefinition('geos','goes_conus','goes',\
                                     {'proj':'geos','h':gip.perspective_point_height,\
				      'lon_0':gip.longitude_of_projection_origin,\
                                      'a':gip.semi_major_axis,'b':gip.semi_minor_axis},nx,ny,extents)
  nc.close()

  if(return_latlons):
    return newgrid, ny, nx, newlons, newlats
  else:
    return newgrid, ny, nx
