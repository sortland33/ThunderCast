#Author: John L. Cintineo Edits: Stephanie M. Ortland
#Converts Multi-Radar Multi-Sensor (MRMS) grib files to netcdfs.
import pygrib as pg
import numpy as np
import sys,os
from subprocess import Popen,PIPE,call
import shutil
import pyresample as pr
import argparse
from datetime import datetime
from collections import OrderedDict
from utils import *
import time
import logging
from logging.config import dictConfig
from logging_def import *
from constants_config import constants_config
if(sys.version_info[0] < 3):
  range = xrange
t0 = time.time()
####################################################################################################################################################
def get_units(varname):
  units={'MergedReflectivityQCComposite':'dBZ',\
         'Reflectivity_-10C':'dBZ',\
         'Reflectivity_-20C':'dBZ',\
         'MergedAzShear_0-2kmAGL':'0.001/s',\
         'MergedAzShear_3-6kmAGL':'0.001/s',\
         'MESH':'mm',\
         'POSH':'%',\
         'VII':'g/m2',\
         'VIL_Density':'g/m3',\
         'VIL':'g/m2',\
         'SHI':'dimensionless',\
         'EchoTop_30':'km',\
         'EchoTop_50':'km',\
         'H50_Above_0C':'km',\
         'H50_Above_-20C':'km',\
         'H60_Above_0C':'km',\
         'H60_Above_-20C':'km'}
  if(varname in units):
    return units[varname]
  else:
    return 'N/A'
####################################################################################################################################################
def MRMS_grib2netcdf():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f","--GRIB_FILE",help="Full path and filename to MRMS grib2 file",type=str,nargs=1)
  parser.add_argument("-o","--OUTROOT",help="Root output directory. Default = ${PROBSEV_DATA}/radar/processed/ if realtime, else $PWD",type=str,nargs=1)
  parser.add_argument("-r","--realtime",help="Run in realtime mode. This means it will move the file to a specified directory first.",action="store_true")
  parser.add_argument("-l","--logfile",help="Assumes operational run, and will add to logfile",type=str,nargs=1)
  args = parser.parse_args()
  if(not args.OUTROOT): #get defaults
    #if(args.realtime):
    #  outdir = probsev_data + '/radar/raw/'
    #else:
    outdir = os.environ['PWD']
  else:
    outdir = args.OUTROOT[0]
  if(not args.logfile):
    logfile = None
  else:
    logfile = args.logfile[0]
  main(args.GRIB_FILE[0],outdir=outdir,realtime=args.realtime,logfile=logfile)
####################################################################################################################################################
def main(grib_file,outdir=os.environ['PWD'],realtime=False,logfile=None):
  FUNCTION_NAME = 'MRMS_GRIB2NETCDF'
  #define the logger
  logDict = logging_def(logfile = logfile)
  dictConfig(logDict)
  #send stderr to logger
  stderr_logger = logging.getLogger('STDERR')
  sl = StreamToLogger(stderr_logger, logging.ERROR)
  sys.stderr = sl
  #load constants
  constants = constants_config()
  MissingVal = -999.
  if(outdir[-1] != '/'): outdir += '/'
  #probsev_data= os.environ['PROBSEV_DATA']
  pathname = os.path.dirname(grib_file)
  filename = os.path.basename(grib_file)
  #move file from LDM incoming directory if realtime
  if(realtime):
    subdir = constants['MRMS']['gribSubdir'] #usu. '00.50'
    tmp_parts = filename.split('_'+subdir+'_')
    tmpdir = ('/').join([probsev_data,'radar','raw','tmp',tmp_parts[0]+'_'+subdir+'_'+tmp_parts[1][0:13]])
    try:
      os.makedirs(tmpdir)
    except (OSError,IOError) as err:
      logging.warning(FUNCTION_NAME + ': ' + tmpdir + ' exists. File already processing. Exiting.')
      try:
        time.sleep(2) #sleep for a couple of seconds just so we don't accidentally delete a grib that another process is moving
        os.remove(grib_file)
      except (OSError,IOError) as err:
        logging.error(FUNCTION_NAME + ": Couldn't remove " + grib_file)
      sys.exit()
    newpath = probsev_data + '/radar/raw/grib2/'
    shutil.move(grib_file,newpath)
    grib_file = newpath + filename
  #gunzip if necessary
  if(grib_file[-3:] == '.gz'):
    s = call(['gunzip','-f',grib_file])
    grib2 = grib_file[:-3]
  else:
    grib2 = grib_file
  #get file info
  subdir = constants['MRMS']['gribSubdir'] #usu. '00.50'
  parts = os.path.basename(grib2).split('_'+subdir+'_')
  varname = parts[0][5:] #skip 'MRMS_'
  timestamp = parts[1][0:15]
  #if(realtime and varname != 'MergedReflectivityQCComposite' and varname != 'MESH' and varname != 'VIL_Density' and 'AzShear' not in varname):
  #  logging.warning(FUNCTION_NAME + ': ' + varname + ' not in list to process. Exiting.')
  #  sys.exit()
  #now read grib2
  grb = pg.open(grib2)
  msg = grb[1]  #should only be 1 'message' in grib2 file
  data = msg.values
  #make file lats/lons. Could pull from grib, e.g.: lats = grb[1]['latitudes'] ; but making them is faster than pulling from grib.
  grb.close()
  orig_nlat,orig_nlon = np.shape(data)
  origNWlat = constants['MRMS']['origNWlat'] #usu. 55.0
  origNWlon = constants['MRMS']['origNWlon'] #usu. -130.0
  if('AzShear' in varname):
    dx = dy = 0.005
  else:
    dx = dy = 0.01
  new_NW_lat = 51
  new_NW_lon = -125
  new_nlat = 2700
  new_nlon = 5900
  NW_lat_slice = int((origNWlat - new_NW_lat) / dy)
  NW_lon_slice = int((-origNWlon + new_NW_lon) / dx)
  data_test = data[:-NW_lat_slice, NW_lon_slice:]
  data_lat_shape, data_lon_shape = data_test.shape
  SE_lat_slice = data_lat_shape - new_nlat
  SE_lon_slice = data_lon_shape - new_nlon
  data_crop = data[SE_lat_slice:-NW_lat_slice,NW_lon_slice:-SE_lon_slice]
  ref10C_data = data_crop.astype(np.float32)
  nlat_crop, nlon_crop = data_crop.shape
  #write out netCDF
  dims = OrderedDict(); dims['Lat'] = nlat_crop; dims['Lon'] = nlon_crop
  datasets = {varname:{'data':ref10C_data,'dims':('Lat','Lon'),'atts':{'units':get_units(varname)}}}
  epoch_secs = int((datetime.strptime(timestamp,'%Y%m%d-%H%M%S') - datetime(1970,1,1)).total_seconds())
  atts = OrderedDict()
  atts['TypeName'] = varname
  atts['DataType'] = 'LatLonGrid'
  atts['Latitude'] = new_NW_lat
  atts['Longitude'] = new_NW_lon
  atts['LatGridSpacing'] = dy
  atts['LonGridSpacing'] = dx
  atts['Height'] = 500.
  atts['FractionalTime'] = 0.
  atts['Time'] = epoch_secs
  atts['attributes'] = ' SubType'
  atts['SubType-unit'] = 'dimensionless'
  atts['SubType-value'] = subdir
  atts['MissingData'] = MissingVal
  outfile = ('/').join([outdir,varname,subdir,timestamp+'.netcdf'])
  write_netcdf(outfile,datasets,dims,atts=atts,gzip=False)
  #if(realtime):
  #  fml_file = probsev_data + '/radar/raw/' + timestamp + '_' + varname + '_' + subdir + '.fml'
  #  fml = open(fml_file,'w')
  #  fml.write('<item>\n')
  #  fml.write((' ').join([' <time','fractional="0.000000">', str(epoch_secs),'</time>','\n']))
  #  fml.write((' ').join([' <params>netcdf','{indexlocation}',varname,subdir,timestamp+'.netcdf','</params>','\n']))
  #  fml.write((' ').join([' <selections>'+timestamp,varname,subdir,'</selections>','\n']))
  #  fml.write('</item>')
  #  fml.close()
  #  fml_dir = probsev_data + '/radar/raw/code_index.fam/'
  #  logging.info(FUNCTION_NAME + ': Wrote ' + fml_file)
  #  logging.info(FUNCTION_NAME + ': Moving to ' + fml_dir)
  #  try:
  #    shutil.move(fml_file,fml_dir)
  #  except (OSError,IOError) as err:
  #    logging.error(FUNCTION_NAME + ': ' + str(err))
  logging.info(FUNCTION_NAME + ': ' + (', ').join([varname,str(time.time()-t0),timestamp]))
if __name__ == "__main__":
  MRMS_grib2netcdf()