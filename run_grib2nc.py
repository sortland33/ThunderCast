#Author: Stephanie M. Ortland
from MRMS_grib2netcdf import main
from glob import glob
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
import utils
import shutil

PWD = os.environ['PWD'] + '/'

t0 = time.time()
##### Desired dates to convert gribs to netcdfs. Modify as needed
dates = [datetime(2021, 7, 1), datetime(2021, 7, 2), datetime(2021, 7, 3),
         datetime(2021, 7, 4), datetime(2021, 7, 5), datetime(2021, 7, 6),
         datetime(2021, 7, 7), datetime(2021, 7, 8), datetime(2021, 7, 9), datetime(2021, 7, 10),
         datetime(2021, 7, 11), datetime(2021, 7, 12), datetime(2021, 7, 13),
         datetime(2021, 7, 14), datetime(2021, 7, 15), datetime(2021, 7, 16), datetime(2021, 7, 17), datetime(2021, 7, 18),
         datetime(2021, 7, 19), datetime(2021, 7, 20), datetime(2021, 7, 21), datetime(2021, 7, 22), datetime(2021, 7, 23),
         datetime(2021, 7, 24), datetime(2021, 7, 25), datetime(2021, 7, 26), datetime(2021, 7, 27), datetime(2021, 7, 28),
         datetime(2021, 7, 29), datetime(2021, 7, 30), datetime(2021, 7, 31)]
for n in range(len(dates)):
    # Setting directory information
    print('Converting Reflectivity at -10C grib files to netcdf for ', dates[n])
    strdate = str(dates[n])
    topdt = datetime.strptime(strdate, '%Y-%m-%d %H:%M:%S')
    prevdt = topdt - timedelta(minutes = 60)
    rootradardir = '/apollo/grain/saved_probsevere_data/radar' #root directory with radar data as gribs- modify this as needed.
    radardir_10C = topdt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/grib2/Reflectivity_-10C/')) #directory where the -10degC data is specifically- modify as needed
    fileloc = os.path.join(PWD, 'CopiedGribs/') #local location where copied files will be stored
    utils.mkdir_p(fileloc)

    #Copy grib files before converting to avoid accidental modification of original file.
    print('Copying radar isotherm gribs...')
    iso_gribs = np.sort(glob(os.path.join(radardir_10C, '*.grib2')))
    for i in range(len(iso_gribs)):
        try:
            shutil.copy(iso_gribs[i], fileloc)
        except (PermissionError, OSError, IOError) as err:
            print(traceback.format_exc())
            print('Copy did not work...')
            continue
    t_copy = time.time() - t0

    ##### Convert Reflectivity -10C files from grib2 to netcdf
    outdir = topdt.strftime('/apollo/grain/saved_ci_ltg_data/radar/%Y/%Y%m%d') #desired output directory- modify as needed
    utils.mkdir_p(outdir)
    iso_gribs = np.sort(glob(os.path.join(fileloc, '*.grib2')))
    for i in range(len(iso_gribs)):
        main(iso_gribs[i], outdir)
    print(time.time()-t_copy)

    print('Removing files from local folder...')
    # Removes copied data to avoid clutter and taking up space
    if os.path.exists(fileloc) and os.path.isdir(fileloc):
        shutil.rmtree(fileloc)