import numpy as np
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SATPY_CONFIG_PATH'] = '/home/sbradshaw/ci_ltg/PPP_config/'

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

import json
import zarr
import sys
from netCDF4 import Dataset
from datetime import datetime, timedelta
#sys.path.insert(0,'static/')
from ABIcmaps import irw_ctc
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import ams_utils
from scipy.ndimage import zoom
import numpy.ma as ma
import innvestigate
from matplotlib.lines import Line2D
import argparse
import collect_data
import shutil
import utils
from solar_angles import solar_angles
from sat_zen_angle import sat_zen_angle
import pickle
from earth_to_fgf import earth_to_fgf
import collections
import keras_metrics
#os.environ['SATPY_CONFIG_PATH'] = f"/home/jcintineo/probsevere_oconus/src/satpy_plotting/SATPY_CONFIG/"
import satpy
from satpy import Scene
from satpy.writers import get_enhanced_image

PWD = os.environ['PWD']
local_dir = 'CopiedData_LRPtest/'
fileloc = os.path.join(PWD, local_dir)
utils.mkdir_p(fileloc)


#######################################################################################
def copysatfiles(abi_files, fileloc):
    abi_locals = []
    for f in abi_files:
        base = os.path.basename(f)
        local = os.path.join(fileloc, base)
        abi_locals.append(local)

        shutil.copy(f, local)

    return abi_locals


#######################################################################################

def plot_lrp_sal(model, model2, abidir=None, glmdir=None, detail=False, rgb_relevance=False,
                 mc=None, mlats=[], mlons=[], dts=[], ind=[], prefix='tmp', outdir='/apollo/grain/saved_ci_ltg_data/LRP'):
    #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlusFast(model, neuron_selection_mode='index')
    #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlus(model, neuron_selection_mode='index')
    #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ(model, neuron_selection_mode='index')
    #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1IgnoreBias(model, neuron_selection_mode='index')
    analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1(model, neuron_selection_mode='index')
    #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0(model, neuron_selection_mode='index')
    #analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model, neuron_selection_mode='index')

    # npix per side of patch
    pl = 200
    #pl = 100
    #zoomed
    sStart = int(pl / 2 - 10)
    sEnd = int(pl / 2 + 10)
    #print('sStart:', sStart)
    #print('sEnd:', sEnd)
    #Non-Zoomed:
    #sStart = int(pl / 2 - 20)
    #sEnd = int(pl / 2 + 20)

    #middlepixelind = int(pl * pl / 2 + pl / 2)

    chans = collections.OrderedDict()
    if (mc):
        # reducefac =  2 ; print(reducefac,sStart,sEnd) #int(mc['input_tuples'][-1][0] / pl); print(reducefac,sStart,sEnd)
        svals = mc['scaling_values']
        scalars = mc['scalar_vars'];
        nscalars = len(scalars)

        dict_keys = mc['inputs'].keys()

        #Need to change from a dictionary to a list of lists
        inputs = []
        for i in dict_keys:
            inputs.append(mc['inputs'][i]) #May need to change this if add a channel with same resolution as C02, C05, C13
        print(inputs)

        npatch_inputs = len(inputs) - 1 if (nscalars > 0) else len(inputs)
        #print(mc['inputs'])
        for ii, inp in enumerate(inputs):
            print(inp)
            for c in inp:
                print(c)
                # index = int(inp[-1])-1 #e.g., input1
                chans[c] = {}
                chans[c]['mean'] = svals['mean'][c] #This is flipped in John's code, but needs to be this way here.
                chans[c]['std'] = svals['stddev'][c]
                if (c == 'C02'):
                    chans[c]['hs'] = int(pl * 2)
                elif (c == 'C05'):
                    chans[c]['hs'] = int(pl)
                else:
                    chans[c]['hs'] = int(pl / 2)
                # chans[c]['hs'] = int(mc['input_tuples'][ii][0]/2/reducefac)

        print(chans)

    slice05km = slice(sStart * 4, sEnd * 4)
    slice1km = slice(sStart * 2, sEnd * 2)
    slice2km = slice(sStart, sEnd)
    #zoomed
    irticks = [10, 20]
    #non-zoomed
    #irticks = [10, 20, 30]
    vis1kmticks = [tt * 2 for tt in irticks]
    visticks = [tt * 4 for tt in irticks]

    #glmvars = []
    for c in chans:
        if (c == 'C02'):
            chans[c]['vmin'] = 0;
            chans[c]['vmax'] = 1;
            chans[c]['vinc'] = 0.5;
            chans[c]['cmap'] = plt.get_cmap('Greys_r')
            chans[c]['cbar_data_lab'] = 'ABI CH02 reflectance []'
            chans[c]['cbar_lab'] = 'ABI CH02 relevance [x 10$^{-2}$]'
            chans[c]['imax'] = 0.1;#0.03;
            chans[c]['imin'] = -0.1;#-0.03;
            chans[c]['iinc'] = 0.05
            chans[c]['irange'] = np.arange(chans[c]['imin'], chans[c]['imax'] + chans[c]['iinc'], chans[c]['iinc'])
            chans[c]['ilabs'] = []
            chans[c]['slice'] = slice05km
            chans[c]['ticks'] = visticks
            for i in chans[c]['irange']:
                i = i * 100
                print('i', i)
                l = str(int(i)) if (i % 1 == 0) else "{:.1f}".format(i)
                print('l', l)
                chans[c]['ilabs'].append(l)
        if (c == 'C05'):
            chans[c]['vmin'] = 0;
            chans[c]['vmax'] = 0.5;
            chans[c]['vinc'] = 0.25;
            chans[c]['cmap'] = plt.get_cmap('Greys_r')
            chans[c]['cbar_data_lab'] = 'ABI CH05 reflectance []'
            chans[c]['cbar_lab'] = 'ABI CH05 relevance [x 10$^{-2}$]'
            chans[c]['imax'] = 0.1#0.05;
            chans[c]['imin'] = -0.1#0.05;
            chans[c]['iinc'] = 0.05#0.25
            chans[c]['irange'] = np.arange(chans[c]['imin'], chans[c]['imax'] + chans[c]['iinc'], chans[c]['iinc'])
            chans[c]['ilabs'] = []
            chans[c]['slice'] = slice1km
            chans[c]['ticks'] = vis1kmticks
            for i in chans[c]['irange']:
                i = i * 100
                l = str(int(i)) if (i % 1 == 0) else "{:.1f}".format(i)
                chans[c]['ilabs'].append(l)
        if (c == 'C13'):
            chans[c]['vmin'] = 180;
            chans[c]['vmax'] = 310;
            chans[c]['vinc'] = 30;
            chans[c]['cmap'] = irw_ctc()
            chans[c]['cbar_data_lab'] = 'ABI CH13 BT [K]'
            chans[c]['cbar_lab'] = 'ABI CH13 relevance [x 10$^{-2}$]'
            chans[c]['imax'] = 0.05#0.03;
            chans[c]['imin'] = -0.05#-0.03;
            chans[c]['iinc'] = 0.025
            chans[c]['irange'] = np.arange(chans[c]['imin'], chans[c]['imax'] + chans[c]['iinc'], chans[c]['iinc'])
            chans[c]['ilabs'] = []
            chans[c]['slice'] = slice2km
            chans[c]['ticks'] = irticks
            for i in chans[c]['irange']:
                i = i * 100
                l = str(int(i)) if (i % 1 == 0) else "{:.1f}".format(i)
                chans[c]['ilabs'].append(l)

    abi_channels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']

    nsamples = len(mlats)
    # get abi row, col
    row2km, col2km = earth_to_fgf(mlats, mlons, sub_lon_grid=-75.)  # assumes GOES-East
    row2km = [int(np.round(rr)) for rr in row2km];
    col2km = [int(np.round(cc)) for cc in col2km]
    row1km = [rr * 2 for rr in row2km];
    col1km = [cc * 2 for cc in col2km]
    row05km = [rr * 4 for rr in row2km];
    col05km = [cc * 4 for cc in col2km]

    # solzen and satzen
    datetimes = [datetime.strptime(dd, '%Y%m%d-%H%M') for dd in dts]
    timestamps = [dd + 'Z' for dd in dts]
    solzen, solaz = solar_angles(dt=datetimes, lat=mlats, lon=mlons)
    satzen = sat_zen_angle(xlat=mlats, xlon=mlons, satlat=0, satlon=-75.0)

    # empty data frames
    for c in chans:
        hs = chans[c]['hs']
        chans[c]['data'] = np.zeros((nsamples, hs * 2, hs * 2, 1))

    lastdt = datetime.strptime(dts[0], '%Y%m%d-%H%M')
    rgbs = []
    for ii in range(nsamples):
        abi_files = [];
        dt = datetime.strptime(dts[ii], '%Y%m%d-%H%M')

        for c in chans:
            print(c[1:])
            if (c[1:] in abi_channels):  # then it's an ABI file
                abi_files.append(glob.glob(dt.strftime(os.path.join(abidir, '*C' + c[1:] + '*_s%Y%j%H%M*')))[0])


        abi_locals = copysatfiles(abi_files, fileloc)
        print(abi_locals)

        # get satpy scene
        scn = satpy.Scene(reader='abi_l1b', filenames=abi_locals)
        scn.load(['C13', 'C15', 'C11', 'C14', 'C02', 'C05'], calibrations=['brightness_temperature',
                                                                           'brightness_temperature',
                                                                           'brightness_temperature',
                                                                           'brightness_temperature',
                                                                           'reflectance',
                                                                           'reflectance'])
        projx = scn['C05'].x.compute().data
        projy = scn['C05'].y.compute().data
        hs = chans['C05']['hs']
        print('hs', hs)

        composite = 'DayCloudPhaseFC'  # DayCloudPhaseFC
        center_lat = lats[ii]
        center_lon = lons[ii]

        try:
            x_idx, y_idx = scn['C05'].attrs['area'].get_xy_from_lonlat(center_lon, center_lat)
            print('middle index x, y:', x_idx, y_idx)
        except KeyError:
            exception_files.append(dt_start)
            continue

        xmin_index = x_idx - hs
        xmax_index = x_idx + hs
        ymax_index = y_idx - hs
        ymin_index = y_idx + hs
        #x_coords, y_coords = scn['C05'].attrs['area'].get_proj_vectors()
        xmin = projx[xmin_index] #x_coords[xmin_index]
        xmax = projx[xmax_index] #x_coords[xmax_index]
        ymin = projy[ymin_index] #y_coords[ymin_index]
        ymax = projy[ymax_index] #y_coords[ymax_index]

        crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
        print(crop_scn)
        newscn = crop_scn
        newscn.load([composite])  # ;scn.load(['IRCloudPhaseFC'])
        resamp = newscn.resample(resampler='native')
        rgbs.append(np.transpose(get_enhanced_image(resamp[composite]).data.values, axes=(1, 2, 0)))
        print('rgb_shape',get_enhanced_image(resamp[composite]).data.values.shape)
        # local_files += abi_locals
       # for loc in abi_locals:
       #     nc = Dataset(loc, 'r')
       #     cc = os.path.basename(loc).split('_G')[0][-2:]
       #     if ('C01' in loc or 'C02' in loc or 'C03' in loc or 'C04' in loc or 'C05' in loc):
       #         chans['C' + cc]['grid'] = nc.variables['Rad'] * nc.variables['kappa0'][:]
       #     else:
       #         chans['C' + cc]['grid'] = (nc.variables['planck_fk2'][:] / (np.log((nc.variables['planck_fk1'][:] / nc.variables['Rad']) + 1)) - nc.variables['planck_bc1'][:]) / nc.variables['planck_bc2'][:]
       #     nc.close()

        for c in chans:
            hs = chans[c]['hs']
            #if (c == 'C02'):
            #    row = row05km[ii];
            #    col = col05km[ii]

            #elif (c == 'C01' or c == 'C03' or c == 'C05'):
            #    row = row1km[ii];
            #    col = col1km[ii]
            #else:
            #    row = row2km[ii];
            #    col = col2km[ii]
            #chans[c]['data'][ii, ..., 0] = chans[c]['grid'][row - hs:row + hs, col - hs:col + hs]
            print(c)
            print(chans[c]['data'].shape)
            print(crop_scn[c].values.shape)
            chans[c]['data'][ii, ..., 0] = crop_scn[c].values[0:hs*2,0:hs*2]


    # for localf in local_files:
    #   try: #clean up
    #     os.remove(localf)
    #   except (NameError,OSError):
    #     pass

    for c in chans:
        CHnorm_vals = (chans[c]['data'] - chans[c]['mean']) / chans[c]['std']
        CHnorm = np.zeros((chans[c]['data'].shape[0], CHnorm_vals.shape[1], CHnorm_vals.shape[2], 1))
        CHnorm[:] = CHnorm_vals
        chans[c]['norm'] = CHnorm
        print(chans[c]['norm'].shape)

    input_list = []
    for inp in inputs:
        tensor = np.array([])
        for c in inp:
            if (tensor.size == 0):
                tensor = chans[c]['norm']
                print('Tensor shape CH', c, tensor.shape)
            else:
                tensor = np.concatenate((tensor, chans[c]['norm']), axis=-1)
                print('CH', c, tensor.shape)
        input_list.append(tensor)

    preds_nf = model2.predict(input_list, batch_size=1)
    preds_nf = preds_nf[0]
    leny = preds_nf.shape[1]

    preds = model.predict(input_list, batch_size=1)
    # Get new index for pixel of interest (center lat and lon)
    cropx_idx, cropy_idx = crop_scn['C05'].attrs['area'].get_xy_from_lonlat(center_lon, center_lat)
    middlepixelind = preds_nf.shape[1]*cropx_idx + cropy_idx
    print('cropped middle index', cropx_idx, cropy_idx)
    print('middle pixel index flattened', middlepixelind)

    lrp = analyzer.analyze(input_list, neuron_selection=middlepixelind)

    print(lrp[0].shape, lrp[1].shape, lrp[2].shape)
    print(np.min(lrp[0]), np.max(lrp[0]))
    print(np.min(lrp[1]), np.max(lrp[1]))
    print(np.min(lrp[2]), np.max(lrp[2]))
    # print(np.sum(lrp[0][0]),np.sum(lrp[1][0]),np.sum(lrp[2][0,...,0]),np.sum(lrp[2][0,...,1]))
    # print(np.sum(lrp[0][1]),np.sum(lrp[1][1]),np.sum(lrp[2][1,...,0]),np.sum(lrp[2][1,...,1]))

    # print(np.sum(lrp[0][0])+np.sum(lrp[1][0])+np.sum(lrp[2][0]))
    # print(np.sum(lrp[0][1])+np.sum(lrp[1][1])+np.sum(lrp[2][1]))

    # saliency_matrix = ams_utils.get_saliency_for_class(cnn_model_object=model, target_class=1,
    #                  list_of_input_matrices=input_list)

    cellsize = 3;
    glw = 0.5;
    ls = '--'
    chanlist = list(chans.keys())
    if (detail):  # make image of each input alongside relevance
        # assert (1 <= nsamples <= 2)
        #
        # wspace = hspace = 0.5
        # nrows = nsamples * 2
        # ncols = len(chans)
        #
        # fig = plt.figure(num=None, figsize=(ncols * cellsize, nrows * cellsize), dpi=300, facecolor='w', edgecolor='k')
        # gs = gridspec.GridSpec(ncols=ncols * cellsize, nrows=nrows * cellsize, figure=fig, hspace=hspace)
        #
        # for row in range(nrows):
        #     if (row % 2 == 0):  # plot images of data
        #         for col in range(ncols):
        #             ch = chanlist[col]
        #             print('ch', ch)
        #             ax = plt.subplot(gs[row * cellsize:(row + 1) * cellsize, col * cellsize:(col + 1) * cellsize])
        #             ax.set_yticklabels([]);
        #             ax.set_xticklabels([])
        #             ax.set_xticks([]);
        #             ax.set_yticks([])
        #
        #             dat = chans[ch]['data'][row, ..., 0][chans[ch]['slice'], chans[ch]['slice']]
        #             dat = np.flipud(dat)
        #             pcimg = ax.pcolormesh(dat, vmin=chans[ch]['vmin'], vmax=chans[ch]['vmax'], cmap=chans[ch]['cmap'])
        #             cbar = plt.colorbar(pcimg, orientation='horizontal', pad=0.01, shrink=0.9, extendfrac=0,
        #                                 drawedges=0, ax=ax,
        #                                 ticks=np.arange(chans[ch]['vmin'], chans[ch]['vmax'] + chans[ch]['vinc'],
        #                                                 chans[ch]['vinc']))
        #             cbar.set_label(chans[ch]['cbar_data_lab'], fontsize=10)
        #             cbar.ax.tick_params(labelsize=10)
        #
        #             # gray gridlines
        #             ax.set_yticks(chans[ch]['ticks'])
        #             ax.set_xticks(chans[ch]['ticks'])
        #             ax.grid(True, linestyle=ls, linewidth=glw)
        #             ax.set_xticklabels([]);
        #             ax.set_yticklabels([]);
        #             ax.tick_params(length=0.0)
        #             # center point, denoting pixel of interest
        #             ax.plot(chans[ch]['ticks'][1], chans[ch]['ticks'][1], marker='x', color='black', markersize=10)
        #
        #             if (col == 0):
        #                 ax.set_ylabel(
        #                     'Lat,Lon: ' + '{:.2f}'.format(mlats[row]) + ', ' + '{:.2f}'.format(mlons[row]) + '  ' +
        #                     timestamps[row], fontsize=7)
        #                 # label = str(int(preds[row][middlepixelind]*100))+'%'
        #                 # ax.annotate(label, (0.81, 0.05), xycoords='axes fraction', va='center', fontsize=14, color='white')
        #
        #     else:  # plot relevance
        #         for col in range(ncols):
        #             print('col', col)
        #             ch = chanlist[col]
        #             ax = plt.subplot(gs[row * cellsize:(row + 1) * cellsize, col * cellsize:(col + 1) * cellsize])
        #             ax.set_yticklabels([]);
        #             ax.set_xticklabels([])
        #             ax.set_xticks([]);
        #             ax.set_yticks([])
        #
        #             # find the right input index and channel
        #             cter = 0
        #             for inp in inputs:
        #                 if(ch in inp):
        #                     iidx = cter
        #                     # find the right channel index
        #                     cidx = inp.index(ch)
        #                     break
        #                 cter += 1
        #
        #             # find the right input index and channel
        #
        #             #cter = 0
        #             #print('inputs', inputs)
        #             #print('ch', ch)
        #             #for inp in inputs:
        #             #    print('inp', inp)
        #             #    if(ch in inp):
        #
        #             #        iidx = cter
        #             #        print('iidx', iidx)
        #                     # find the right channel index
        #             #        cidx = inp.index(ch)
        #             #        print('cidx', cidx)
        #             #        break
        #             #    cter += 1
        #             for ii, inp in enumerate(inputs):
        #                 if(ch in inp):
        #                     iidx = ii
        #                     # find the right channel index
        #                     cidx = inp.index(ch)
        #                     break
        #
        #             print('cidx', cidx)
        #             print(lrp[iidx].shape)
        #             dat = np.flipud(lrp[iidx][row - (row // 2 + 1), ..., cidx])
        #             pcimage = ax.pcolormesh(dat[chans[ch]['slice'], chans[ch]['slice']], vmin=chans[ch]['imin'],
        #                                     vmax=chans[ch]['imax'], cmap=plt.get_cmap('Reds'))
        #
        #             # gray gridlines
        #             ax.set_yticks(chans[ch]['ticks'])
        #             ax.set_xticks(chans[ch]['ticks'])
        #             ax.grid(True, linestyle=ls, linewidth=glw)
        #             ax.set_xticklabels([]);
        #             ax.set_yticklabels([]);
        #             ax.tick_params(length=0.0)
        #             # center point, denoting pixel of interest
        #             ax.plot(chans[ch]['ticks'][1], chans[ch]['ticks'][1], marker='x', color='black', markersize=10)
        #
        #             cbar = plt.colorbar(pcimage, orientation='horizontal', pad=0.01, shrink=0.9, extendfrac=0,
        #                                 drawedges=0, ticks=chans[ch]['irange'], ax=ax)
        #             cbar.ax.set_xticklabels(chans[ch]['ilabs'])
        #             cbar.set_label(chans[ch]['cbar_lab'], fontsize=10)
        #             cbar.ax.tick_params(labelsize=10)

                    # if (col == 0):
                    #     label = str(int(preds[row - (row // 2 + 1)][middlepixelind] * 100)) + '%'
                    #     ax.annotate(label, (0.81, 0.05), xycoords='axes fraction', va='center', fontsize=14,
                    #                 color='black')
        print('detail is not working yet')

    elif (rgb_relevance):
        nrows = nsamples
        ncols = 2  # 1 image rgb and 1 relevance rgb

        fig = plt.figure(num=None, figsize=(ncols * cellsize, nrows * cellsize), dpi=300, facecolor='w',
                         edgecolor='k')  # , constrained_layout=True)
        gs = gridspec.GridSpec(ncols=ncols * cellsize, nrows=nrows * cellsize,
                               figure=fig)  # the +1 is basically to fit the legends at the botto

        sample_cter = 0

        for row in range(nrows):
            for col in range(ncols):
                # new axis required
                ax = plt.subplot(gs[row * cellsize:(row + 1) * cellsize, col * cellsize:(col + 1) * cellsize])
                ax.set_yticklabels([]);
                ax.set_xticklabels([])
                ax.set_xticks([]);
                ax.set_yticks([])

                if (col == 0):  # VIS/IR sandwich
                    gridticks = visticks
                    pcimg = ax.imshow(rgbs[sample_cter]);
                    #pcimg = ax.imshow(rgbs[sample_cter][slice05km, slice05km, slice(None)]);
                    sample_cter += 1
                    plt.title('')
                    ax.set_xlabel('')

                    # gray gridlines
                    ax.set_yticks(gridticks)
                    ax.set_xticks(gridticks)
                    ax.grid(True, linestyle=ls, linewidth=glw)
                    ax.set_xticklabels([]);
                    ax.set_yticklabels([]);
                    ax.tick_params(length=0.0)
                    # center point, denoting pixel of interest
                    # ax.plot(gridticks[1],gridticks[1],marker='x',color='black',markersize=10)

                    ax.set_ylabel(timestamps[row], fontsize=10)
                    plt.suptitle('LRP for Lat,Lon: ' + '{:.2f}'.format(mlats[row]) + ', ' + '{:.2f}'.format(mlons[row]), fontsize=14, y=.93)
                    print('row', row)
                    label = str(round(preds[row][middlepixelind] * 100, 4)) + '%'
                    print('pred label', label)
                    ax.annotate(label, (0.78, 0.07), xycoords='axes fraction', va='center', fontsize=14,
                                backgroundcolor='black', color='white')

                    if (row == nrows - 1):
                        # colors = ['black','red','black','green','black','blue','black','red','black','green','black','blue','black']
                        # words = ['(','R',',','G',',','B',') = (','CH13',', ','CH02',', ', 'CH05',')']
                        # rainbow_text.rainbow_text(0.01, -0.16, words, colors, size=10)
                        ax.annotate('Daytime cloud phase RGB', (0.01, -0.06), xycoords='axes fraction', va='center',
                                    fontsize=10, color='black')
                        ax.annotate('(R,G,B) = (CH13, CH02, CH05)', (0.01, -0.16), xycoords='axes fraction',
                                    va='center', fontsize=10, color='black')
            else:
                # RGB of relevance; assumes you want CH13 = red, CH02 = green, CH05 = blue

                # placeholder for the bytes
                ch02shape = np.shape(lrp[0][row, ..., -1])  # input index = 0 (i.e., CH02)
                newshape = (ch02shape[0], ch02shape[1], 3)
                rgb_rel = np.zeros(newshape, dtype=np.ushort)[
                    slice05km, slice05km, slice(None)]  # kind of a weird way to do it, but it works

                # find the right input index and channel
                for cc, ch in enumerate(['C13', 'C02', 'C05']):
                    for ii, inp in enumerate(inputs):
                        if (ch in inp):
                            iidx = ii
                            # find the right channel index
                            cidx = inp.index(ch)
                            break

                    dat = np.array(lrp[iidx][row, ..., cidx][chans[ch]['slice'], chans[ch]['slice']])
                    new_dat = ((dat - chans[ch]['imin']) * (1 / (chans[ch]['imax'] - chans[ch]['imin']) * 255)).astype(
                        'uint8')
                    if (ch == 'C13'):
                        new_dat = new_dat.repeat(4, axis=0).repeat(4, axis=1)  # artificially make 0.5-km res
                    elif (ch == 'C05'):
                        new_dat = new_dat.repeat(2, axis=0).repeat(2, axis=1)  # artificially make 0.5-km res

                    rgb_rel[..., cc] = new_dat

                pcimg = ax.imshow(rgb_rel)

                # gray gridlines
                ax.set_yticks(chans['C02']['ticks'])
                ax.set_xticks(chans['C02']['ticks'])
                ax.grid(True, linestyle=ls, linewidth=glw)
                ax.set_xticklabels([]);
                ax.set_yticklabels([]);
                ax.tick_params(length=0.0)
                # center point, denoting pixel of interest
                # ax.plot(chans['CH02']['ticks'][1],chans['CH02']['ticks'][1],marker='x',color='gray',markersize=10)

                if (row == nrows - 1):
                    ax.annotate('Relevance RGB', (0.01, -0.06), xycoords='axes fraction', va='center', fontsize=10,
                                color='black')
                    ax.annotate('(R,G,B) = (CH13, CH02, CH05)', (0.01, -0.16), xycoords='axes fraction', va='center',
                                fontsize=10, color='black')

    else:
        nrows = nsamples
        ncols = 1 + len(chans)  # image + LRP for each channel

        fig = plt.figure(num=None, figsize=(ncols * cellsize, nrows * cellsize), dpi=300, facecolor='w',
                         edgecolor='k')  # , constrained_layout=True)
        gs = gridspec.GridSpec(ncols=ncols * cellsize, nrows=nrows * cellsize,
                               figure=fig)  # the +1 is basically to fit the legends at the bottom

        sample_cter = 0

        for row in range(nrows):
            alpha = 0.25 if (solzen[row] < 85) else 1
            for col in range(ncols):
                ch = chanlist[col - 1]
                # new axis required
                if (False and row == nrows - 1 and col != 0):  # last row; 1 through n column
                    ax = plt.subplot(gs[row * cellsize:(row + 1) * cellsize + 1, col * cellsize:(col + 1) * cellsize])
                else:
                    ax = plt.subplot(gs[row * cellsize:(row + 1) * cellsize, col * cellsize:(col + 1) * cellsize])

                ax.set_yticklabels([]);
                ax.set_xticklabels([])
                ax.set_xticks([]);
                ax.set_yticks([])

                if (col == 0):  # VIS/IR sandwich
                    irdata = chans['C13']['data'][row, ..., 0]
                    #irdata = chans['C13']['data'][row, ..., 0][slice2km, slice2km]
                    # if(alpha < 1):
                    #  dat = np.flipud(chans['CH02']['data'][row,...,0][slice05km,slice05km])
                    #  pcimg = ax.pcolormesh(dat, vmin=0, vmax=1, cmap=plt.get_cmap('Greys_r'))
                    #  irdata = zoom(np.flipud(ma.masked_greater(irdata,273)),4)
                    #  pcimg = ax.pcolormesh(irdata, vmin=180, vmax=310, cmap=irw_ctc(), alpha=alpha, linewidth=0, antialiased=True)
                    #  gridticks=visticks
                    # else:
                    #  pcimg = ax.pcolormesh(np.flipud(irdata), vmin=180, vmax=310, cmap=irw_ctc())
                    #  gridticks=irticks

                    gridticks = visticks
                    pcimg = ax.imshow(rgbs[sample_cter][slice05km, slice05km, slice(None)]);
                    #pcimg = ax.imshow(rgbs[sample_cter]);
                    sample_cter += 1
                    # rgbs[sample_cter].plot.imshow(rgb='bands',ax=ax); sample_cter += 1
                    # ax.axis('off')
                    plt.title('')
                    ax.set_xlabel('')

                    # gray gridlines
                    ax.set_yticks(gridticks)
                    ax.set_xticks(gridticks)
                    ax.grid(True, linestyle=ls, linewidth=glw)
                    ax.set_xticklabels([]);
                    ax.set_yticklabels([]);
                    ax.tick_params(length=0.0)
                    # center point, denoting pixel of interest
                    # ax.plot(gridticks[1],gridticks[1],marker='x',color='black',markersize=10)

                    #ax.set_ylabel(
                    #    'Lat,Lon: ' + '{:.2f}'.format(mlats[row]) + ', ' + '{:.2f}'.format(mlons[row]) + '  ' +
                    #    timestamps[row], fontsize=10)
                    ax.set_ylabel(timestamps[row], fontsize=10)
                    plt.suptitle('LRP for Lat, Lon: ' + '{:.2f}'.format(mlats[row]) + ', ' + '{:.2f}'.format(mlons[row]),
                                 fontsize=14, y=.93)
                    #label = str(int(preds[row][middlepixelind] * 100)) + '%'
                    print('row', row)
                    label = str(int(round(preds[row][middlepixelind] * 100))) + '%'
                    print('label: ', label)
                    ax.annotate(label, (0.78, 0.07), xycoords='axes fraction', va='center', fontsize=14,
                                backgroundcolor='black', color='white')

                    if (row == nrows - 1):
                        ax.annotate('Daytime cloud phase RGB', (0.01, -0.06), xycoords='axes fraction', va='center',
                                    fontsize=10, color='black')
                        ax.annotate('(R,G,B) = (CH13, CH02, CH05)', (0.01, -0.16), xycoords='axes fraction',
                                    va='center', fontsize=10, color='black')

                #   cbar = plt.colorbar(pcimg, orientation='horizontal', pad=0.01, shrink=0.9, extendfrac=0, drawedges=0, ax=ax, ticks=np.arange(180,311,30))
                #   cbar.set_label('ABI CH13 BT [K]',fontsize=10)
                #   cbar.ax.tick_params(labelsize=10)
                #   cbar.set_alpha(1.0)
                #   cbar.draw_all()

                else:  # LRP for everything else
                    # find the right input index and channel
                    for ii, inp in enumerate(inputs):
                        if (ch in inp):
                            iidx = ii
                            # find the right channel index
                            cidx = inp.index(ch)
                            break

                    dat = np.flipud(lrp[iidx][row, ..., cidx])
                    # pcimage = ax.pcolormesh(dat[chans[ch]['slice'],chans[ch]['slice']], vmin=0, vmax=dat.max()*0.5, cmap=plt.get_cmap('Reds'))
                    print('LRP ', ch, ' Max:', np.nanmax(dat[chans[ch]['slice'], chans[ch]['slice']]))
                    print('LRP ', ch, ' Min:', np.nanmin(dat[chans[ch]['slice'], chans[ch]['slice']]))
                    pcimage = ax.pcolormesh(dat[chans[ch]['slice'], chans[ch]['slice']], vmin=chans[ch]['imin'],
                                            vmax=chans[ch]['imax'], cmap=plt.get_cmap('seismic'))  # 'Reds'))
                    #pcimage = ax.pcolormesh(dat[chans[ch]['slice'], chans[ch]['slice']], vmin=chans[ch]['imin'],
                                            #vmax=chans[ch]['imax'], cmap=plt.get_cmap('gist_stern'))#'Reds'))

                    # gray gridlines
                    ax.set_yticks(chans[ch]['ticks'])
                    ax.set_xticks(chans[ch]['ticks'])
                    ax.grid(True, linestyle=ls, linewidth=glw)
                    ax.set_xticklabels([]);
                    ax.set_yticklabels([]);
                    ax.tick_params(length=0.0)
                    # center point, denoting pixel of interest
                    #ax.plot(chans[ch]['ticks'][1], chans[ch]['ticks'][1], marker='x', color='black', markersize=10)

                    if (row == nrows - 1):  # last img in col
                        cbaxes = fig.add_axes(
                            [0.32+ (col - 1) * 0.2, 0.07, 0.18, 0.025])  # left bottom width height
                        cbar = fig.colorbar(pcimage, orientation='horizontal', shrink=0.9, pad=0.02, extendfrac=0,
                                            drawedges=0,
                                            cax=cbaxes, ticks=chans[ch]['irange'])
                        # cbar = fig.colorbar(pcimage, orientation='horizontal', aspect=15,shrink=0.9, pad=0.1, extendfrac=0, drawedges=0,
                        #       ticks=chans[ch]['irange'])
                        cbar.ax.set_xticklabels(chans[ch]['ilabs'])
                        cbar.set_label(chans[ch]['cbar_lab'], fontsize=10)
                        cbar.ax.tick_params(labelsize=10)

    print('OUTDIR', outdir)
    filepath = os.path.join(outdir, 'LRP_' + prefix + '.png')
    plt.savefig(filepath, bbox_inches='tight')


#################################################################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Takes a list of lats/lons/datetimes OR a list of zarr inds and creates LRP/SAL plots')
    parser.add_argument('filelist', help='A list of lats/lons/datetimes or zarr inds. See PLOT_LIST.config', type=str,
                        nargs=1)
    parser.add_argument('model', help='CNN model used for predictions.', type=str, nargs=1)
    parser.add_argument('-mc', '--model_config_file',
                        help='This file contains mean and SD of channels (a pickle file). Supply if using "LATLONDT" method.',
                        type=str, nargs=1)
    parser.add_argument('-g', '--glmdir',
                        help='Directory to glm FED. Default = /apollo/grain/saved_probsevere_data/lightning/Ymd/GLM/goes_east/agg/', \
                        default=['/apollo/grain/saved_probsevere_data/lightning/%Y%m%d/GLM/goes_east/agg/'], nargs=1,
                        type=str)
    parser.add_argument('-a', '--abidir',
                        help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/', \
                        default=['/arcdata/goes/grb/goes16/%Y/%Y_%m_%d_%j/abi/L1b/RadC/'], type=str, nargs=1)
    parser.add_argument('-p', '--prefix', help='Prefix for outfile. Default: "tmp"', type=str, nargs=1, default=['tmp'])
    parser.add_argument('-d', '--detail',
                        help='Make image and relevance for each channel. Only 1 or 2 samples recommended.',
                        action="store_true")
    parser.add_argument('-r', '--rgb_relevance', help='Make image and relevance RGBs.', action="store_true")
    parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/home/sbradshaw/ci_ltg'])

    args = parser.parse_args()

    UNET_METRIC_LIST = ['accuracy', keras_metrics.iou, keras_metrics.iou_thresholded]
    METRIC_FUNCTION_DICT = {
        'accuracy': keras_metrics.accuracy,
        'iou': keras_metrics.iou,
        'iou_thresholded': keras_metrics.iou_thresholded
    }

    print(args.model[0])
    cnn = models.load_model(args.model[0])  # ,custom_objects=METRIC_FUNCTION_DICT) #,compile=False,)
    cnn_nf = cnn

    x = cnn.layers[-1].output
    x_nf = cnn_nf.layers[-1].output
    x = layers.Flatten()(x)

    cnn = models.Model(inputs=cnn.input, outputs=x)
    print(cnn.summary())

    cnn_nf = models.Model(inputs=cnn_nf.input, outputs=x_nf)


    if (args.model_config_file):
        mc = pickle.load(open(args.model_config_file[0], 'rb'))
    else:
        mc = pickle.load(open(f"{os.path.dirname(args.model[0])}/model_config.pkl", "rb"))

    cf = args.filelist[0]

    # config list
    file1 = open(args.filelist[0], 'r')
    lines = file1.readlines()

    count = 0

    lats = [];
    lons = [];
    dts = [];
    ind = []
    for line in lines:
        if (count == 0):
            latlon = True if (line[0] == 'L') else False
            count += 1
            continue

        if (line[0] == '#'): continue

        if (latlon):
            parts = line.split()
            lats.append(float(parts[0]));
            lons.append(float(parts[1]));
            dts.append(parts[2])
        else:
            parts = line.split()
            ind.append(int(parts[0]))

        count += 1

    print('OUTDIR', args.outdir[0])
    plot_lrp_sal(cnn, cnn_nf, abidir=args.abidir[0], glmdir=args.glmdir[0],
                 mlats=lats, mlons=lons, dts=dts, ind=ind, mc=mc, prefix=args.prefix[0],
                 detail=args.detail, rgb_relevance=args.rgb_relevance, outdir=args.outdir[0])

    print('It should have plotted.')

##    inputs = []
#    for i in dict_keys:
#        inputs.append(mc['inputs'][i][0])

#    print(inputs)