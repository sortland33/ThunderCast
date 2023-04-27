#Author: Stephanie M. Ortland
import os, sys
from glob import glob
import torch
import psutil
import pathlib
from netCDF4 import Dataset
import torch.nn as nn
from datetime import datetime,timedelta
from time import time
from torch.utils.data import DataLoader
import errno
import numpy as np
import pytorch_lightning as pl
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
import dask
import scipy.ndimage

# Define functions
def mkdir_p(path):
    try:
        os.makedirs(path)
    except (OSError, IOError) as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_netcdf(file, params):
    """
    This function loads data from a netcdf file and stores it in a torch tensor
    If the values going into this function are inputs, they are saved as a list of tensors. If targets, this outputs
    the tensor.
    """
    fields = params['vars']
    threshold = params['threshold']
    norm = params['norm_vals']
    #device = params['device']
    ds   = Dataset(file, 'r')
    X = []
    for res_group in fields:
        if res_group == 'target':
            y = None
            target_dict = fields[res_group]
            target_group = list(target_dict.keys())[0]
            target_var = target_dict[target_group][0]
            ds_group = ds.groups[target_group]
            y = torch.tensor(ds_group[target_var]).double()#, device=device).double()
            #y = torch.where((y <= threshold) & (y != -999), 0., y)
            y = torch.where((y <= threshold), 0., y)
            y = torch.where((y >= threshold), 1., y)
            if len(torch.unique(y)) > 2:
                print('Error file with value other than 0 or 1:', file)
                print('Unique values:', torch.unique(y))
            if params['target_buffer']:
                y = torch.tensor(scipy.ndimage.maximum_filter(y, footprint=np.ones((21,21)))) #(5,5) is 2 km buffer, (7, 7) is 3km, (9, 9) is 4km and (11, 11) is 5km and (13, 13) is 6km and (15, 15) is 7km (17, 17) is 8km (19, 19) is 9 km (21, 21) is 10km
            y = y.float()
            #y = y[None]

        else:
            x = None
            for variable in fields[res_group]:
                ds_group = ds.groups[res_group]
                tmp = torch.tensor(ds_group[variable]).double()#, device=device).double()
                mean_std_dict = norm[variable]
                tmp = (tmp - mean_std_dict['mean'])/mean_std_dict['std']
                tmp = tmp[None] #expands dimension so the first dimension becomes number of channels in group
                if x is None:
                    x = tmp + 0.
                else:
                    x = torch.cat((x,tmp))
            X.append(x.to(dtype=torch.float))

    ds.close()
    return X, y

def save_pt(file, params):
    X, y = load_netcdf(file, params)
    basefile = os.path.basename(file)[:-3]
    # save target tensor
    targets_PTfilepath = params['target_path'] + '/' + basefile + '.pt'
    torch.save(y, targets_PTfilepath)

    # format and save inputs
    inputs_PTfilepath = params['inputs_path'] + '/' + basefile + '.pt'
    if 'goes16_half_km' in params['vars'].keys():
        upsamp = nn.Upsample(size=(640, 640), mode='nearest')  # size here should match the size of the images at 0.5km resolution
        desired_shape = 640
    else:
        upsamp = nn.Upsample(size=(320, 320), mode='nearest')
        desired_shape = 320

    inp = X[0]
    for i in range(len(X)):
        if i == 0:
            if X[i].shape[-1] != desired_shape:
                inp = torch.unsqueeze(X[i], 0)
                inp = upsamp(inp)
                inp = torch.squeeze(inp, 0)
            else:
                inp = X[i]
        else:
            if X[i].shape[-1] != desired_shape:
                inp_temp = torch.unsqueeze(X[i], 0)
                inp_temp = upsamp(inp_temp)
                inp_temp = torch.squeeze(inp_temp, 0)
                inp = torch.cat((inp, inp_temp), dim=0)
            else:
                inp_temp = X[i]
                inp = torch.cat((inp, inp_temp), dim=0)
    torch.save(inp, inputs_PTfilepath)
    return inputs_PTfilepath

#Main program
if __name__ == "__main__":
    t_start = time()
    print('Time at the start of the program:', datetime.now())
    #____________________________User Parameters_____________________________________________
    #Day only
    # norm_vals = {'C02': {'mean': 23.08378, 'std': 13.866257},
    #              'C05': {'mean': 16.540283, 'std': 6.8535457},
    #              'C13': {"mean": 268.9491, "std": 20.004034},
    #              'C11': {"mean": 266.74457, "std": 19.122293},
    #              'C14': {'mean': 267.59595, 'std': 19.97203},
    #              'C15': {'mean': 264.79877, 'std': 19.170725},
    #              'solar_azimuth': {'mean': 201.90517, 'std': 1.8463212},
    #              'solar_zenith': {'mean': 48.27021, 'std': 1.021574}}
    #
    #All
    norm_vals = {'C02': {'mean': 13.338485, 'std': 7.5519395},
                 'C05': {'mean': 9.244341, 'std': 3.830976},
                 'C13': {"mean": 263.5579, "std": 19.133537},
                 'C11': {"mean": 261.74683, "std": 18.389097},
                 'C14': {'mean': 262.35043, 'std': 19.177515},
                 'C15': {'mean': 260.04007, 'std': 18.560625},
                 'solar_azimuth': {'mean': 194.34933, 'std': 2.9718854},
                 'solar_zenith': {'mean': 78.79489, 'std': 1.0483047}}
    #Night only
    # norm_vals = {'C02': {'mean': 0.22909436, 'std': 0.23226656},
    #              'C05': {'mean': 0.17578533, 'std': 0.224306},
    #              'C13': {"mean": 260.64502, "std": 19.135427},
    #              'C11': {"mean": 259.07098, "std": 18.46789},
    #              'C14': {'mean': 259.52588, 'std': 19.256767},
    #              'C15': {'mean': 257.51395, 'std': 18.768995},
    #              'solar_azimuth': {'mean': 190.32436, 'std': 4.4344344},
    #              'solar_zenith': {'mean': 112.99715, 'std': 1.0937598}}

    vars = {'goes16_half_km':['C02'], 'goes16_one_km':['C05'], 'goes16_two_km':['C13', 'C15'], 'target':{'MRMS_one_km':['reflectivity_60min_neg10C_max']}} #in format group:[variable(s)]
    data_set = 'testing' #Can be either validation or training or testing
    training_file_path = '/ships19/grain/convective_init/data_lists/training/train_files.txt'
    validation_file_path = '/ships19/grain/convective_init/data_lists/validation/val_files.txt'
    testing_file_path = '/ships19/grain/convective_init/data_lists/testing/test_files.txt'
    pt_path = '/ships19/grain/convective_init/pt_tensors/'
    add_target_buffer = True
    buffer_value = '10km'

    inputs = []
    for k in vars.keys():
        if k != 'target':
            inputs.append(vars[k])
    print('inputs', inputs)
    ps_2km = (400, 400)
    ps_1km = tuple([_ * 2 for _ in ps_2km])
    ps_05km = tuple([_ * 4 for _ in ps_2km])
    input_tuples = []
    for i, inp in enumerate(inputs):  # channels first for pytorch
        if (inp[0] in vars['goes16_two_km']):
            input_tuples.append((len(inp), ps_2km[0], ps_2km[1]))
        elif (inp[0] in vars['goes16_one_km']):
            input_tuples.append((len(inp), ps_1km[1], ps_1km[1]))
        elif (inp[0] in vars['goes16_half_km']):
            input_tuples.append((len(inp), ps_05km[0], ps_05km[1]))
        else:
            print("Can't find " + inp + " in vars!")
            sys.exit()
    print('input tuples', input_tuples)

    if data_set == 'validation':
        file_path = validation_file_path
    elif data_set == 'testing':
        file_path = testing_file_path
    else:
        file_path = training_file_path

    with open(file_path, 'r') as data_files:
        files = data_files.read().splitlines()
    print('Number of files:', len(files))

    pt_basepath = os.path.basename(file_path)[:-4]
    if add_target_buffer:
        pt_path_target = pt_path + pt_basepath + '_' + buffer_value + '/target'
        pt_path_inputs = pt_path + pt_basepath + '_' + buffer_value + '/inputs'
    else:
        pt_path_target = pt_path + pt_basepath + '/target'
        pt_path_inputs = pt_path + pt_basepath + '/inputs'
    mkdir_p(pt_path_target)
    mkdir_p(pt_path_inputs)
    params = {'norm_vals': norm_vals, 'threshold': 30.0, 'vars': vars, 'target_path': pt_path_target, 'inputs_path': pt_path_inputs, 'target_buffer':add_target_buffer}

    #If you don't want to use dask:
    # for j in files:
    #     print('j', j)
    #     res = save_pt(j, params)
    t = time()
    print('Converting now...')
    client = Client(n_workers=10, threads_per_worker=1, memory_limit='10GB')
    res = client.map(save_pt, files, params=params)
    info = client.gather(res)
    client.close()
    print('number of input files:', len(info))
    print('time to complete conversion: ', time() - t)