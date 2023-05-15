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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import kornia.augmentation as K
from torchmetrics import Metric
from PL_Metrics import CriticalSuccessIndex as CSI
from PL_Metrics import POD, FAR, SR, Accuracy, Specificity, F1Score, Recall, Precision, NoRes, CEFs
from satpy import Scene
import argparse
from pyproj import Proj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyresample as pr
import pygrib as pg
from metpy.plots import ctables
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy.ma as ma
from scipy.ndimage import zoom
import pyart
import matplotlib

# python plot_pred_subplots.py -DT 2021-08-25-18-31 -t 5 -lat 41.41 -lon -90.85 -hs 55
#hole punch: python plot_pred_subplots.py -DT 2022-08-11-14-01 -t 5 -lat 29.08 -lon -80.96 -hs 55
#sea breeze: python plot_pred_subplots.py -DT 2022-08-31-19-01 -t 5 -lat 28.52 -lon -80.65 -hs 39
#UT: python plot_pred_subplots.py -DT 2022-08-27-17-46 -t 20 -lat 36.13 -lon -109.56 -hs 71
#Thundersnow: python plot_pred_subplots_grid.py -DT 2022-12-15-05-21 -t 5 -lat 43.07 -lon -89.41 -hs 119 -c C13


os.environ['SATPY_CONFIG_PATH'] = '/home/sortland/ci_ltg/PPP_config/'
os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

def mkdir_p(path):
  try:
    os.makedirs(path)
  except (OSError,IOError) as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise

def get_abifile_lists(rootdatadir, dts):
    list = []
    abi_channels = ['02', '05', '11', '13', '14', '15'] #for this application we do not need all the channels
    for i in range(len(dts)):
        satdir = dts[i].strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        filenames = []
        for ch in abi_channels:
            netcdfpattern = dts[i].strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
            try:
                filenames.append(glob(netcdfpattern)[0])
            except IndexError:
                continue
        list.append(filenames)
    return list

def get_abifile_dict(rootdatadir, dts):
    dict = {}
    abi_channels = ['02', '05', '11', '13', '14', '15'] #for this application we do not need all the channels
    for i in range(len(dts)):
        satdir = dts[i].strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
        filenames = []
        for ch in abi_channels:
            netcdfpattern = dts[i].strftime(satdir + 'OR_ABI-L1b-RadC-M*C' + ch + '_G16_s%Y%j%H%M*')
            try:
                filenames.append(glob(netcdfpattern)[0])
            except IndexError:
                continue
        dict[dts[i]] = filenames
    return dict

def get_satpy_scene(abi_files):  # get satpy scene to be cropped to patches
    CHs = ['C13', 'C15', 'C11', 'C14', 'C02', 'C05']
    bad_data = False
    try:
        scn = Scene(reader='abi_l1b', filenames=abi_files)
        scn.load(CHs)  # Load the scene and check for data problems
    except:
        print('Scene would not load.')
        bad_data = True
        scn = None
        return scn, bad_data
    return scn, bad_data

def format_inputs(scn, norm_vals, vars, center_x_idx, center_y_idx, hs1km, full_US):
    X = []
    for res_group in vars:
        if res_group == 'target':
            continue
        else:
            x = None
            if full_US:
                crop_scn = scn
            else:
                center_x_ind = int(center_x_idx)
                center_y_ind = int(center_y_idx)
                hs = hs1km
                ymin_index = center_y_ind + hs
                ymax_index = center_y_ind - hs
                xmin_index = center_x_ind - hs
                xmax_index = center_x_ind + hs
                area_def = scn['C05'].attrs['area']
                x_coords, y_coords = area_def.get_proj_vectors()
                center_y = y_coords[center_y_ind]
                center_x = x_coords[center_x_ind]
                xmin = x_coords[xmin_index]
                xmax = x_coords[xmax_index]
                ymin = y_coords[ymin_index]
                ymax = y_coords[ymax_index]
                crop_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])
            for variable in vars[res_group]:
                tmp = torch.tensor(crop_scn[variable].values).double()
                mean_std_dict = norm_vals[variable]
                tmp = (tmp - mean_std_dict['mean']) / mean_std_dict['std']
                tmp = tmp[None]
                if x is None:
                    x = tmp + 0.
                else:
                    x = torch.cat((x, tmp))
            X.append(x.to(dtype=torch.float))

    if 'goes16_half_km' in vars.keys() or 'goes16_one_km' in vars.keys():
        desired_shape = X[0].shape[-1]
        upsamp = nn.Upsample(size=(desired_shape, desired_shape),
                             mode='nearest')  # size here should match the size of the images at 0.5km or 1km resolution
    else:
        desired_shape = (X[-1].shape[-1])*2 #2km to 1km
        upsamp = nn.Upsample(size=(desired_shape, desired_shape), mode='nearest')

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
    #X = torch.unsqueeze(inp,0) #need to have an extra dimensions since not coming from data loader [1, 4, 800, 800]
    X = inp
    return X

def load_pt(file, params):
    X = torch.load(file)
    base = os.path.basename(file)
    parts = file.split('/')
    target_folder = '/'
    for f in range(len(parts)):
        if f != 0 and f <= len(parts) - 3:
            target_folder = target_folder + parts[f] + '/'
    target_file = target_folder + 'target/' + base
    y = torch.load(target_file)
    return X, y

class DataAugmentation(nn.Module):
    """
    This sets up data augmentation methods
    """
    def __init__(self):
        super(DataAugmentation, self).__init__()

        self.hflip = K.RandomHorizontalFlip(p=0.25)
        self.vflip = K.RandomVerticalFlip(p=0.25)

    def forward(self, inputs, targets):
        inputs = self.hflip(inputs)
        targets = self.hflip(targets, self.hflip._params)

        inputs = self.vflip(inputs)
        targets = self.vflip(targets, self.vflip._params)
        return inputs, targets

class UNet2D_LitModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.nfmaps = nfmaps = params['num_conv_filters']  # an int
        self.nclass = nclass = params['nclass']
        self.loss_fcn = nn.CrossEntropyLoss(weight=params['weights'])
        self.learning_rate = params['learning_rate']
        self.patience = params['rlr_patience']
        self.factor = params['rlr_factor']
        self.cooldown = params['rlr_cooldown']
        self.batch_size = params['batch_size']
        self.nIn = nIn = params['number_inputs']
        self.max_needed = params['maxpool_end']
        self.training_files = params['training_files']
        self.validation_files = params['validation_files']
        self.testing_files = params['testing_files']
        self.transform = DataAugmentation()

        self.val_accuracy = Accuracy()
        self.val_precision = Precision()
        self.val_recall = Recall()
        self.val_F1score = F1Score()
        self.val_specificity = Specificity()

        self.test_F1score = F1Score()
        self.test_accuracy = Accuracy()
        self.test_recall = Recall()
        self.test_specificity = Specificity()
        self.test_FAR = FAR()
        self.test_precision = Precision()
        self.test_NoRes = NoRes()
        #Need the following by threshold:
        #CEFs
        self.test_CEF_0 = CEFs(threshold1=0, threshold2=0.05)
        self.test_CEF_5 = CEFs(threshold1=0.05, threshold2=0.10)
        self.test_CEF_10 = CEFs(threshold1=0.10, threshold2=0.15)
        self.test_CEF_15 = CEFs(threshold1=0.15, threshold2=0.20)
        self.test_CEF_20 = CEFs(threshold1=0.20, threshold2=0.25)
        self.test_CEF_25 = CEFs(threshold1=0.25, threshold2=0.30)
        self.test_CEF_30 = CEFs(threshold1=0.30, threshold2=0.35)
        self.test_CEF_35 = CEFs(threshold1=0.35, threshold2=0.40)
        self.test_CEF_40 = CEFs(threshold1=0.40, threshold2=0.45)
        self.test_CEF_45 = CEFs(threshold1=0.45, threshold2=0.50)
        self.test_CEF_50 = CEFs(threshold1=0.50, threshold2=0.55)
        self.test_CEF_55 = CEFs(threshold1=0.55, threshold2=0.60)
        self.test_CEF_60 = CEFs(threshold1=0.60, threshold2=0.65)
        self.test_CEF_65 = CEFs(threshold1=0.65, threshold2=0.70)
        self.test_CEF_70 = CEFs(threshold1=0.70, threshold2=0.75)
        self.test_CEF_75 = CEFs(threshold1=0.75, threshold2=0.80)
        self.test_CEF_80 = CEFs(threshold1=0.80, threshold2=0.85)
        self.test_CEF_85 = CEFs(threshold1=0.85, threshold2=0.90)
        self.test_CEF_90 = CEFs(threshold1=0.90, threshold2=0.95)
        self.test_CEF_95 = CEFs(threshold1=0.95, threshold2=1.0)
        #POD
        self.test_POD_5 = POD(threshold= 0.05)
        self.test_POD_10 = POD(threshold= 0.1)
        self.test_POD_15 = POD(threshold=0.15)
        self.test_POD_20 = POD(threshold= 0.2)
        self.test_POD_25 = POD(threshold=0.25)
        self.test_POD_30 = POD(threshold= 0.3)
        self.test_POD_35 = POD(threshold=0.35)
        self.test_POD_40 = POD(threshold= 0.4)
        self.test_POD_45 = POD(threshold=0.45)
        self.test_POD_50 = POD()
        self.test_POD_55 = POD(threshold=0.55)
        self.test_POD_60 = POD(threshold= 0.6)
        self.test_POD_65 = POD(threshold=0.65)
        self.test_POD_70 = POD(threshold= 0.7)
        self.test_POD_75 = POD(threshold=0.75)
        self.test_POD_80 = POD(threshold= 0.8)
        self.test_POD_85 = POD(threshold=0.85)
        self.test_POD_90 = POD(threshold= 0.9)
        self.test_POD_95 = POD(threshold= 0.95)
        #CSI
        self.test_CSI_5 = CSI(threshold= 0.05)
        self.test_CSI_10 = CSI(threshold= 0.1)
        self.test_CSI_15 = CSI(threshold=0.15)
        self.test_CSI_20 = CSI(threshold= 0.2)
        self.test_CSI_25 = CSI(threshold=0.25)
        self.test_CSI_30 = CSI(threshold= 0.3)
        self.test_CSI_35 = CSI(threshold=0.35)
        self.test_CSI_40 = CSI(threshold= 0.4)
        self.test_CSI_45 = CSI(threshold=0.45)
        self.test_CSI_50 = CSI()
        self.test_CSI_55 = CSI(threshold=0.55)
        self.test_CSI_60 = CSI(threshold= 0.6)
        self.test_CSI_65 = CSI(threshold=0.65)
        self.test_CSI_70 = CSI(threshold= 0.7)
        self.test_CSI_75 = CSI(threshold=0.75)
        self.test_CSI_80 = CSI(threshold= 0.8)
        self.test_CSI_85 = CSI(threshold=0.85)
        self.test_CSI_90 = CSI(threshold= 0.9)
        self.test_CSI_95 = CSI(threshold= 0.95)
        #SR
        self.test_SR_5 = SR(threshold=0.05)
        self.test_SR_10 = SR(threshold=0.1)
        self.test_SR_15 = SR(threshold=0.15)
        self.test_SR_20 = SR(threshold=0.2)
        self.test_SR_25 = SR(threshold=0.25)
        self.test_SR_30 = SR(threshold=0.3)
        self.test_SR_35 = SR(threshold=0.35)
        self.test_SR_40 = SR(threshold=0.4)
        self.test_SR_45 = SR(threshold=0.45)
        self.test_SR_50 = SR()
        self.test_SR_55 = SR(threshold=0.55)
        self.test_SR_60 = SR(threshold=0.6)
        self.test_SR_65 = SR(threshold=0.65)
        self.test_SR_70 = SR(threshold=0.7)
        self.test_SR_75 = SR(threshold=0.75)
        self.test_SR_80 = SR(threshold=0.8)
        self.test_SR_85 = SR(threshold=0.85)
        self.test_SR_90 = SR(threshold=0.9)
        self.test_SR_95 = SR(threshold=0.95)

        #self.stat_scores = StatScores(average = 'micro', num_classes=2, mdmc_reduce='samplewise')

        self.save_hyperparameters()

        self.dconv_down1 = self.double_conv(nIn, nfmaps * 1)
        self.dconv_down2 = self.double_conv(nfmaps * 1, nfmaps * 2)
        self.dconv_down3 = self.double_conv(nfmaps * 2, nfmaps * 4)
        self.dconv_down4 = self.double_conv(nfmaps * 4, nfmaps * 8)
        self.dconv_down5 = self.double_conv(nfmaps * 8, nfmaps * 16)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.maxpool4 = nn.MaxPool2d(4, 4)

        self.convtrans1 = nn.ConvTranspose2d(nfmaps * 16, nfmaps * 8, kernel_size=2, stride=2, padding=0)
        self.dconv_up1 = self.double_conv(nfmaps * 16, nfmaps * 8)
        self.convtrans2 = nn.ConvTranspose2d(nfmaps * 8, nfmaps * 4, kernel_size=2, stride=2, padding=0)
        self.dconv_up2 = self.double_conv(nfmaps * 8, nfmaps * 4)
        self.convtrans3 = nn.ConvTranspose2d(nfmaps * 4, nfmaps * 2, kernel_size=2, stride=2, padding=0)
        self.dconv_up3 = self.double_conv(nfmaps * 4, nfmaps * 2)
        self.convtrans4 = nn.ConvTranspose2d(nfmaps * 2, nfmaps * 1, kernel_size=2, stride=2, padding=0)
        self.dconv_up4 = self.double_conv(nfmaps * 2, nfmaps * 1)  # if you want concatenation, it would be something like this 128 +  64, 64

        self.conv_last = nn.Sequential(
            nn.Conv2d(nfmaps, nclass, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1))

    def forward(self, x):
        inp = torch.stack(x)
        #print('inp shape', inp.shape)

        conv1 = self.dconv_down1(inp)
        conv = self.maxpool(conv1)

        conv2 = self.dconv_down2(conv)
        conv = self.maxpool(conv2)

        conv3 = self.dconv_down3(conv)
        conv = self.maxpool(conv3)

        conv4 = self.dconv_down4(conv)
        conv = self.maxpool(conv4)

        # middle
        conv = self.dconv_down5(conv)

        conv = self.convtrans1(conv)
        conv = torch.cat([conv, conv4], dim=1)
        conv = self.dconv_up1(conv)

        conv = self.convtrans2(conv)
        conv = torch.cat([conv, conv3], dim=1)
        conv = self.dconv_up2(conv)

        conv = self.convtrans3(conv)
        conv = torch.cat([conv, conv2], dim=1)
        conv = self.dconv_up3(conv)

        conv = self.convtrans4(conv)
        conv = torch.cat([conv, conv1], dim=1)
        conv = self.dconv_up4(conv)

        if (self.max_needed):  # reduce "500m" to "1km"
            conv = self.maxpool(conv)

        conv = self.conv_last(conv)

        return conv

    def double_conv(self, in_channels, out_channels, ks=3, bn=False):
        P = ks // 2
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, ks, padding=P),
                             nn.ReLU(inplace=False),
                             nn.Conv2d(out_channels, out_channels, ks, padding=P),
                             nn.ReLU(inplace=False))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience,
                                                               factor=self.factor,
                                                               threshold=0.0001,
                                                               cooldown=self.cooldown)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            y = y.unsqueeze(1)
            x, y = self.transform(x, y)
            y = y.squeeze()
        return x, y

    def training_step(self, train_batch, batch_idx):
        local_batch_list = []
        local_inputs, local_labels = train_batch
        for lb in local_inputs:
            local_batch_list.append(lb.to(dtype=torch.float))
        outputs = self(local_batch_list)
        local_labels = local_labels.to(dtype=torch.long)
        loss = self.loss_fcn(outputs, local_labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        local_batch_list = []
        local_inputs, local_labels = val_batch
        for lb in local_inputs:
            local_batch_list.append(lb.to(dtype=torch.float))
        outputs = self(local_batch_list)
        local_labels = local_labels.to(dtype=torch.long)
        loss = self.loss_fcn(outputs, local_labels)
        #tp, fp, tn, fn, sup = self.stat_scores(outputs, local_labels)

        self.val_accuracy.update(outputs, local_labels)
        self.val_precision.update(outputs, local_labels)
        self.val_recall.update(outputs, local_labels)
        self.val_F1score.update(outputs, local_labels)
        self.val_specificity.update(outputs, local_labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, metric_attribute='val_loss')
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True, metric_attribute='val_specificity')
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, metric_attribute='val_accuracy')
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, metric_attribute='val_precision')
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, metric_attribute='val_recall')
        self.log('val_F1score', self.val_F1score, on_step=False, on_epoch=True, metric_attribute='val_F1score')
        return loss

    def test_step(self, test_batch, batch_idx):
        local_batch_list = []
        local_inputs, local_labels = test_batch
        for lb in local_inputs:
            local_batch_list.append(lb.to(dtype=torch.float))
        outputs = self(local_batch_list)
        local_labels = local_labels.to(dtype=torch.long)
        # print(outputs.shape)
        # print(local_labels.shape)
        loss = self.loss_fcn(outputs, local_labels)
        self.test_accuracy.update(outputs, local_labels)
        self.test_recall.update(outputs, local_labels)
        self.test_F1score.update(outputs, local_labels)
        self.test_FAR.update(outputs, local_labels)
        self.test_specificity.update(outputs, local_labels)
        self.test_precision.update(outputs, local_labels)
        self.test_NoRes.update(outputs, local_labels)

        self.test_POD_10.update(outputs, local_labels)
        self.test_POD_20.update(outputs, local_labels)
        self.test_POD_30.update(outputs, local_labels)
        self.test_POD_40.update(outputs, local_labels)
        self.test_POD_50.update(outputs, local_labels)
        self.test_POD_60.update(outputs, local_labels)
        self.test_POD_70.update(outputs, local_labels)
        self.test_POD_80.update(outputs, local_labels)
        self.test_POD_90.update(outputs, local_labels)
        self.test_SR_10.update(outputs, local_labels)
        self.test_SR_20.update(outputs, local_labels)
        self.test_SR_30.update(outputs, local_labels)
        self.test_SR_40.update(outputs, local_labels)
        self.test_SR_50.update(outputs, local_labels)
        self.test_SR_60.update(outputs, local_labels)
        self.test_SR_70.update(outputs, local_labels)
        self.test_SR_80.update(outputs, local_labels)
        self.test_SR_90.update(outputs, local_labels)
        self.test_CSI_10.update(outputs, local_labels)
        self.test_CSI_20.update(outputs, local_labels)
        self.test_CSI_30.update(outputs, local_labels)
        self.test_CSI_40.update(outputs, local_labels)
        self.test_CSI_50.update(outputs, local_labels)
        self.test_CSI_60.update(outputs, local_labels)
        self.test_CSI_70.update(outputs, local_labels)
        self.test_CSI_80.update(outputs, local_labels)
        self.test_CSI_90.update(outputs, local_labels)
        self.test_CEF_0.update(outputs, local_labels)
        self.test_CEF_5.update(outputs, local_labels)
        self.test_CEF_10.update(outputs, local_labels)
        self.test_CEF_15.update(outputs, local_labels)
        self.test_CEF_20.update(outputs, local_labels)
        self.test_CEF_25.update(outputs, local_labels)
        self.test_CEF_30.update(outputs, local_labels)
        self.test_CEF_35.update(outputs, local_labels)
        self.test_CEF_40.update(outputs, local_labels)
        self.test_CEF_45.update(outputs, local_labels)
        self.test_CEF_50.update(outputs, local_labels)
        self.test_CEF_55.update(outputs, local_labels)
        self.test_CEF_60.update(outputs, local_labels)
        self.test_CEF_65.update(outputs, local_labels)
        self.test_CEF_70.update(outputs, local_labels)
        self.test_CEF_75.update(outputs, local_labels)
        self.test_CEF_80.update(outputs, local_labels)
        self.test_CEF_85.update(outputs, local_labels)
        self.test_CEF_90.update(outputs, local_labels)
        self.test_CEF_95.update(outputs, local_labels)
        #self.stat_scores.update(outputs, local_labels)

        self.log('test_loss', loss, on_step=True, on_epoch=True, metric_attribute='test_loss')
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True,metric_attribute='test_accuracy')
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, metric_attribute='test_recall')
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True,metric_attribute='test_specificity')
        self.log('test_F1score', self.test_F1score, on_step=False, on_epoch=True, metric_attribute='test_F1score')
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, metric_attribute='test_precision')
        self.log('test_NoRes', self.test_NoRes, on_step=False, on_epoch=True, metric_attribute='test_NoRes')

        self.log('test_FAR', self.test_FAR, on_step=False, on_epoch=True, metric_attribute='test_FAR')
        self.log('test_CSI_10', self.test_CSI_10, on_step=False, on_epoch=True, metric_attribute='test_CSI_10')
        self.log('test_CSI_20', self.test_CSI_20, on_step=False, on_epoch=True, metric_attribute='test_CSI_20')
        self.log('test_CSI_30', self.test_CSI_30, on_step=False, on_epoch=True, metric_attribute='test_CSI_30')
        self.log('test_CSI_40', self.test_CSI_40, on_step=False, on_epoch=True, metric_attribute='test_CSI_40')
        self.log('test_CSI_50', self.test_CSI_50, on_step=False, on_epoch=True, metric_attribute='test_CSI_50')
        self.log('test_CSI_60', self.test_CSI_60, on_step=False, on_epoch=True, metric_attribute='test_CSI_60')
        self.log('test_CSI_70', self.test_CSI_70, on_step=False, on_epoch=True, metric_attribute='test_CSI_70')
        self.log('test_CSI_80', self.test_CSI_80, on_step=False, on_epoch=True, metric_attribute='test_CSI_80')
        self.log('test_CSI_90', self.test_CSI_90, on_step=False, on_epoch=True, metric_attribute='test_CSI_90')
        self.log('test_SR_10', self.test_SR_10, on_step=False, on_epoch=True, metric_attribute='test_SR_10')
        self.log('test_SR_20', self.test_SR_20, on_step=False, on_epoch=True, metric_attribute='test_SR_20')
        self.log('test_SR_30', self.test_SR_30, on_step=False, on_epoch=True, metric_attribute='test_SR_30')
        self.log('test_SR_40', self.test_SR_40, on_step=False, on_epoch=True, metric_attribute='test_SR_40')
        self.log('test_SR_50', self.test_SR_50, on_step=False, on_epoch=True, metric_attribute='test_SR_50')
        self.log('test_SR_60', self.test_SR_60, on_step=False, on_epoch=True, metric_attribute='test_SR_60')
        self.log('test_SR_70', self.test_SR_70, on_step=False, on_epoch=True, metric_attribute='test_SR_70')
        self.log('test_SR_80', self.test_SR_80, on_step=False, on_epoch=True, metric_attribute='test_SR_80')
        self.log('test_SR_90', self.test_SR_90, on_step=False, on_epoch=True, metric_attribute='test_SR_90')
        self.log('test_POD_10', self.test_POD_10, on_step=False, on_epoch=True, metric_attribute='test_POD_10')
        self.log('test_POD_20', self.test_POD_20, on_step=False, on_epoch=True, metric_attribute='test_POD_20')
        self.log('test_POD_30', self.test_POD_30, on_step=False, on_epoch=True, metric_attribute='test_POD_30')
        self.log('test_POD_40', self.test_POD_40, on_step=False, on_epoch=True, metric_attribute='test_POD_40')
        self.log('test_POD_50', self.test_POD_50, on_step=False, on_epoch=True, metric_attribute='test_POD_50')
        self.log('test_POD_60', self.test_POD_60, on_step=False, on_epoch=True, metric_attribute='test_POD_60')
        self.log('test_POD_70', self.test_POD_70, on_step=False, on_epoch=True, metric_attribute='test_POD_70')
        self.log('test_POD_80', self.test_POD_80, on_step=False, on_epoch=True, metric_attribute='test_POD_80')
        self.log('test_POD_90', self.test_POD_90, on_step=False, on_epoch=True, metric_attribute='test_POD_90')
        self.log('test_CEF_0', self.test_CEF_0, on_step=False, on_epoch=True, metric_attribute='test_CEF_0')
        self.log('test_CEF_5', self.test_CEF_5, on_step=False, on_epoch=True, metric_attribute='test_CEF_5')
        self.log('test_CEF_10', self.test_CEF_10, on_step=False, on_epoch=True, metric_attribute='test_CEF_10')
        self.log('test_CEF_15', self.test_CEF_15, on_step=False, on_epoch=True, metric_attribute='test_CEF_15')
        self.log('test_CEF_20', self.test_CEF_20, on_step=False, on_epoch=True, metric_attribute='test_CEF_20')
        self.log('test_CEF_25', self.test_CEF_25, on_step=False, on_epoch=True, metric_attribute='test_CEF_25')
        self.log('test_CEF_30', self.test_CEF_30, on_step=False, on_epoch=True, metric_attribute='test_CEF_30')
        self.log('test_CEF_35', self.test_CEF_35, on_step=False, on_epoch=True, metric_attribute='test_CEF_35')
        self.log('test_CEF_40', self.test_CEF_40, on_step=False, on_epoch=True, metric_attribute='test_CEF_40')
        self.log('test_CEF_45', self.test_CEF_45, on_step=False, on_epoch=True, metric_attribute='test_CEF_45')
        self.log('test_CEF_50', self.test_CEF_50, on_step=False, on_epoch=True, metric_attribute='test_CEF_50')
        self.log('test_CEF_55', self.test_CEF_55, on_step=False, on_epoch=True, metric_attribute='test_CEF_55')
        self.log('test_CEF_60', self.test_CEF_60, on_step=False, on_epoch=True, metric_attribute='test_CEF_60')
        self.log('test_CEF_65', self.test_CEF_65, on_step=False, on_epoch=True, metric_attribute='test_CEF_65')
        self.log('test_CEF_70', self.test_CEF_70, on_step=False, on_epoch=True, metric_attribute='test_CEF_70')
        self.log('test_CEF_75', self.test_CEF_75, on_step=False, on_epoch=True, metric_attribute='test_CEF_75')
        self.log('test_CEF_80', self.test_CEF_80, on_step=False, on_epoch=True, metric_attribute='test_CEF_80')
        self.log('test_CEF_85', self.test_CEF_85, on_step=False, on_epoch=True, metric_attribute='test_CEF_85')
        self.log('test_CEF_90', self.test_CEF_90, on_step=False, on_epoch=True, metric_attribute='test_CEF_90')
        self.log('test_CEF_95', self.test_CEF_95, on_step=False, on_epoch=True, metric_attribute='test_CEF_95')
        #self.log('fp', self.stat_scores[1], on_step=False, on_epoch=True)
        return outputs

    def predict_step(self, pred_batch, batch_idx):
        local_batch_list = []
        local_inputs, local_labels = pred_batch
        for lb in local_inputs:
            local_batch_list.append(lb.to(dtype=torch.float))
        pred = self(local_batch_list)
        return pred

    def train_dataloader(self):
        train_data = ThunderCastDataset(self.training_files, params)
        train_loader = DataLoader(train_data, shuffle=True, num_workers=3, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_data = ThunderCastDataset(self.validation_files, params)
        val_loader = DataLoader(val_data, shuffle=False, num_workers=3, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_data = ThunderCastDataset(self.testing_files, params)
        test_loader = DataLoader(test_data, shuffle=False, num_workers=3, batch_size=self.batch_size)
        return test_loader

def get_radar_list(dts, rootradardir):
    radar_file_list = []
    for dt in dts:
        radardir = dt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
        ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
        rad_end = dt + timedelta(minutes=5)
        radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(radardir, '*.netcdf'))}
        five_min_radfilelist = [radfiles[x] for x in radfiles if dt <= x <= rad_end]
        radar_file_list.append(five_min_radfilelist[0])
    return radar_file_list

def get_radar_dict(dts, rootradardir):
    radar_file_dict = {}
    for dt in dts:
        radardir = dt.strftime(os.path.join(rootradardir, '%Y/%Y%m%d/Reflectivity_-10C/00.50/'))
        ###Grab radar files within 5 minutes of the ABI file and choose the first one to resample and plot.
        rad_end = dt + timedelta(minutes=5)
        radfiles = {datetime.strptime(x.split('/')[-1], '%Y%m%d-%H%M%S.netcdf'): x for x in
                    glob(os.path.join(radardir, '*.netcdf'))}
        five_min_radfilelist = [radfiles[x] for x in radfiles if dt <= x <= rad_end]
        radar_file_dict[dt] = five_min_radfilelist[0]
    return radar_file_dict

def get_radar(radar_file, sat_grid):
    dataset_rad = Dataset(radar_file, 'r')
    nlat = len(dataset_rad.dimensions['Lat'])
    nlon = len(dataset_rad.dimensions['Lon'])
    NW_lat = getattr(dataset_rad, 'Latitude')
    NW_lon = getattr(dataset_rad, 'Longitude')
    dy = getattr(dataset_rad, 'LonGridSpacing')
    dx = getattr(dataset_rad, 'LatGridSpacing')
    radardataset = dataset_rad.variables['Reflectivity_-10C'][:, :]
    dataset_rad.close()

    print('remapping radar')
    ###Remap radar data to same domain as satellite data
    radar_lons = np.zeros((nlat, nlon))
    radar_lats = np.zeros((nlat, nlon))
    for ii in range(nlat):
        radar_lats[ii, :] = NW_lat - ii * dy
    for ii in range(nlon):
        radar_lons[:, ii] = NW_lon + ii * dx
    radar_grid = pr.geometry.GridDefinition(lons=radar_lons, lats=radar_lats)

    remapped_radar = pr.kd_tree.resample_nearest(radar_grid, radardataset, sat_grid, radius_of_influence=10000,
                                                 fill_value=-999)
    print('remap complete')
    return remapped_radar

def make_subplot(scn, preds, center_x_idx, center_y_idx, hs, radar_file, c, timestamp, i, j, fig, axes, geoproj, straight):
    state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                 scale='50m', facecolor='none')
    CVD_cmap_all = matplotlib.colormaps.get_cmap('pyart_HomeyerRainbow')
    colors1 = CVD_cmap_all(np.linspace(0, 0.25, 98))
    colors2 = CVD_cmap_all(np.linspace(0.45, 1.0, 158))
    cmap_colors = np.vstack((colors1, colors2))
    CVD_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap_colors)

    ymin_index = int(center_y_idx + hs)
    ymax_index = int(center_y_idx - hs)
    xmin_index = int(center_x_idx - hs)
    xmax_index = int(center_x_idx + hs)
    x_coords, y_coords = scn['C05'].attrs['area'].get_proj_vectors()
    xmin = x_coords[xmin_index]
    xmax = x_coords[xmax_index]
    ymin = y_coords[ymin_index]
    ymax = y_coords[ymax_index]
    cropped_scn = scn.crop(xy_bbox=[xmin, ymin, xmax, ymax])

    #####Get one km
    C05vals = cropped_scn['C05'].values
    C13vals = cropped_scn['C13'].values
    C02vals = cropped_scn['C02'].values

    x1km = cropped_scn['C05']['x']
    y1km = cropped_scn['C05']['y']
    xx1km, yy1km = np.meshgrid(x1km, y1km)
    xhalfkm = cropped_scn['C02']['x']
    yhalfkm = cropped_scn['C02']['y']
    xxhalfkm, yyhalfkm = np.meshgrid(xhalfkm, yhalfkm)

    print(C05vals.shape, len(xx1km), len(yy1km))
    sat_grid = cropped_scn['C05'].attrs['area']

    # #Get radar
    radar_data = get_radar(radar_file, sat_grid)
    print('Max radar', np.nanmax(radar_data))

    #POINT
    x_idx, y_idx = cropped_scn['C05'].attrs['area'].get_array_indices_from_lonlat(-89.407, 43.071) #madison
    x_coords, y_coords = cropped_scn['C05'].attrs['area'].get_proj_vectors()
    x_point = x_coords[x_idx]
    y_point = y_coords[y_idx]


    xmin = np.min(x1km);
    xmax = np.max(x1km);
    ymin = np.min(y1km);
    ymax = np.max(y1km)
    axes[i, j].set_extent([xmin, xmax, ymin, ymax], crs=geoproj)
    axes[i, j].set_aspect('equal')

    if c == 'C05':
        pred_img = axes[i, j].pcolormesh(xx1km, yy1km, preds, cmap=matplotlib.colormaps['plasma'],vmin=0.1, vmax=1.0)
    elif c == 'C13':
        pred_img = axes[i, j].pcolormesh(xx1km, yy1km, preds, cmap=matplotlib.colormaps['plasma'],vmin=0.1, vmax=1.0)  # , alpha=0.9)
    else:
        pred_img = axes[i, j].pcolormesh(xxhalfkm, yyhalfkm, zoom(preds, 2, order=0), cmap=matplotlib.colormaps['plasma'],vmin=0.05, vmax=1.0)
    axes[i, j].set_title((' ').join([timestamp, 'UTC']), fontsize=16)
    axes[i, j].add_feature(cfeature.BORDERS)
    axes[i, j].add_feature(state_borders, edgecolor='chartreuse')
    axes[i, j].plot(x_point, y_point, marker='*', color='aqua', markersize=12)
    return pred_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes a datetime and gets prediction images.')
    parser.add_argument('-a', '--abidir',
                        help='Directory to ABI data. Default = /arcdata/goes/grb/goes16/Y/Y_m_d_j/abi/L1b/RadC/',
                        default=['/arcdata/goes/grb/goes16'], type=str, nargs=1)
    parser.add_argument('-r', '--radardir', help='Root directory with radar netcdfs.',
                        default=['/apollo/grain/saved_ci_ltg_data/radar'],
                        nargs=1, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/predictions/'])
    parser.add_argument('-DTstart', '--DT', help='A datetime in %Y-%m-%d-%H-%M',
                        type=str, nargs=1)
    parser.add_argument('-m', '--model', help='Model checkpoint to use.', type=str, nargs=1,
                        default=['/ships19/grain/convective_init/models/TC_CHs_02_05_13_15/thundercast-ckpt-epoch=27-val_loss_epoch=0.4490.ckpt']) #Night_Trial1/thundercast-ckpt-epoch=15-val_loss_epoch=0.4614.ckpt'
    parser.add_argument('-t', '--deltat', help='Time between subplots.',
                        type=int, nargs=1)
    parser.add_argument('-np', '--numplots', help='Number of subplots subplots. (4 or 6)',
                        type=int, nargs=1, default=[6])
    parser.add_argument('-lat', '--center_lat', help='Center latitude for prediction',
                        type=str, nargs=1)
    parser.add_argument('-lon', '--center_lon', help='Center longitude for prediction',
                        type=str, nargs=1)
    parser.add_argument('-hs', '--hs', help='Half size for prediction area indices',
                        type=int, nargs=1, default=[127])
    parser.add_argument('-c', '--c', help='Channel to plot',
                        type=str, nargs=1, default=['C02'])
    parser.add_argument('-f', '--full_us', help='If you want the full US prediction.', action="store_true")

    args = parser.parse_args()

    rootoutdir = args.outdir[0]
    rootdatadir = args.abidir[0]
    rootradardir = args.radardir[0]
    model_checkpoint_path = args.model[0]
    time_delta = args.deltat[0]
    date = args.DT[0]
    center_lon = float(args.center_lon[0])
    center_lat = float(args.center_lat[0])
    hs1km = args.hs[0]
    model_name = model_checkpoint_path.split('/')[5]
    rootpreddir = rootoutdir + model_name + '/' + date[:10]
    numplots = args.numplots[0]
    straight = False
    if numplots % 2 != 0:
        if numplots % 3 == 0:
            numcols = int(numplots/3)
            numrows = int(numplots/numcols)
        print('Number of subplots should be 4 or 6')
    elif straight:
        numcols = numplots
        numrows = 1
    else:
        numcols = int(numplots/2)
        numrows = int(numplots/numcols)
    print(numcols, numrows)
    c = args.c[0]
    full_US = args.full_us

    mkdir_p(rootpreddir)

    t_start = time()
    print('Time at the start of the program:', datetime.now())
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
    # All
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
    # vars = {'goes16_half_km': ['C02'], 'goes16_one_km': ['C05'], 'goes16_two_km': ['C13', 'C15'],
    #         'target': {'MRMS_one_km': ['reflectivity_60min_neg10C_max']}}  # in format group:[variable(s)]

    vars = {'goes16_half_km':['C02'], 'goes16_one_km':['C05'], 'goes16_two_km':['C13', 'C15'], 'target':{'MRMS_one_km':['reflectivity_60min_neg10C_max']}} #in format group:[variable(s)]



    inputs = []
    for k in vars.keys():
        if k != 'target':
            inputs.append(vars[k])

    print('date', date)
    topdt = datetime.strptime(date, '%Y-%m-%d-%H-%M')
    dts = []
    for i in range(numplots):
        min = i*time_delta
        dt = topdt + timedelta(minutes=min)
        dts.append(dt)
    print(np.array(dts))
    hrs = [x.hour for x in dts]
    hr_min = {}
    for hr in hrs:
        mins = []
        for x in dts:
            if x.hour == hr:
                mins.append(x.minute)
        hr_min[str(hr)] = mins
    satdir = topdt.strftime(rootdatadir + '/%Y/%Y_%m_%d_%j/abi/L1b/RadC/')
    netcdfpattern = satdir + '*C02*'
    satfiles = {datetime.strptime(os.path.basename(x).split('_')[3][1:-1], '%Y%j%H%M%S'): x for x in
                glob(netcdfpattern)}
    sat_netcdfs = []
    new_dts = []
    for x in satfiles:
        if str(x.hour) in hr_min.keys():
            mins = hr_min[str(x.hour)]
            if x.minute in mins:
                sat_netcdfs.append(satfiles[x])
                if x not in new_dts:
                    new_dts.append(x)

    abifile_dict = get_abifile_dict(rootdatadir, new_dts)
    radar_file_dict = get_radar_dict(new_dts, rootradardir)
    dts = np.array(np.sort(new_dts)).reshape((numrows, numcols))
    goesnc = Dataset('/home/sortland/ci_ltg/proj_files/GOES_East.nc', 'r')
    gip = goesnc.variables['goes_imager_projection']
    geoproj = ccrs.Geostationary(central_longitude=gip.longitude_of_projection_origin, \
                                 sweep_axis=gip.sweep_angle_axis, \
                                 satellite_height=gip.perspective_point_height)

    if straight:
        fig, axes = plt.subplots(numrows, numcols, figsize=(20, 5), subplot_kw={'projection': geoproj})
    elif numplots == 9:
        fig, axes = plt.subplots(numrows, numcols, figsize=(15, 15), subplot_kw={'projection': geoproj})
    else:
        fig, axes = plt.subplots(numrows, numcols, figsize=(15, 10), subplot_kw={'projection': geoproj})

    print(numcols, numrows)
    for i in range(numrows):
        for j in range(numcols):
            print('i, j:', i, j)
            dt = dts[i,j]
            timestamp = str(dt)
            abi_files = abifile_dict[dt]
            scn, bad_data = get_satpy_scene(abi_files)
            center_x_idx, center_y_idx = scn['C05'].attrs['area'].get_array_indices_from_lonlat(center_lon, center_lat)  # get center lat/lon
            inputs = format_inputs(scn, norm_vals, vars, center_x_idx, center_y_idx, hs1km, full_US)
            inputs = [inputs]
            radar_file = radar_file_dict[dt]
            model = UNet2D_LitModel.load_from_checkpoint(model_checkpoint_path)
            model.eval()
            with torch.no_grad():
                raw_preds = model(inputs)
            preds = np.zeros((raw_preds[0].size(dim=-2), raw_preds[0].size(dim=-1)))
            preds[:, :] = raw_preds[0, 1, :, :]
            pred_img = make_subplot(scn, preds, center_x_idx, center_y_idx, hs1km, radar_file, c, timestamp, i, j, fig, axes, geoproj, straight)

    cbaxes2 = fig.add_axes([0.125, .06, 0.25, 0.03])   # left bottom width height
    cbar2 = plt.colorbar(pred_img, orientation='horizontal', pad=0.01, extend='min', drawedges=0,
                         ticks=np.arange(0.2, 1.1, 0.2), cax=cbaxes2)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label('0-60 min. Thunderstorm Probability', fontsize=14)
    cbar2.set_alpha(1.0)
    cbar2.draw_all()
    time_nm = timestamp[11:13] + '_' + timestamp[14:16]
    center_lat_nm = str(int(center_lat * 100))
    center_lon_nm = str(int(center_lon * 100))
    figure_path = rootpreddir + '/subplots_' + time_nm + '_' + center_lat_nm + '_' + center_lon_nm + '_' + str(
        hs1km) + '.png'
    print(figure_path)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)
    print('Predictions complete')