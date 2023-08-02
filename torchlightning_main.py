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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import kornia.augmentation as K
from torchmetrics import Metric
from PL_Metrics import CriticalSuccessIndex as CSI
from PL_Metrics import POD, FAR, SR, Accuracy, Specificity, F1Score, Recall, Precision, CEFs, NoRes, FSS, FSS_random, FSS_uniform

#This file uses PyTorch-Lightning to train or test ThunderCast

#These are for accessing the data on a remote server
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def mkdir_p(path):
    """
    Create a directory
    """
    try:
        os.makedirs(path)
    except (OSError,IOError) as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

class ThunderCastDataset(torch.utils.data.Dataset):
    """
    Creates a dataset instance. The inputs are read using the given "load function"
    """
    def __init__(self, files, params):
        # get input and out fields
        self.files = files
        self.load_fcn = params['load_fcn']
        self.params = params
        print(f"ThunderCastDataset: fields={self.params['vars']} num examples = {len(self.files)}\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x, y = self.load_fcn(self.files[index], self.params)
        return x, y

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
            y = y.float()
            # y = y[None]

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

def load_pt(file, params):
    """
    This function loads data from pt tensor files.
    """
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

        self.test_F1score = F1Score(threshold=0.2)
        self.test_accuracy = Accuracy(threshold=0.2)
        self.test_recall = Recall(threshold=0.2)
        self.test_specificity = Specificity(threshold=0.2)
        # self.test_FAR = FAR()
        self.test_precision = Precision(threshold=0.2)
        # self.test_NoRes = NoRes()

        # self.test_FSS_random = FSS_random()
        # self.test_FSS_uniform = FSS_uniform()
        # self.test_FSS_P1 = FSS(threshold=0.1, window=1)
        # self.test_FSS_P2 = FSS(threshold=0.2, window=1)
        # self.test_FSS_P3 = FSS(threshold=0.3, window=1)
        # self.test_FSS_P4 = FSS(threshold=0.4, window=1)
        # self.test_FSS_P5 = FSS(threshold=0.5, window=1)
        # self.test_FSS_P6 = FSS(threshold=0.6, window=1)
        # self.test_FSS_P7 = FSS(threshold=0.7, window=1)
        # self.test_FSS_P8 = FSS(threshold=0.8, window=1)
        # self.test_FSS_P9 = FSS(threshold=0.9, window=1)
        #If you don't need to make the attribute diagram or performance diagram you can delete these extra stats and their logging.
        #*There may be a better way to do this.
        # #CEFs
        # self.test_CEF_0 = CEFs(threshold1=0, threshold2=0.05)
        # self.test_CEF_5 = CEFs(threshold1=0.05, threshold2=0.10)
        # self.test_CEF_10 = CEFs(threshold1=0.10, threshold2=0.15)
        # self.test_CEF_15 = CEFs(threshold1=0.15, threshold2=0.20)
        # self.test_CEF_20 = CEFs(threshold1=0.20, threshold2=0.25)
        # self.test_CEF_25 = CEFs(threshold1=0.25, threshold2=0.30)
        # self.test_CEF_30 = CEFs(threshold1=0.30, threshold2=0.35)
        # self.test_CEF_35 = CEFs(threshold1=0.35, threshold2=0.40)
        # self.test_CEF_40 = CEFs(threshold1=0.40, threshold2=0.45)
        # self.test_CEF_45 = CEFs(threshold1=0.45, threshold2=0.50)
        # self.test_CEF_50 = CEFs(threshold1=0.50, threshold2=0.55)
        # self.test_CEF_55 = CEFs(threshold1=0.55, threshold2=0.60)
        # self.test_CEF_60 = CEFs(threshold1=0.60, threshold2=0.65)
        # self.test_CEF_65 = CEFs(threshold1=0.65, threshold2=0.70)
        # self.test_CEF_70 = CEFs(threshold1=0.70, threshold2=0.75)
        # self.test_CEF_75 = CEFs(threshold1=0.75, threshold2=0.80)
        # self.test_CEF_80 = CEFs(threshold1=0.80, threshold2=0.85)
        # self.test_CEF_85 = CEFs(threshold1=0.85, threshold2=0.90)
        # self.test_CEF_90 = CEFs(threshold1=0.90, threshold2=0.95)
        # self.test_CEF_95 = CEFs(threshold1=0.95, threshold2=1.0)
        # #POD
        # self.test_POD_5 = POD(threshold= 0.05)
        # self.test_POD_10 = POD(threshold= 0.1)
        # self.test_POD_15 = POD(threshold=0.15)
        # self.test_POD_20 = POD(threshold= 0.2)
        # self.test_POD_25 = POD(threshold=0.25)
        # self.test_POD_30 = POD(threshold= 0.3)
        # self.test_POD_35 = POD(threshold=0.35)
        # self.test_POD_40 = POD(threshold= 0.4)
        # self.test_POD_45 = POD(threshold=0.45)
        # self.test_POD_50 = POD()
        # self.test_POD_55 = POD(threshold=0.55)
        # self.test_POD_60 = POD(threshold= 0.6)
        # self.test_POD_65 = POD(threshold=0.65)
        # self.test_POD_70 = POD(threshold= 0.7)
        # self.test_POD_75 = POD(threshold=0.75)
        # self.test_POD_80 = POD(threshold= 0.8)
        # self.test_POD_85 = POD(threshold=0.85)
        # self.test_POD_90 = POD(threshold= 0.9)
        # self.test_POD_95 = POD(threshold= 0.95)
        # #CSI
        # self.test_CSI_5 = CSI(threshold= 0.05)
        # self.test_CSI_10 = CSI(threshold= 0.1)
        # self.test_CSI_15 = CSI(threshold=0.15)
        self.test_CSI_20 = CSI(threshold= 0.2)
        # self.test_CSI_25 = CSI(threshold=0.25)
        # self.test_CSI_30 = CSI(threshold= 0.3)
        # self.test_CSI_35 = CSI(threshold=0.35)
        # self.test_CSI_40 = CSI(threshold= 0.4)
        # self.test_CSI_45 = CSI(threshold=0.45)
        # self.test_CSI_50 = CSI()
        # self.test_CSI_55 = CSI(threshold=0.55)
        # self.test_CSI_60 = CSI(threshold= 0.6)
        # self.test_CSI_65 = CSI(threshold=0.65)
        # self.test_CSI_70 = CSI(threshold= 0.7)
        # self.test_CSI_75 = CSI(threshold=0.75)
        # self.test_CSI_80 = CSI(threshold= 0.8)
        # self.test_CSI_85 = CSI(threshold=0.85)
        # self.test_CSI_90 = CSI(threshold= 0.9)
        # self.test_CSI_95 = CSI(threshold= 0.95)
        # #SR
        # self.test_SR_5 = SR(threshold=0.05)
        # self.test_SR_10 = SR(threshold=0.1)
        # self.test_SR_15 = SR(threshold=0.15)
        # self.test_SR_20 = SR(threshold=0.2)
        # self.test_SR_25 = SR(threshold=0.25)
        # self.test_SR_30 = SR(threshold=0.3)
        # self.test_SR_35 = SR(threshold=0.35)
        # self.test_SR_40 = SR(threshold=0.4)
        # self.test_SR_45 = SR(threshold=0.45)
        # self.test_SR_50 = SR()
        # self.test_SR_55 = SR(threshold=0.55)
        # self.test_SR_60 = SR(threshold=0.6)
        # self.test_SR_65 = SR(threshold=0.65)
        # self.test_SR_70 = SR(threshold=0.7)
        # self.test_SR_75 = SR(threshold=0.75)
        # self.test_SR_80 = SR(threshold=0.8)
        # self.test_SR_85 = SR(threshold=0.85)
        # self.test_SR_90 = SR(threshold=0.9)
        # self.test_SR_95 = SR(threshold=0.95)
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

        self.log('val_loss', loss, on_step=True, on_epoch=True, metric_attribute='val_loss', sync_dist=True)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True, metric_attribute='val_specificity', sync_dist=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, metric_attribute='val_accuracy', sync_dist = True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, metric_attribute='val_precision', sync_dist=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, metric_attribute='val_recall', sync_dist=True)
        self.log('val_F1score', self.val_F1score, on_step=False, on_epoch=True, metric_attribute='val_F1score', sync_dist=True)
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
        # self.test_FAR.update(outputs, local_labels)
        self.test_specificity.update(outputs, local_labels)
        self.test_precision.update(outputs, local_labels)
        # self.test_NoRes.update(outputs, local_labels)

        # self.test_FSS_random.update(local_labels)
        # self.test_FSS_uniform.update(local_labels)
        # self.test_FSS_P1.update(outputs, local_labels)
        # self.test_FSS_P2.update(outputs, local_labels)
        # self.test_FSS_P3.update(outputs, local_labels)
        # self.test_FSS_P4.update(outputs, local_labels)
        # self.test_FSS_P5.update(outputs, local_labels)
        # self.test_FSS_P6.update(outputs, local_labels)
        # self.test_FSS_P7.update(outputs, local_labels)
        # self.test_FSS_P8.update(outputs, local_labels)
        # self.test_FSS_P9.update(outputs, local_labels)

        # self.test_POD_10.update(outputs, local_labels)
        # self.test_POD_20.update(outputs, local_labels)
        # self.test_POD_30.update(outputs, local_labels)
        # self.test_POD_40.update(outputs, local_labels)
        # self.test_POD_50.update(outputs, local_labels)
        # self.test_POD_60.update(outputs, local_labels)
        # self.test_POD_70.update(outputs, local_labels)
        # self.test_POD_80.update(outputs, local_labels)
        # self.test_POD_90.update(outputs, local_labels)
        # self.test_SR_10.update(outputs, local_labels)
        # self.test_SR_20.update(outputs, local_labels)
        # self.test_SR_30.update(outputs, local_labels)
        # self.test_SR_40.update(outputs, local_labels)
        # self.test_SR_50.update(outputs, local_labels)
        # self.test_SR_60.update(outputs, local_labels)
        # self.test_SR_70.update(outputs, local_labels)
        # self.test_SR_80.update(outputs, local_labels)
        # self.test_SR_90.update(outputs, local_labels)
        # self.test_CSI_10.update(outputs, local_labels)
        self.test_CSI_20.update(outputs, local_labels)
        # self.test_CSI_30.update(outputs, local_labels)
        # self.test_CSI_40.update(outputs, local_labels)
        # self.test_CSI_50.update(outputs, local_labels)
        # self.test_CSI_60.update(outputs, local_labels)
        # self.test_CSI_70.update(outputs, local_labels)
        # self.test_CSI_80.update(outputs, local_labels)
        # self.test_CSI_90.update(outputs, local_labels)
        # self.test_CEF_0.update(outputs, local_labels)
        # self.test_CEF_5.update(outputs, local_labels)
        # self.test_CEF_10.update(outputs, local_labels)
        # self.test_CEF_15.update(outputs, local_labels)
        # self.test_CEF_20.update(outputs, local_labels)
        # self.test_CEF_25.update(outputs, local_labels)
        # self.test_CEF_30.update(outputs, local_labels)
        # self.test_CEF_35.update(outputs, local_labels)
        # self.test_CEF_40.update(outputs, local_labels)
        # self.test_CEF_45.update(outputs, local_labels)
        # self.test_CEF_50.update(outputs, local_labels)
        # self.test_CEF_55.update(outputs, local_labels)
        # self.test_CEF_60.update(outputs, local_labels)
        # self.test_CEF_65.update(outputs, local_labels)
        # self.test_CEF_70.update(outputs, local_labels)
        # self.test_CEF_75.update(outputs, local_labels)
        # self.test_CEF_80.update(outputs, local_labels)
        # self.test_CEF_85.update(outputs, local_labels)
        # self.test_CEF_90.update(outputs, local_labels)
        # self.test_CEF_95.update(outputs, local_labels)
        #self.stat_scores.update(outputs, local_labels)

        self.log('test_loss', loss, on_step=True, on_epoch=True, metric_attribute='test_loss', sync_dist=True)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True,metric_attribute='test_accuracy', sync_dist=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, metric_attribute='test_recall', sync_dist=True)
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True,metric_attribute='test_specificity', sync_dist=True)
        self.log('test_F1score', self.test_F1score, on_step=False, on_epoch=True, metric_attribute='test_F1score', sync_dist=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, metric_attribute='test_precision', sync_dist=True)
        # self.log('test_NoRes', self.test_NoRes, on_step=False, on_epoch=True, metric_attribute='test_NoRes', sync_dist=True)
        # self.log('test_FSS_random', self.test_FSS_random, on_step=False, on_epoch=True, metric_attribute='test_FSS_random', sync_dist=True)
        # self.log('test_FSS_uniform', self.test_FSS_uniform, on_step=False, on_epoch=True, metric_attribute='test_FSS_uniform', sync_dist=True)

        # self.log('test_FAR', self.test_FAR, on_step=False, on_epoch=True, metric_attribute='test_FAR', sync_dist=True)
        # self.log('test_CSI_10', self.test_CSI_10, on_step=False, on_epoch=True, metric_attribute='test_CSI_10', sync_dist=True)
        self.log('test_CSI_20', self.test_CSI_20, on_step=False, on_epoch=True, metric_attribute='test_CSI_20', sync_dist=True)
        # self.log('test_CSI_30', self.test_CSI_30, on_step=False, on_epoch=True, metric_attribute='test_CSI_30', sync_dist=True)
        # self.log('test_CSI_40', self.test_CSI_40, on_step=False, on_epoch=True, metric_attribute='test_CSI_40', sync_dist=True)
        # self.log('test_CSI_50', self.test_CSI_50, on_step=False, on_epoch=True, metric_attribute='test_CSI_50', sync_dist=True)
        # self.log('test_CSI_60', self.test_CSI_60, on_step=False, on_epoch=True, metric_attribute='test_CSI_60', sync_dist=True)
        # self.log('test_CSI_70', self.test_CSI_70, on_step=False, on_epoch=True, metric_attribute='test_CSI_70', sync_dist=True)
        # self.log('test_CSI_80', self.test_CSI_80, on_step=False, on_epoch=True, metric_attribute='test_CSI_80', sync_dist=True)
        # self.log('test_CSI_90', self.test_CSI_90, on_step=False, on_epoch=True, metric_attribute='test_CSI_90', sync_dist=True)
        # self.log('test_SR_10', self.test_SR_10, on_step=False, on_epoch=True, metric_attribute='test_SR_10', sync_dist=True)
        # self.log('test_SR_20', self.test_SR_20, on_step=False, on_epoch=True, metric_attribute='test_SR_20', sync_dist=True)
        # self.log('test_SR_30', self.test_SR_30, on_step=False, on_epoch=True, metric_attribute='test_SR_30', sync_dist=True)
        # self.log('test_SR_40', self.test_SR_40, on_step=False, on_epoch=True, metric_attribute='test_SR_40', sync_dist=True)
        # self.log('test_SR_50', self.test_SR_50, on_step=False, on_epoch=True, metric_attribute='test_SR_50', sync_dist=True)
        # self.log('test_SR_60', self.test_SR_60, on_step=False, on_epoch=True, metric_attribute='test_SR_60', sync_dist=True)
        # self.log('test_SR_70', self.test_SR_70, on_step=False, on_epoch=True, metric_attribute='test_SR_70', sync_dist=True)
        # self.log('test_SR_80', self.test_SR_80, on_step=False, on_epoch=True, metric_attribute='test_SR_80', sync_dist=True)
        # self.log('test_SR_90', self.test_SR_90, on_step=False, on_epoch=True, metric_attribute='test_SR_90', sync_dist=True)
        # self.log('test_POD_10', self.test_POD_10, on_step=False, on_epoch=True, metric_attribute='test_POD_10', sync_dist=True)
        # self.log('test_POD_20', self.test_POD_20, on_step=False, on_epoch=True, metric_attribute='test_POD_20', sync_dist=True)
        # self.log('test_POD_30', self.test_POD_30, on_step=False, on_epoch=True, metric_attribute='test_POD_30', sync_dist=True)
        # self.log('test_POD_40', self.test_POD_40, on_step=False, on_epoch=True, metric_attribute='test_POD_40', sync_dist=True)
        # self.log('test_POD_50', self.test_POD_50, on_step=False, on_epoch=True, metric_attribute='test_POD_50', sync_dist=True)
        # self.log('test_POD_60', self.test_POD_60, on_step=False, on_epoch=True, metric_attribute='test_POD_60', sync_dist=True)
        # self.log('test_POD_70', self.test_POD_70, on_step=False, on_epoch=True, metric_attribute='test_POD_70', sync_dist=True)
        # self.log('test_POD_80', self.test_POD_80, on_step=False, on_epoch=True, metric_attribute='test_POD_80', sync_dist=True)
        # self.log('test_POD_90', self.test_POD_90, on_step=False, on_epoch=True, metric_attribute='test_POD_90', sync_dist=True)
        # self.log('test_CEF_0', self.test_CEF_0, on_step=False, on_epoch=True, metric_attribute='test_CEF_0', sync_dist=True)
        # self.log('test_CEF_5', self.test_CEF_5, on_step=False, on_epoch=True, metric_attribute='test_CEF_5', sync_dist=True)
        # self.log('test_CEF_10', self.test_CEF_10, on_step=False, on_epoch=True, metric_attribute='test_CEF_10', sync_dist=True)
        # self.log('test_CEF_15', self.test_CEF_15, on_step=False, on_epoch=True, metric_attribute='test_CEF_15', sync_dist=True)
        # self.log('test_CEF_20', self.test_CEF_20, on_step=False, on_epoch=True, metric_attribute='test_CEF_20', sync_dist=True)
        # self.log('test_CEF_25', self.test_CEF_25, on_step=False, on_epoch=True, metric_attribute='test_CEF_25', sync_dist=True)
        # self.log('test_CEF_30', self.test_CEF_30, on_step=False, on_epoch=True, metric_attribute='test_CEF_30', sync_dist=True)
        # self.log('test_CEF_35', self.test_CEF_35, on_step=False, on_epoch=True, metric_attribute='test_CEF_35', sync_dist=True)
        # self.log('test_CEF_40', self.test_CEF_40, on_step=False, on_epoch=True, metric_attribute='test_CEF_40', sync_dist=True)
        # self.log('test_CEF_45', self.test_CEF_45, on_step=False, on_epoch=True, metric_attribute='test_CEF_45', sync_dist=True)
        # self.log('test_CEF_50', self.test_CEF_50, on_step=False, on_epoch=True, metric_attribute='test_CEF_50', sync_dist=True)
        # self.log('test_CEF_55', self.test_CEF_55, on_step=False, on_epoch=True, metric_attribute='test_CEF_55', sync_dist=True)
        # self.log('test_CEF_60', self.test_CEF_60, on_step=False, on_epoch=True, metric_attribute='test_CEF_60', sync_dist=True)
        # self.log('test_CEF_65', self.test_CEF_65, on_step=False, on_epoch=True, metric_attribute='test_CEF_65', sync_dist=True)
        # self.log('test_CEF_70', self.test_CEF_70, on_step=False, on_epoch=True, metric_attribute='test_CEF_70', sync_dist=True)
        # self.log('test_CEF_75', self.test_CEF_75, on_step=False, on_epoch=True, metric_attribute='test_CEF_75', sync_dist=True)
        # self.log('test_CEF_80', self.test_CEF_80, on_step=False, on_epoch=True, metric_attribute='test_CEF_80', sync_dist=True)
        # self.log('test_CEF_85', self.test_CEF_85, on_step=False, on_epoch=True, metric_attribute='test_CEF_85', sync_dist=True)
        # self.log('test_CEF_90', self.test_CEF_90, on_step=False, on_epoch=True, metric_attribute='test_CEF_90', sync_dist=True)
        # self.log('test_CEF_95', self.test_CEF_95, on_step=False, on_epoch=True, metric_attribute='test_CEF_95', sync_dist=True)
        # self.log('test_FSS_P1', self.test_FSS_P1, on_step=False, on_epoch=True, metric_attribute='test_FSS_P1', sync_dist=True)
        # self.log('test_FSS_P2', self.test_FSS_P2, on_step=False, on_epoch=True, metric_attribute='test_FSS_P2', sync_dist=True)
        # self.log('test_FSS_P3', self.test_FSS_P3, on_step=False, on_epoch=True, metric_attribute='test_FSS_P3', sync_dist=True)
        # self.log('test_FSS_P4', self.test_FSS_P4, on_step=False, on_epoch=True, metric_attribute='test_FSS_P4', sync_dist=True)
        # self.log('test_FSS_P5', self.test_FSS_P5, on_step=False, on_epoch=True, metric_attribute='test_FSS_P5', sync_dist=True)
        # self.log('test_FSS_P6', self.test_FSS_P6, on_step=False, on_epoch=True, metric_attribute='test_FSS_P6', sync_dist=True)
        # self.log('test_FSS_P7', self.test_FSS_P7, on_step=False, on_epoch=True, metric_attribute='test_FSS_P7', sync_dist=True)
        # self.log('test_FSS_P8', self.test_FSS_P8, on_step=False, on_epoch=True, metric_attribute='test_FSS_P8', sync_dist=True)
        # self.log('test_FSS_P9', self.test_FSS_P9, on_step=False, on_epoch=True, metric_attribute='test_FSS_P9', sync_dist=True)
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


#Main program
if __name__ == "__main__":
    t_start = time()
    print('Time at the start of the program:', datetime.now())
    #____________________________User Parameters_____________________________________________
    # Day only
    # norm_vals = {'C02': {'mean': 23.08378, 'std': 13.866257},
    #              'C05': {'mean': 16.540283, 'std': 6.8535457},
    #              'C13': {"mean": 268.9491, "std": 20.004034},
    #              'C11': {"mean": 266.74457, "std": 19.122293},
    #              'C14': {'mean': 267.59595, 'std': 19.97203},
    #              'C15': {'mean': 264.79877, 'std': 19.170725},
    #              'solar_azimuth': {'mean': 201.90517, 'std': 1.8463212},
    #              'solar_zenith': {'mean': 48.27021, 'std': 1.021574}}
    # All
    norm_vals = {'C02': {'mean': 13.338485, 'std': 7.5519395},
                 'C05': {'mean': 9.244341, 'std': 3.830976},
                 'C13': {"mean": 263.5579, "std": 19.133537},
                 'C11': {"mean": 261.74683, "std": 18.389097},
                 'C14': {'mean': 262.35043, 'std': 19.177515},
                 'C15': {'mean': 260.04007, 'std': 18.560625},
                 'solar_azimuth': {'mean': 194.34933, 'std': 2.9718854},
                 'solar_zenith': {'mean': 78.79489, 'std': 1.0483047}}

    class_weights = torch.tensor([0.05,0.95])
    vars = {'goes16_half_km':['C02'], 'goes16_one_km':['C05'], 'goes16_two_km':['C13', 'C15'], 'target':{'MRMS_one_km':['reflectivity_60min_neg10C_max']}} #in format group:[variable(s)]
    number_inputs = 4
    if 'goes16_half_km' in vars.keys():
        maxpool_end = True
    else:
        maxpool_end = False

    """
    Set up directories. Can get these with glob or reading in a file.
    """
    training_file_path = '/ships22/grain/sortland/pt_tensors/train_files/inputs/*'
    validation_file_path = '/ships22/grain/sortland/pt_tensors/val_files/inputs/*'
    testing_file_path = '/ships22/grain/sortland/pt_tensors/test_files_7km/inputs/*'
    training_files = glob(training_file_path)
    validation_files = glob(validation_file_path)
    testing_files = glob(testing_file_path)
    # training_file_path = '/ships19/grain/convective_init/data_lists/training/sublists/night_train_files.txt'
    # validation_file_path = '/ships19/grain/convective_init/data_lists/validation/sublists/night_val_files.txt'
    #testing_file_path = '/ships22/grain/sortland/data_lists/testing/test_files.txt'
    print('testing file', testing_file_path)
    # with open(training_file_path, 'r') as data_files:
    #     training_files = data_files.read().splitlines()
    # with open(validation_file_path, 'r') as data_files:
    #     validation_files = data_files.read().splitlines()
    #with open(testing_file_path, 'r') as data_files:
    #    testing_files = data_files.read().splitlines()

    #### Code below can be used for testing code without waiting for full dataset.
    # training_files = training_files[:1000]
    # validation_files = validation_files[:1000]
    #testing_files = testing_files[:64]
    # training_files = training_files[::2]
    # validation_files = validation_files[::2]
    # testing_files = testing_files[::2]
    ####

    print('length of training, val, and testing files:', len(training_files), len(validation_files), len(testing_files))
    outdir = '/ships22/grain/sortland/models/TC_CHs_02_05_13_15/' #name the model
    os.makedirs(outdir, exist_ok=True)
    #Specify path for starting training from a checkpoint or testing the model with a certain model checkpoint
    model_checkpoint_path = outdir + 'thundercast-ckpt-epoch=29-val_loss_epoch=0.4488.ckpt'#None #model path if using a model checkpoint

    load_fcn = load_pt

    """
    Set parameters for Training/Validation/Testing.
    If testing, make sure to set ngpu to 1 and num_nodes to 1 since it is not recommended to use more than that at 
    https://pytorch-lightning.readthedocs.io/en/stable/common/evaluation_intermediate.html
    For training num_nodes=2, ngpu=2
    """

    params = {'ngpu':1, 'num_nodes':1, 'norm_vals':norm_vals, 'threshold':30.0, 'nepochs':30, 'nclass':2, 'vars':vars,
              'maxpool_end':maxpool_end, 'number_inputs':number_inputs, 'rlr_factor':0.1, 'rlr_patience':1, 'batch_size':32,
              'learning_rate': 1e-3, 'rlr_cooldown': 2, 'weights': class_weights, 'es_patience':4, 'num_conv_filters': 16,
              'use_model_checkpoint':True, 'load_fcn':load_fcn, 'outdir':outdir, 'test_model':True, 'train_val_model':False,
              'training_file_path':training_file_path, 'validation_file_path':validation_file_path, 'testing_file_path':testing_file_path,
              'num_train':len(training_files), 'num_val':len(validation_files), 'num_test':len(testing_files), 'training_files':training_files,
              'validation_files':validation_files, 'testing_files':testing_files}


    #Keep record of model parameters
    f = open(os.path.join(outdir, 'model_params.txt'), 'w')
    for key, val in params.items():
        f.write('%s:%s\n' % (key, val))
    f.close()

    model = UNet2D_LitModel(params)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience = params['es_patience'], mode='min')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss_epoch",
        mode="min",
        dirpath=outdir,
        filename='thundercast-ckpt-{epoch:02d}-{val_loss_epoch:.4f}',
    )

    trainer = pl.Trainer(accelerator='gpu', devices=params['ngpu'], strategy=DDPStrategy(find_unused_parameters=False),
                         default_root_dir=params['outdir'], max_epochs=params['nepochs'],
                         callbacks=[early_stop_callback, checkpoint_callback], num_nodes=params['num_nodes'],
                         log_every_n_steps=100)

    if params['train_val_model']:
        print('Time of training start:', datetime.now())
        if params['use_model_checkpoint']:
            trainer.fit(model, ckpt_path=model_checkpoint_path)
        else:
            trainer.fit(model)
        print('Best model:', checkpoint_callback.best_model_path)
        print('Time of training end:', datetime.now())
        print('Delta t for training:', time() - t_start)
    if params['test_model']:
        print('Testing Model....')
        if params['use_model_checkpoint']:
            trainer.test(model, ckpt_path=model_checkpoint_path)
        else:
            print('need to provide model checkpoint for testing- check you have set this to true')
        print('Testing Complete.')