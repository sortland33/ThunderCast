#Author: Stephanie M. Ortland
#Metrics for evaluating the model during validation/testing
from typing import List, Optional

import torch
from torch import Tensor
from typing_extensions import Literal
import torch.nn as nn
#from torchmetrics.regression import MeanSquaredError

from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)

import warnings
from enum import Enum
from torchmetrics import Metric

def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """Safe division, by preventing division by zero.
    Additionally casts to float if input is not already to secure backwards compatibility. -> from Pytorch Lightning originally
    """
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom

class CriticalSuccessIndex(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.tp, self.tp + self.fp + self.fn)

#accuracy
class Accuracy(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn)

#precision
class Precision(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.tp, self.tp + self.fp)

#recall
class Recall(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.tp, self.tp + self.fn)

#specificity
class Specificity(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.tn, self.tn + self.fp)

#F1 Score
class F1Score(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        prec = _safe_divide(self.tp, self.tp + self.fp)
        rec = _safe_divide(self.tp, self.tp + self.fn)
        num = 2*prec*rec
        denom = prec+rec
        return _safe_divide(num, denom)

#probability of detection
class POD(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.tp, self.tp + self.fn)

#false alarm ratio
class FAR(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return _safe_divide(self.fp, self.tp + self.fp)

#success ratio = 1- false alarm ratio
class SR(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        FAR = _safe_divide(self.fp, self.tp + self.fp)
        val = 1 - FAR
        return val

#heidke skill score
class HSS(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        num = 2*(self.tp*self.tn - self.fp*self.fn)
        denom = (self.tp + self.fn)*(self.fn + self.tn) + (self.tp +  self.fp)*(self.fp + self.tn)
        return _safe_divide(num, denom)

#true skill score
class TSS(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        num = self.tp*self.tn - self.fp*self.fn
        denom = (self.tp + self.fn)*(self.fp + self.tn)
        return _safe_divide(num, denom)

#Conditional Event Frequency
class CEFs(Metric):
    def __init__(self, threshold1= 0.5, threshold2= 0.6):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold1) & (preds < self.threshold2) & (target == 1.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold1) & (preds < self.threshold2) & (target == 0.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp

    def compute(self):
        return _safe_divide(self.tp, self.tp + self.fp)

#No Resolution
class NoRes(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        tp = torch.sum(torch.where((preds >= self.threshold) & (target == 1.0), 1., 0.)).long()
        tn = torch.sum(torch.where((preds < self.threshold) & (target == 0.0), 1., 0.)).long()
        fp = torch.sum(torch.where((preds >= self.threshold) & (target == 0.0), 1., 0.)).long()
        fn = torch.sum(torch.where((preds < self.threshold) & (target == 1.0), 1., 0.)).long()
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        num = self.tp + self.fn #observed yes
        denom = self.tp + self.fn + self.fp + self.tn #all observed
        return _safe_divide(num, denom)

#fraction skill score
class FSS(Metric):
    def __init__(self, threshold= 0.5, window=3):
        super().__init__()
        self.threshold = threshold
        self.window = window
        self.add_state('fss', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds[:, 1, :, :]
        batch_size = preds.shape[0]
        preds_binary = torch.where((preds >= self.threshold), 1., 0.)
        pool = nn.AvgPool2d(self.window, stride=1, padding=0)
        target_density = pool(target.to(torch.float32))
        pred_density = pool(preds_binary)
        n_density_pixels = (target_density.shape[1]*target_density.shape[2])
        O_M = target_density - pred_density
        for i in range(batch_size):
            O_M_ssum = torch.sum(O_M[i, :, :]*O_M[i, :, :])
            MSE_n = O_M_ssum/n_density_pixels
            O_n_ssum = torch.sum(target_density[i, :, :]*target_density[i, :, :])
            M_n_ssum = torch.sum(pred_density[i, :, :]*pred_density[i, :, :])
            MSE_n_ref = (O_n_ssum + M_n_ssum)/n_density_pixels
            #print(i, MSE_n, MSE_n_ref)
            if MSE_n_ref == 0:
                FSS = 1 - MSE_n
            else:
                FSS = 1 - (MSE_n/MSE_n_ref)
            self.fss.append(FSS)

    def compute(self):
        return torch.mean(self.fss)

class FSS_random(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('obs_true', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('obs_all', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor):
        obs_true = torch.sum(target).long()
        obs_all = torch.numel(target)
        self.obs_true += obs_true
        self.obs_all += obs_all

    def compute(self):
        return _safe_divide(self.obs_true, self.obs_all)

class FSS_uniform(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('obs_true', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('obs_all', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor):
        obs_true = torch.sum(target).long()
        obs_all = torch.numel(target)
        self.obs_true += obs_true
        self.obs_all += obs_all

    def compute(self):
        f_o = _safe_divide(self.obs_true, self.obs_all)
        uniform = 0.5 + (f_o/2)
        return uniform


