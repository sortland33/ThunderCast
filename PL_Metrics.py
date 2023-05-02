#Author: Stephanie M. Ortland
#Metrics for evaluating the model during validation/testing
from typing import List, Optional

import torch
from torch import Tensor
from typing_extensions import Literal

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