import numpy as np
import torch
from torch.nn import functional as ffunc


from metrics.MetricBase import Activation, MetricBase
from metrics import functional as F

from utils.UtilRegister import Metric
from utils.UtilBase import to_device

class IoU(MetricBase):

    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(MetricBase):

    __name__ = 'f_score'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(MetricBase):

    __name__ = 'oa_score'

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(MetricBase):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(MetricBase):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class MetricScore:

    def __init__(self, *args):
        self.luncher = args[0]
        self.test_loader = args[1]
        self.device = args[2]

    @staticmethod
    def _public_metric(y_true: torch.Tensor, y_pre: torch.Tensor, fun) -> tuple:

        metric = fun(y_pre, y_true)

        return metric

    @staticmethod
    def per_class_iu(hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)+1e-7)


@Metric.register('Metric')
class MetricEval(MetricScore):

    def __init__(self, *args):

        super(MetricEval, self).__init__(*args)

    def forword(self, funs):

        metric_iou1, metric_iou2, metric_iou3 = 0,  0, 0
        metric_oa1, metric_oa2, metric_oa3 = 0, 0, 0
        metric_f1, metric_f2, metric_f3 = 0, 0, 0

        for data_list in self.test_loader:

            data_list = to_device(data_list, self.device)

            y_pred = self.luncher.model(*data_list)

            y_true = data_list[1]

            # y_true_one_hot = ffunc.one_hot(y_true[:, 0].long(), 6)

            for fun in funs:

                metric_i1 = fun(y_pred[0], y_true[:, 0])
                metric_i2 = fun(y_pred[1], y_true[:, 0])
                metric_i3 = fun(y_pred[2], y_true[:, 0])

                if fun.__name__ == 'iou_score':
                    metric_iou1 = metric_iou1 + metric_i1
                    metric_iou2 = metric_iou2 + metric_i2
                    metric_iou3 = metric_iou3 + metric_i3
                elif fun.__name__ == 'oa_score':
                    metric_oa1 = metric_oa1 + metric_i1
                    metric_oa2 = metric_oa2 + metric_i2
                    metric_oa3 = metric_oa3 + metric_i3
                else:
                    metric_f1 = metric_f1 + metric_i1
                    metric_f2 = metric_f2 + metric_i2
                    metric_f3 = metric_f3 + metric_i3

        size = len(self.test_loader)

        per_class_iou1 = self.per_class_iu(metric_iou1)
        per_class_iou2 = self.per_class_iu(metric_iou2)
        per_class_iou3 = self.per_class_iu(metric_iou3)

        iou1 = np.mean(per_class_iou1)
        iou2 = np.mean(per_class_iou2)
        iou3 = np.mean(per_class_iou3)

        oa1 = metric_oa1 / size
        oa2 = metric_oa2 / size
        oa3 = metric_oa3 / size

        # f1 = metric_f1 / size
        print(per_class_iou3)
        metric_dict = {
            "IOU": [iou1,iou2,iou3],
            "OA": [oa1,oa2,oa3],
            "F1": [oa1,oa2,oa3]
        }

        return metric_dict

    @property
    def metric_scores(self):
        return self.forword(
            [
                IoU(),
                Accuracy()
            ]
        )

