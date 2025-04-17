from utils.UtilBase import to_device
from utils.UtilRegister import Metric


def fcn_evaluate_fn(self, test_loader, device='cuda'):

    # metrics = Metric['Metric'](pred_true)

    metrics = Metric['Metric'](self,test_loader,device)

    metric_dict = metrics.metric_scores

    self._ckpt.is_global_acc(metric_dict['IOU'][0])

    self._ckpt.log_info(
        **{
            "step":self._ckpt._global_step,
            "epoch": self.epoch,
            "train_flag":'Test',
            "metric_dict":metric_dict
        }
    )


def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)
