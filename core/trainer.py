import types
import functools
import torch
from torch.nn.utils import clip_grad

from utils.UtilBase import to_device
from utils.UtilRegister import Training
from core.checkpoint import CheckPoint


class Iterator(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._iterator = iter(self._data_loader)
        self._current_epoch = 0

    @torch.no_grad()
    def next(self, launcher):
        launcher._current_epoch = self._current_epoch
        try:
            data = next(self._iterator)
            self._current_epoch += 1
        except StopIteration:
            self.reset()
            self._current_epoch = 0
            if self._current_epoch % launcher.eval_interval_epoch == 0:
                for f in launcher.call_backs:
                    f()
            data = next(self._iterator)
        launcher.data = data

    def reset(self):
        self._iterator = iter(self._data_loader)


@Training.register('Launcher')
class Launcher(object):
    def __init__(self,
                 model,
                 optimizer,
                 lr_schedule,
                 device,
                 parameters):

        self._model = model
        self._optimizer = optimizer
        self._parameters = parameters
        self._lr_schedule = lr_schedule
        self._device = device
        self._ckpt = CheckPoint(launcher=self)
        self._training = False
        self.total_epoch = 0
        self.data = None

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model_dir(self):
        return self._parameters['model_dir']

    @property
    def log_dir(self):
        return self._parameters['log_dir']

    @property
    def map_dir(self):
        return self._parameters['map_dir']

    @property
    def epoch(self):
        return self._parameters['epoch']

    @property
    def eval_interval_epoch(self):
        return self._parameters['eval_interval_epoch']

    @property
    def checkpoint(self):
        return self._ckpt

    def _update_lr(self):
        self._lr_schedule.step(self._ckpt.global_step, self._optimizer)

    def backward_apply_gradient(self, total_loss):
        total_loss.backward()
        clip_grad.clip_grad_norm_(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            ),
            **self._optimizer.grad_config
        )
        # self._ckpt.step()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._update_lr()

    def override_evaluate(self, fn):
        self._evaluate_fn = types.MethodType(fn, self)

    def override_backward(self, fn):
        self.backward = types.MethodType(fn, self)

    def _evaluate_fn(self, data_loader, **kwargs):
        raise NotImplementedError

    def evaluate(self, data_loader, **kwargs):
        self._evaluate_fn(data_loader, **kwargs)

    def set_iterator_call_backs(self,train_data_loader,test_data_loader,**kwargs):
        self.iterator = Iterator(train_data_loader)
        self.call_backs = [functools.partial(self.evaluate, test_data_loader, **kwargs)]

    @torch.no_grad()
    def eval_model(self):
        self.model.train(False)
        self.iterator.next(self)

    @torch.enable_grad()
    def train_model(self):
        self.model.train(True)
        self.data = to_device(self.data, self._device)
        total_loss, loss_teams = self.model(*self.data,epoch=self._ckpt.global_step)
        self.backward_apply_gradient(total_loss)
        if self._parameters.get('is_output_train_log'):
            self.checkpoint.log_info(
                **{
                    "step": self._current_epoch,
                    "epoch": self.total_epoch,
                    "loss_dict": loss_teams,
                    "train_flag": 'Train',
                }
        )

    def train_epochs(self, train_data_loader, test_data_loader):
        self.total_epoch = train_data_loader.total_epoch
        self.set_iterator_call_backs(train_data_loader, test_data_loader)
        for _ in range(self.epoch):
            for _ in range(self.total_epoch):
                self.eval_model()
                self.train_model()
            self._ckpt.step()









