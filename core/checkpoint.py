import os
import time
import json
from collections import OrderedDict

import torch
from utils.UtilLogger import Logger
from utils.UtilAttribute import AttrDict

class CheckPoint(object):
    MODEL = 'model'
    OPTIMIZER = 'opt'
    GLOBALSTEP = 'global_step'
    LASTCHECKPOINT='last_check_name'
    CHECKPOINT_NAME = 'checkpoint_info.json'

    def __init__(self, **kwargs):
        self.config = AttrDict(**kwargs)
        self._launcher = self.config.get('launcher',None)
        self.model = self.config.get('model',None)
        self._global_step = 0
        self._global_acc = 0
        if self._launcher is not None:
            self._logger = Logger('LuoWR', log_file=self._launcher.log_dir)
            self._json_log = dict(
                last_check_name='model.pth',
                step=0,
                total_acc=0
            )

    def is_global_acc(self, value):
        if self._global_acc < value:
            self._global_acc = value
            self.save_model()

    @staticmethod
    def load_checkpoint_info(model_dir):
        json_path = os.path.join(model_dir, CheckPoint.CHECKPOINT_NAME)
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        return json_file

    @staticmethod
    def load(filepath):
        ckpt = torch.load(filepath, map_location=torch.device("cpu"))
        return ckpt

    @property
    def global_step(self):
        return self._global_step

    def step(self):
        self._global_step += 1

    def set_launcher(self, launcher):
        self._launcher = launcher

    def save_model(self):
        filepath = os.path.join(self._launcher.model_dir, str(time.time()) + 'model.pth')
        ckpt = OrderedDict({
            CheckPoint.MODEL: self._launcher.model.state_dict(),
            CheckPoint.OPTIMIZER: self._launcher.optimizer.state_dict(),
            CheckPoint.GLOBALSTEP: self.global_step
        })
        torch.save(ckpt, filepath)
        self._json_log["last_check_name"] = filepath
        self._json_log["step"] = self._global_step
        self._json_log["total_acc"] = self._global_acc
        self.save_checkpoint_info(self._launcher.model_dir)

    def save_checkpoint_info(self, model_dir):
        with open(os.path.join(model_dir, CheckPoint.CHECKPOINT_NAME), 'w') as f:
            json.dump(self._json_log, f)

    def try_resume(self):
        """ json -> ckpt_path -> ckpt -> launcher

        Returns:

        """
        if self._launcher is not None:
            model_dir = self._launcher.model_dir
        else:
            model_dir = self.config.get('model_dir')
        # 1. json
        json_log = self.load_checkpoint_info(model_dir)
        if json_log is None:
            return
        # 2. ckpt path
        last_path = os.path.join(model_dir, json_log[CheckPoint.LASTCHECKPOINT])
        # 3. ckpt
        ckpt = self.load(last_path)
        # 4. resume
        if self._launcher is not None:
            self._launcher.model.load_state_dict(ckpt[CheckPoint.MODEL])
            if self._launcher.optimizer is not None:
                self._launcher.optimizer.load_state_dict(ckpt[CheckPoint.OPTIMIZER])
            if self._launcher.checkpoint is not None:
                self._launcher.checkpoint.set_global_step(ckpt[CheckPoint.GLOBALSTEP])
        else:
            self.model.load_state_dict(ckpt[CheckPoint.MODEL])
            self.model.eval()

    def log_info(self, **kwargs):
        self._logger.log_info(**kwargs)