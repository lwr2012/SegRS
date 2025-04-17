import os
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    return logger


def get_console_file_logger(name,level=logging.INFO, log_dir=None):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(log_dir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger


class Logger(object):
    def __init__(self,
                 name,
                 level=logging.INFO,
                 log_file=None):
        # self._logger = logging.getLogger(name)
        self._level = level
        # self._logger.setLevel(level)
        self._logger = get_console_file_logger(name, level, log_file)

    def info(self, value):
        self._logger.info(value)

    def on(self):
        self._logger.setLevel(self._level)

    def off(self):
        self._logger.setLevel(100)

    def log_info(self,
                  step=1,
                  epoch=1,
                  loss_dict=None,
                  metric_dict=None,
                  train_flag='train'
                  ):
        if loss_dict:
            loss_info = ''.join(
                ['loss_{name} = {value}, '.format(name=i+1, value=np.round(value.item(), 6)) for i, value in enumerate(loss_dict)])
        else:
            loss_info = ''

        if metric_dict:
            metric_info = ''.join(
                ['{name} = {value}, '.format(
                    name=name,
                    value=[np.round(val.item(), 6) for val in value]
                ) for name, value in metric_dict.items()])
        else:
            metric_info = ''

        train_flag = '[{}===>{}/{}], '.format(train_flag, int(step+1), int(epoch))

        msg = '{train_flag}{loss}{metric}'.format(
            train_flag=train_flag,
            loss=loss_info,
            metric=metric_info,
        )

        self._logger.info(msg)


