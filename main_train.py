import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random

import numpy as np
import torch

from metrics.MetricHelper import register_evaluate_fn
from utils.UtilRegister import Parameter, SampleLoader, Loss, Model, Optimizer, LearningRate, Training

from license import Register

def run(config_path, gpu_mode=True):

    # 1. get parametershaoduole
    parameters = Parameter['Setting'](config_path)
    # 2. setting dataset
    train_loader = SampleLoader['Dataset'](training=True, **parameters['dataset'])()
    val_loader = SampleLoader['Dataset'](training=False, **parameters['dataset'])()
    # 3. setting device
    if gpu_mode:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    # 4. setting loss
    loss = Loss['MultiLoss'](**parameters['loss'])
    # 5. build model
    model = Model['Model'](callback=loss, **parameters['model']).to(device)
    # 6. build optimizer
    lr_schedule = LearningRate['LR'](parameters['optimizer'])
    optimizer = Optimizer['Optimizer'](model.parameters(), parameters['optimizer'])
    # 7. build trainer
    trainer = Training['Launcher'](model, optimizer, lr_schedule, device, parameters['model'])
    # 8. register function
    register_evaluate_fn(trainer)
    # 9. running
    trainer.train_epochs(train_loader, val_loader)


def set_seed(seed=666666):
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':

    set_seed()
    run('configs/config.json')




