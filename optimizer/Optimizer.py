import torch.optim

from utils.UtilRegister import Optimizer

Optimizer.register('sgd', torch.optim.SGD)
Optimizer.register('adam', torch.optim.Adam)


@Optimizer.register('Optimizer')
def make_optimizer(net_params, parameters):

    opt_method = parameters['method']
    opt_type = opt_method['type']

    if opt_type in Optimizer:
        opt = Optimizer[opt_type](params=net_params, **opt_method['params'])
        if 'grad_clip' in parameters:
            opt.grad_config = parameters['grad_clip']
        return opt
    else:
        raise ValueError('{} is not support now.'.format(opt_type))
