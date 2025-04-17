import math
from utils.UtilRegister import LearningRate


class LearningRateBase(object):
    def __init__(self, base_lr):
        self._base_lr = base_lr

    @property
    def base_lr(self):
        return self._base_lr

    def step(self, global_step, optimizer):
        raise NotImplementedError


@LearningRate.register('LR')
def make_learningrate(parameters):
    lr = parameters['learning_rate']
    lr_type = lr['type']
    parameters['method']['params']['lr'] = lr['params']['base_lr']
    if lr_type in LearningRate:
        lr_module = LearningRate[lr_type]
        return lr_module(**lr['params'])
    else:
        raise ValueError('{} is not support now.'.format(lr_type))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@LearningRate.register('poly')
class PolyLearningRate(LearningRateBase):
    def __init__(self,
                 base_lr,
                 power,
                 max_iters,
                 ):
        super(PolyLearningRate, self).__init__(base_lr)
        self.power = power
        self.max_iters = max_iters

    def step(self, global_step, optimizer):

        factor = (1 - global_step / self.max_iters) ** self.power
        cur_lr = self.base_lr * factor
        set_lr(optimizer, cur_lr)


@LearningRate.register('cosine')
class CosineAnnealingLearningRate(LearningRateBase):
    def __init__(self, base_lr, max_iters, eta_min):
        super(CosineAnnealingLearningRate, self).__init__(base_lr)
        self.eta_min = eta_min
        self.max_iters = max_iters

    def step(self, global_step, optimizer):

        cur_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * global_step / self.max_iters))

        set_lr(optimizer, cur_lr)
