import re
import functools
import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation is None or activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softmax':
            self.activation = functools.partial(torch.softmax, dim=1)
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError

    def forward(self, x):
        return self.activation(x)


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class MetricBase(BaseObject):
    pass
