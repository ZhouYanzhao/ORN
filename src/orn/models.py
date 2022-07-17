import re

import torch.nn as nn
import torchvision.models as M

from orn.oriented_response_convolution import ORConv2d
from orn.rotation_invariant_encoding import ORAlign1d, ORPool1d


def upgrade_to_orn(model, num_orientation=8, scale_factor=2, classifier=None, features=None, invariant_encoding=None, encode_after_features=True):
    '''Recursively replace Conv layers to ORConv layers.
    '''
    state = {'counter': 0, 'num_feature': 0}

    def _replace_handler(module, state):
        for attr, target in module.named_children():
            if type(target) == nn.Conv2d:
                layer = ORConv2d(
                    (target.in_channels if state['counter'] == 0 else target.in_channels // scale_factor),
                    target.out_channels // scale_factor,
                    arf_config=(1 if state['counter'] == 0 else num_orientation, num_orientation),
                    kernel_size=target.kernel_size,
                    stride=target.stride,
                    padding=target.padding,
                    dilation=target.dilation,
                    groups=target.groups,
                    bias=target.bias is not None)
                setattr(module, attr, layer)
                state['counter'] += 1
                state['num_feature'] = layer.out_channels * num_orientation
            elif type(target) == nn.BatchNorm2d:
                layer = nn.BatchNorm2d(
                    state['num_feature'], target.eps, target.momentum, 
                    target.affine, target.track_running_stats)
                setattr(module, attr, layer)
            elif classifier is not None and target == classifier:
                layer = nn.Linear(
                    target.in_features // scale_factor * (1 if invariant_encoding == 'pool' else num_orientation),
                    target.out_features,
                    bias=target.bias is not None)
                setattr(module, attr, layer)
            
            if features is not None and target == features:
                layer = getattr(module, attr)
                if invariant_encoding == 'align':
                    if encode_after_features:
                        layer = nn.Sequential(layer, ORAlign1d(num_orientation))
                    else:
                        layer = nn.Sequential(ORAlign1d(num_orientation), layer)
                elif invariant_encoding == 'pool':
                    if encode_after_features:
                        layer = nn.Sequential(layer, ORPool1d(num_orientation))
                    else:
                        layer = nn.Sequential(ORPool1d(num_orientation), layer)
                setattr(module, attr, layer)
            else:
                _replace_handler(target, state)

    _replace_handler(model, state)
    
class ModelFactor(object):

    def __init__(self):
        self.regex = {
            'alexnet': r'alexnet',
            'vgg': r'vgg\d+[_bn]*',
            'inception': r'inception_v\d',
            'resnet': r'resnet\d+',
            'resnext': r'resnext\d+_\d+x\d+d',
            'wrn': r'wide_resnet\d+_\d+',
        }
        self.valid_models = {}
        for v in dir(M):
            for k, u in self.regex.items():
                if re.match(u, v):
                    for num_orientation in [4, 8]:
                        self.valid_models[f'or_{v}_{num_orientation}'] = {'base': v, 'num_orientation': num_orientation, 'invairant_encoding': None}
                        self.valid_models[f'or_{v}_pool_{num_orientation}'] = {'base': v, 'num_orientation': num_orientation, 'invairant_encoding': 'pool'}
                        if k not in ['alexnet', 'vgg']:
                            self.valid_models[f'or_{v}_align_{num_orientation}'] = {'base': v, 'num_orientation': num_orientation, 'invairant_encoding': 'align'}
                    break

    def __getattr__(self, name):
        if name in self.valid_models.keys():
            setting = self.valid_models[name]
            base = getattr(M, setting['base'])
            def _handler():
                model = base()
                if re.match(self.regex['alexnet'], setting['base']):
                    upgrade_to_orn(model, num_orientation=setting['num_orientation'], classifier=model.classifier[0], features=model.avgpool, invariant_encoding=setting['invairant_encoding'])
                elif re.match(self.regex['vgg'], setting['base']):
                    upgrade_to_orn(model, num_orientation=setting['num_orientation'], classifier=model.classifier[0], features=model.avgpool, invariant_encoding=setting['invairant_encoding'])
                elif re.match(self.regex['inception'], setting['base']):
                    upgrade_to_orn(model, num_orientation=setting['num_orientation'], classifier=model.fc, features=model.avgpool, invariant_encoding=setting['invairant_encoding'])
                elif re.match(self.regex['resnet'], setting['base']):
                    upgrade_to_orn(model, num_orientation=setting['num_orientation'], classifier=model.fc, features=model.avgpool, invariant_encoding=setting['invairant_encoding'])
                elif re.match(self.regex['resnext'], setting['base']):
                    upgrade_to_orn(model, num_orientation=setting['num_orientation'], classifier=model.fc, features=model.avgpool, invariant_encoding=setting['invairant_encoding'])
                elif re.match(self.regex['wrn'], setting['base']):
                    upgrade_to_orn(model, num_orientation=setting['num_orientation'], classifier=model.fc, features=model.avgpool, invariant_encoding=setting['invairant_encoding'])
                return model
            return _handler

    def __dir__(self):
        return self.valid_models

models = ModelFactor()
