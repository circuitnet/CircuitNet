# Copyright 2022 CircuitNet. All rights reserved.

import models
import torch

def build_model(opt):
    model = models.__dict__[opt.pop('model_type')](**opt)
    model.init_weights(**opt)
    if opt['test_mode']:
        model.eval()
    return model
