import models
import torch

def build_model(args):
    model = models.__dict__[args.pop('model_type')](**args)
    model.init_weights(**args)
    if args['test_mode']:
        model.eval()
    return model
