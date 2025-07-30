# Copyright 2022 CircuitNet. All rights reserved.

from torch.utils.data import DataLoader, Subset
import datasets
import time
import numpy as np

from .augmentation import Flip, Rotation


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self


def build_dataset(opt):
    aug_methods = {'Flip': Flip(), 'Rotation': Rotation(**opt)}
    pipeline=[aug_methods[i] for i in opt.pop('aug_pipeline')] if 'aug_pipeline' in opt and not opt['test_mode'] else None
    dataset = datasets.__dict__[opt.pop('dataset_type')](**opt, pipeline=pipeline)

    # print(f'original dataset size: {len(dataset)}')

    data_ratio_value = opt.get('data_ratio', 1.0) # Get data_ratio, default to 1.0 if not present

    if not opt['test_mode'] and 0.0 <= data_ratio_value < 1.0:
        num_original_samples = len(dataset)
        if num_original_samples > 0:
            num_subset_samples = int(num_original_samples * data_ratio_value)
            if num_subset_samples < num_original_samples and num_subset_samples > 0 : # Only create a true subset
                indices = np.random.choice(num_original_samples, num_subset_samples, replace=False)
                dataset = Subset(dataset, indices)
            elif num_subset_samples == 0 : # If ratio results in zero samples
                dataset = Subset(dataset, []) # Use an empty subset

    # print(f'subset dataset size: {len(dataset)}')

    if opt['test_mode']:
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        return IterLoader(DataLoader(dataset=dataset, num_workers=16, batch_size=opt.pop('batch_size'), shuffle=True, drop_last=True, pin_memory=True))
