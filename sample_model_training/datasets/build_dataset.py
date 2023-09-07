from torch.utils.data import DataLoader
import datasets
import time

from .augmentation import Flip, Rotation, Crop


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self


def build_dataset(args):
    aug_methods = {'Flip': Flip(), 'Rotation': Rotation(**args)}
    pipeline=[aug_methods[i] for i in args.pop('aug_pipeline')] if 'aug_pipeline' in args and not args['test_mode'] else None
    dataset = datasets.__dict__[args.pop('dataset_type')](**args, pipeline=pipeline)
    if args['test_mode']:
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        return IterLoader(DataLoader(dataset=dataset, num_workers=16, batch_size=args.pop('batch_size'), shuffle=True, drop_last=True, pin_memory=True))
