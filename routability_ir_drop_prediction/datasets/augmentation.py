# Copyright 2022 CircuitNet. All rights reserved.

import mmcv
import numpy as np

class Flip:
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys=['feature', 'label'], flip_ratio=0.5, direction='horizontal', **kwargs):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        mmcv.imflip_(v, self.direction)
                else:
                    mmcv.imflip_(results[key], self.direction)

        return results



class Rotation:
    def __init__(self, keys=['feature', 'label'], axis=(0,1), rotate_ratio=0.5, **kwargs):
        self.keys = keys
        self.axis = {k:axis for k in keys} if isinstance(axis, tuple) else axis
        self.rotate_ratio = rotate_ratio
        self.direction = [0, -1, -2, -3]

    def __call__(self, results):
        rotate = np.random.random() < self.rotate_ratio

        if rotate:
            rotate_angle = self.direction[int(np.random.random()/(10.0/3.0))+1]
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        results[key] = np.ascontiguousarray(np.rot90(v, rotate_angle, axes=self.axis[key]))
                else:
                    results[key] = np.ascontiguousarray(np.rot90(results[key], rotate_angle, axes=self.axis[key]))

        return results