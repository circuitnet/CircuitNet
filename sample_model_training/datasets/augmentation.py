import mmcv
import numpy as np
import cv2

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

class Crop:
    def __init__(self, keys=['feature', 'label'], crop_ratio=0.5, resize_size = 384, crop_szie = 256, **kwargs):
        self.keys = keys
        self.crop_ratio = crop_ratio
        self.resize_size = resize_size
        self.crop_szie = crop_szie

    def __call__(self, results):
        crop = np.random.random() < self.crop_ratio

        if crop:
            for key in self.keys:
                    results[key] = self.random_crop(results[key], self.resize_size, self.crop_szie)

        return results

    def random_crop(self, image,resize_size, crop_size):
        # print(image.shape)
        image = cv2.resize(image, (resize_size, resize_size), interpolation = cv2.INTER_CUBIC)
        height, width = image.shape[:2]
        crop_height = crop_size
        crop_width = crop_size

        # 确定裁剪的起始位置
        y = np.random.randint(0, height - crop_height + 1)
        x = np.random.randint(0, width - crop_width + 1)

        # 进行裁剪
        cropped_image = image[y:y+crop_height, x:x+crop_width]
        if len(cropped_image.shape) == 2:
            cropped_image = np.expand_dims(cropped_image, axis=2)
        return cropped_image
