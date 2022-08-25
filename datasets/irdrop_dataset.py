# Copyright 2022 CircuitNet. All rights reserved.

import os.path as osp
import copy
import numpy as np


class IRDropDataset(object):
    def __init__(self, ann_file, dataroot, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

        self.temporal_key = 'Power_t'

    def load_annotations(self):  
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                infos = line.strip().split(',')
                label = infos[-1]
                features = infos[:-1]
                info_dict = dict()
                if self.dataroot is not None:
                    for feature in features:
                        info_dict[feature.split('/')[0]] = osp.join(self.dataroot, feature)
                    feature_path = info_dict
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])

        spatial_feature = []
        for k, v in results['feature_path'].items():
            if k != self.temporal_key:
                spatial_feature.append(np.load(v))
            else:
                temporal_feature = np.load(v)
        feature = [np.array(spatial_feature), temporal_feature]
        feature = np.ascontiguousarray(np.concatenate(feature, axis=0).astype(np.float32))
        feature = np.expand_dims(feature, axis=0)
        label = np.load(results['label_path']).astype(np.float32)
        return feature, label, results['label_path']


    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        return self.prepare_data(idx)