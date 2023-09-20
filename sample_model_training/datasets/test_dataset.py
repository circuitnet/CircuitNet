import os
import copy
import numpy as np


class TestDataset(object):
    def __init__(self, ann_file, dataroot, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if not line.strip():
                    continue
                else:
                    feature, label, instance_count, instance_IR_drop, instance_name = line.strip().split(',')
                    if self.dataroot is not None:
                        feature_path = os.path.join(self.dataroot, feature)
                        label_path = os.path.join(self.dataroot, label)
                        instance_count_path = os.path.join(self.dataroot, instance_count)
                        instance_IR_drop_path = os.path.join(self.dataroot, instance_IR_drop)
                        instance_name_path = os.path.join(self.dataroot, instance_name)
                    data_infos.append(dict(feature_path=feature_path, label_path=label_path, instance_count_path=instance_count_path, instance_IR_drop_path=instance_IR_drop_path, instance_name_path=instance_name_path))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        
        feature = np.load(results['feature_path']).transpose(2, 0, 1).astype(np.float32)
        label = np.load(results['label_path']).transpose(2, 0, 1).astype(np.float32)
        return feature, label,  results['instance_count_path'], results['instance_IR_drop_path'], results['instance_name_path']


    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        return self.prepare_data(idx)