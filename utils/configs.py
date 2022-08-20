# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())


class Paraser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='congestion_gpdl')
        self.parser.add_argument('--save-path', default='work_dir/congestion/')
    
        self.parser.add_argument('--pretrained', default=None)
        
        self.parser.add_argument('--max_iters', default=200000)
        self.parser.add_argument('--save_as_npy', default=False, type=bool)
        self.get_remainder()
        
    def get_remainder(self):
        if self.parser.parse_args().task == 'congestion_gpdl':
            self.parser.add_argument('--dataroot', default='./datasets/')
            self.parser.add_argument('--ann_file_train', default='./files/congestion/congestion_train.csv')
            self.parser.add_argument('--ann_file_test', default='./files/congestion/congestion_val.csv')
            self.parser.add_argument('--dataset_type', default='CongestionDataset')
            self.parser.add_argument('--batch_size', default=16)
            self.parser.add_argument('--aug_pipeline', default=['Flip', 'Rotation'])
            
            self.parser.add_argument('--model_type', default='GPDL')
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=0)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval-matric', default=['PSNR', 'SSIM', 'EMD', 'NRMS'])
        else:
            raise ValueError
