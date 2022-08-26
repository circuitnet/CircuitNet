# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())


class Paraser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='congestion_gpdl')

        self.parser.add_argument('--save-path', default='work_dir/congestion_gpdl/')
    
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
            self.parser.add_argument('--eval-metric', default=['PSNR', 'SSIM', 'EMD', 'NRMS'])

        elif self.parser.parse_args().task == 'drc_routenet':
            self.parser.add_argument('--dataroot', default='./datasets/')
            self.parser.add_argument('--ann_file_train', default='./files/drc_data/drc_train.csv')
            self.parser.add_argument('--ann_file_test', default='./files/drc_data/drc_val.csv')
            self.parser.add_argument('--dataset_type', default='DRCDataset')
            self.parser.add_argument('--batch_size', default=8)
            self.parser.add_argument('--aug_pipeline', default=['Flip', 'Rotation'])

            self.parser.add_argument('--model_type', default='RouteNet')
            self.parser.add_argument('--in_channels', default=9)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-4)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['PSNR', 'SSIM'])
            self.parser.add_argument('--threashold', default=0.1)


        elif self.parser.parse_args().task == 'irdrop_mavi':
            self.parser.add_argument('--dataroot', default='./datasets/')
            self.parser.add_argument('--ann_file_train', default='/files/irdrop/irdrop_train.csv')
            self.parser.add_argument('--ann_file_test', default='./files/irdrop/irdrop_val.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=2)

            self.parser.add_argument('--model_type', default='MAVI')
            self.parser.add_argument('--in_channels', default=1)
            self.parser.add_argument('--out_channels', default=4)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['PSNR', 'SSIM'])
            self.parser.add_argument('--threashold', default=0.05)

        else:
            raise ValueError
