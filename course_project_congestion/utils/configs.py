# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())


class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='congestion_gpdl')

        self.parser.add_argument('--save_path', default='./congestion_gpdl')
    
        self.parser.add_argument('--pretrained', default=None)

        self.parser.add_argument('--max_iters', type=int, default=200000)
        self.parser.add_argument('--plot_roc', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--cpu', action='store_true')

        self.parser.add_argument('--data_ratio', type=float, default=1.0)
        self.parser.add_argument('--log_dir', type=str, default='./logs')
        self.parser.add_argument('--quant_bits', type=int, default=8)
        self.get_remainder()

    def get_remainder(self):
        args, unknown = self.parser.parse_known_args()
        
        if args.task == 'congestion_gpdl':
            self.parser.add_argument('--dataroot', default='../../training_set/congestion')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='CongestionDataset')
            self.parser.add_argument('--batch_size', default=16)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])
            
            self.parser.add_argument('--model_type', default='GPDL')
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=0)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM', 'EMD'])

            self.parser.add_argument('--distill', action='store_true', help='Enable distillation')
            self.parser.add_argument('--teacher_model_type', default=None, help='Model type of pretrained teacher model')
            self.parser.add_argument('--teacher_pretrained', default=None, help='Path to pretrained teacher model')
            self.parser.add_argument('--distill_alpha', default=0.8, type=float, help='Weight for distillation loss')

        elif args.task == 'drc_routenet':
            self.parser.add_argument('--dataroot', default='../../training_set/DRC')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='DRCDataset')
            self.parser.add_argument('--batch_size', default=8)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])

            self.parser.add_argument('--model_type', default='RouteNet')
            self.parser.add_argument('--in_channels', default=9)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-4)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.1)

        elif args.task == 'irdrop_mavi':
            self.parser.add_argument('--dataroot', default='../../training_set/IR_drop')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=2)

            self.parser.add_argument('--model_type', default='MAVI')
            self.parser.add_argument('--in_channels', default=1)
            self.parser.add_argument('--out_channels', default=4)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.9885) # 5% after log

        else:
            raise ValueError
