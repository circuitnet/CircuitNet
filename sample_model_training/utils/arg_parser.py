import argparse
import os
import sys
sys.path.append(os.getcwd())

class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--save_path', default='work_dir/sample/')
        self.parser.add_argument('--pretrained', default=None)

        self.parser.add_argument('--max_iters', default=200)
        self.parser.add_argument('--args', default=None)
        self.parser.add_argument('--cpu', action='store_true')
        self.parser.add_argument('--gpu', default = None)

        self.parser.add_argument('--dataroot', default='../feature_extraction')
        self.parser.add_argument('--ann_file', default='../feature_extraction/train.csv')
        self.parser.add_argument('--dataset_type', default='TrainDataset')
        self.parser.add_argument('--batch_size', default=1)
        self.parser.add_argument('--aug_pipeline', default=['Flip'])
        
        self.parser.add_argument('--model_type', default='FCN')
        self.parser.add_argument('--in_channels', default=2)
        self.parser.add_argument('--out_channels', default=1)
        self.parser.add_argument('--lr', default=2e-4)
        self.parser.add_argument('--weight_decay', default=0)
        self.parser.add_argument('--loss_type', default='MSELoss')
        self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM', 'EMD'])
