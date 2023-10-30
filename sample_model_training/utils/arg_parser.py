import argparse
import os
import sys
sys.path.append(os.getcwd())

class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--save_path', default='work_dir/sample/', help='Save path.')
        self.parser.add_argument('--args', default=None, help='Path to json arguments file.')
        self.parser.add_argument('--cpu', action='store_true', help='Use CPU for training or testing.')
        self.parser.add_argument('--gpu', default=None, help='Specify GPU ID to use.')

        self.parser.add_argument('--dataroot', default='../feature_extraction', help='Path to data')
        self.parser.add_argument('--ann_file', default='../feature_extraction/train.csv', help='The path to the csv that controls data allocation')
        self.parser.add_argument('--dataset_type', default='TrainDataset', help='Specify the dataloader.')
        self.parser.add_argument('--batch_size', default=1, help='Batch size.')

        self.parser.add_argument('--model_type', default='FCN', help='Specify the model.')
        self.parser.add_argument('--pretrained', default=None, help='Load model checkpoint, can be used in both testing and training.')
        self.parser.add_argument('--in_channels', default=2, help='Model parameters, input channels.')
        self.parser.add_argument('--out_channels', default=1, help='Model parameters, output channels.')

        # Training

        self.parser.add_argument('--max_iters', default=200, help='Max training iterations.')
        self.parser.add_argument('--aug_pipeline', default=['Flip'], help='Data augmentation, only random flipping in this sample.')
        self.parser.add_argument('--lr', default=2e-4, help='Learning rate.')
        self.parser.add_argument('--weight_decay', default=0, help='Weight decay.')
        self.parser.add_argument('--loss_type', default='MSELoss', help='Training loss, MSELoss or L1Loss')

        # Testing

        self.parser.add_argument('--plot', default=False, help='Plot the prediction and label in testing.')
        self.parser.add_argument('--eval_metric', default=["MAE", "corrcoef"], help='Metrics.')
