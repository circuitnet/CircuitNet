# Copyright 2022 CircuitNet. All rights reserved.

import os
import os.path as osp
import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from math import cos, pi
from datetime import datetime # Added for logging timestamps


def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
        


class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim_item in optimizer.items(): # renamed optim to avoid conflict
                for param_group, lr in zip(optim_item.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


def train():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    # --- START LOGGING SETUP ---
    log_dir = arg_dict.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"train_log_{arg_dict.get('task', 'unknown_task')}_{current_time_str}.txt"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    _log_file_handle = open(log_file_path, 'a')

    def log_message(message, console_print=True):
        timestamped_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        if console_print:
            print(message)
        _log_file_handle.write(timestamped_message + "\n")
        _log_file_handle.flush()
    # --- END LOGGING SETUP ---

    log_message(f"Training session started. Logging to: {log_file_path}", console_print=True)
    log_message("Configuration parameters:", console_print=False)
    for key, value in sorted(arg_dict.items()):
        log_message(f"  {key}: {value}", console_print=False)
    log_message("----------------------------------------------------", console_print=False)

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)
    log_message(f"Arguments saved to {os.path.join(arg_dict['save_path'], 'arg.json')}", console_print=False)


    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False 
    log_message(f"Using data_ratio: {arg_dict.get('data_ratio', 1.0)} (passed to build_dataset)", console_print=True)


    log_message('===> Loading datasets')
    dataset = build_dataset(arg_dict) # data_ratio is used within build_dataset

    log_message('===> Building model')
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()
    
    loss = build_loss(arg_dict)
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000

    log_message(f"Starting training for {arg_dict['max_iters']} iterations.", console_print=True)
    log_message(f"Logging training progress every {print_freq} iterations.", console_print=False)
    log_message(f"Saving checkpoints every {save_freq} iterations to {arg_dict['save_path']}", console_print=False)

    try:
        while iter_num < arg_dict['max_iters']:
            # tqdm description will show overall iteration progress
            with tqdm(total=print_freq, desc=f"Iter {iter_num}/{arg_dict['max_iters']}") as bar:
                for feature, label, _ in dataset:        
                    if arg_dict['cpu']:
                        input_data, target_data = feature, label # Original 'input', 'target'
                    else:
                        input_data, target_data = feature.cuda(), label.cuda()

                    regular_lr_list = cosine_lr.get_regular_lr(iter_num)
                    cosine_lr._set_lr(optimizer, regular_lr_list)
                    current_lr_for_log = optimizer.param_groups[0]['lr'] # For logging

                    prediction = model(input_data)

                    optimizer.zero_grad()
                    pixel_loss = loss(prediction, target_data)

                    epoch_loss += pixel_loss.item()
                    pixel_loss.backward()
                    optimizer.step()

                    iter_num += 1
                    
                    bar.update(1)
                    bar.set_postfix_str(f"Loss: {pixel_loss.item():.4f}, LR: {current_lr_for_log:.2e}")


                    if iter_num % print_freq == 0:
                        break
            
            avg_loss_for_period = epoch_loss / print_freq
            # The original print statement for loss:
            # print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
            # Replaced with log_message:
            log_message(f"===> Iters[{iter_num}]({iter_num}/{arg_dict['max_iters']}): Avg Loss: {avg_loss_for_period:.4f}, Current LR: {current_lr_for_log:.6e}", console_print=True)
            
            if iter_num % save_freq == 0:
                checkpoint(model, iter_num, arg_dict['save_path']) # checkpoint() already prints
                log_message(f"Checkpoint saved for iter {iter_num}.", console_print=False) # Additional log to file
            epoch_loss = 0

        log_message("Training finished successfully.", console_print=True)

    except Exception as e:
        log_message(f"AN ERROR OCCURRED DURING TRAINING: {str(e)}", console_print=True)
        import traceback
        log_message(traceback.format_exc(), console_print=False) # Log full traceback to file
        # raise # Optionally re-raise the exception if you want the script to terminate with an error code
    finally:
        if _log_file_handle:
            log_message("Closing log file.", console_print=False)
            _log_file_handle.close()


if __name__ == "__main__":
    train()