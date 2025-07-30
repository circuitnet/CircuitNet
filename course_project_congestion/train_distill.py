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

class ThreeStageDistillLr(object):
    """三阶段蒸馏学习率调度器
    Stage 0 (Warmup): 保持base_lr
    Stage 1 (Pure Distillation): 保持base_lr或轻微衰减
    Stage 2 (Mixed Training): 从reduced_lr开始余弦衰减
    """
    def __init__(self,
                 base_lr,
                 max_iters,
                 warmup_ratio=0.05,
                 stage1_ratio=0.05,
                 stage1_end_ratio=0.8,      # Stage1结束时LR相对base_lr的比例
                 stage2_start_ratio=0.5,    # Stage2开始时LR相对base_lr的比例
                 min_lr_ratio=0.01):        # 最小LR相对base_lr的比例
        
        # 🔥 保存原始参数
        self.original_base_lr = base_lr  # 用于所有计算的基准
        self.base_lr = base_lr           # 兼容接口，可能会被set_init_lr修改
        self.max_iters = max_iters
        self.warmup_ratio = warmup_ratio
        self.stage1_ratio = stage1_ratio
        self.stage1_end_ratio = stage1_end_ratio
        self.stage2_start_ratio = stage2_start_ratio
        self.min_lr_ratio = min_lr_ratio
        
        # 计算各阶段的迭代数
        self.warmup_iters = int(max_iters * warmup_ratio)
        self.stage1_iters = int(max_iters * stage1_ratio)
        self.stage2_iters = max_iters - self.warmup_iters - self.stage1_iters
        
        # 计算各阶段的学习率范围（基于original_base_lr）
        self._update_stage_lrs()
        
        super().__init__()

    def _update_stage_lrs(self):
        """更新各阶段的学习率范围"""
        base = self.original_base_lr if hasattr(self, 'original_base_lr') else self.base_lr
        if isinstance(base, list):
            base = base[0]  # 如果是列表，取第一个
            
        self.stage1_start_lr = base
        self.stage1_end_lr = base * self.stage1_end_ratio
        self.stage2_start_lr = base * self.stage2_start_ratio
        self.min_lr = base * self.min_lr_ratio

    def annealing_cos(self, start: float, end: float, factor: float, weight: float = 1.) -> float:
        """余弦退火函数（复用CosineRestartLr的逻辑）"""
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_stage_from_iter(self, iter_num: int):
        """根据迭代数确定当前阶段"""
        if iter_num < self.warmup_iters:
            return 0  # Warmup
        elif iter_num < self.warmup_iters + self.stage1_iters:
            return 1  # Stage1
        else:
            return 2  # Stage2

    def get_lr(self, iter_num: int, base_lr: float):
        """计算指定迭代数的学习率"""
        if iter_num >= self.max_iters:
            return self.min_lr
            
        stage = self.get_stage_from_iter(iter_num)
        
        if stage == 0:  # Warmup阶段
            return base_lr  # 保持base_lr不变
            
        elif stage == 1:  # Stage1阶段
            # 从base_lr轻微余弦衰减到stage1_end_lr
            if self.stage1_iters == 0:
                return self.stage1_start_lr
            stage1_iter = iter_num - self.warmup_iters
            progress = stage1_iter / self.stage1_iters
            return self.annealing_cos(self.stage1_start_lr, self.stage1_end_lr, progress)
            
        else:  # Stage2阶段
            # 从stage2_start_lr余弦衰减到min_lr
            if self.stage2_iters == 0:
                return self.stage2_start_lr
            stage2_iter = iter_num - self.warmup_iters - self.stage1_iters
            progress = stage2_iter / self.stage2_iters
            return self.annealing_cos(self.stage2_start_lr, self.min_lr, progress)

    def get_regular_lr(self, iter_num):
        """返回学习率列表（兼容CosineRestartLr接口）"""
        # 🔥 始终使用original_base_lr进行计算
        base_lr_for_calc = self.original_base_lr
        if isinstance(base_lr_for_calc, list):
            base_lr_for_calc = base_lr_for_calc[0]
        
        # 计算当前学习率
        current_lr = self.get_lr(iter_num, base_lr_for_calc)
        
        # 返回对应optimizer参数组数量的学习率列表
        if isinstance(self.base_lr, list):
            return [current_lr] * len(self.base_lr)
        else:
            return [current_lr]

    def _set_lr(self, optimizer, lr_groups):
        """设置optimizer的学习率（复用CosineRestartLr的逻辑）"""
        if isinstance(optimizer, dict):
            for k, optim_item in optimizer.items():
                for param_group, lr in zip(optim_item.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def set_init_lr(self, optimizer):
        """初始化base_lr（兼容CosineRestartLr接口）"""
        # 设置initial_lr
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        
        # 🔥 更新base_lr以兼容接口，但保持original_base_lr不变
        if len(optimizer.param_groups) > 1:
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups]
        else:
            self.base_lr = optimizer.param_groups[0]['initial_lr']
        
        # 🔥 确保original_base_lr与实际初始学习率一致
        if not hasattr(self, 'original_base_lr'):
            self.original_base_lr = self.base_lr
        
        # 重新计算各阶段学习率（基于实际的初始学习率）
        self._update_stage_lrs()

    def get_stage_info(self, iter_num):
        """获取当前阶段信息（用于日志）"""
        stage = self.get_stage_from_iter(iter_num)
        stage_names = ['Warmup', 'Stage1', 'Stage2']
        
        if stage == 0:
            stage_progress = iter_num
            stage_total = self.warmup_iters
        elif stage == 1:
            stage_progress = iter_num - self.warmup_iters
            stage_total = self.stage1_iters
        else:
            stage_progress = iter_num - self.warmup_iters - self.stage1_iters
            stage_total = self.stage2_iters
            
        return {
            'stage': stage,
            'stage_name': stage_names[stage],
            'stage_progress': stage_progress,
            'stage_total': stage_total,
            'stage_ratio': stage_progress / stage_total if stage_total > 0 else 1.0
        }

    def get_current_stage_name(self, iter_num):
        """获取当前阶段名称（简化版）"""
        stage = self.get_stage_from_iter(iter_num)
        stage_names = ['Warmup', 'Stage1', 'Stage2']
        return stage_names[stage]

    def is_stage_transition(self, iter_num):
        """检查是否处于阶段转换点"""
        return (iter_num == self.warmup_iters or 
                iter_num == self.warmup_iters + self.stage1_iters)

    def get_stage_summary(self):
        """获取所有阶段的摘要信息"""
        return {
            'warmup': {
                'iters': self.warmup_iters,
                'ratio': self.warmup_ratio,
                'lr': self.original_base_lr,  # 保持base_lr不变
            },
            'stage1': {
                'iters': self.stage1_iters,
                'ratio': self.stage1_ratio,
                'lr_range': (self.stage1_start_lr, self.stage1_end_lr)
            },
            'stage2': {
                'iters': self.stage2_iters,
                'ratio': 1.0 - self.warmup_ratio - self.stage1_ratio,
                'lr_range': (self.stage2_start_lr, self.min_lr)
            },
            'total_iters': self.max_iters
        }

    def __repr__(self):
        return (f"ThreeStageDistillLr("
                f"base_lr={self.original_base_lr}, "
                f"max_iters={self.max_iters}, "
                f"warmup_ratio={self.warmup_ratio}, "
                f"stage1_ratio={self.stage1_ratio}, "
                f"stage2_start_ratio={self.stage2_start_ratio}, "
                f"warmup_iters={self.warmup_iters}, "
                f"stage1_iters={self.stage1_iters}, "
                f"stage2_iters={self.stage2_iters})")

    def __str__(self):
        """友好的字符串表示"""
        summary = self.get_stage_summary()
        return (f"ThreeStageDistillLr Schedule:\n"
                f"  Warmup:  {summary['warmup']['iters']:6d} iters ({summary['warmup']['ratio']*100:4.1f}%) "
                f"- LR: {summary['warmup']['lr']:.6f}\n"
                f"  Stage1:  {summary['stage1']['iters']:6d} iters ({summary['stage1']['ratio']*100:4.1f}%) "
                f"- LR: {summary['stage1']['lr_range'][0]:.6f} → {summary['stage1']['lr_range'][1]:.6f}\n"
                f"  Stage2:  {summary['stage2']['iters']:6d} iters ({summary['stage2']['ratio']*100:4.1f}%) "
                f"- LR: {summary['stage2']['lr_range'][0]:.6f} → {summary['stage2']['lr_range'][1]:.6f}\n"
                f"  Total:   {summary['total_iters']:6d} iters")      

def load_teacher_model(teacher_config, device='cuda'):
    if not teacher_config.get('teacher_pretrained'):
        raise ValueError("Teacher pretrained path not provided!")
    
    teacher_model_config = teacher_config.copy()
    teacher_model_config['model_type'] = teacher_config.get('teacher_model_type', 'GPDL')
    teacher_model_config['pretrained'] = teacher_config.get('teacher_pretrained', None)
    
    teacher_model = build_model(teacher_model_config)
    
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    if device != 'cpu':
        teacher_model = teacher_model.cuda()
    
    return teacher_model

def train():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))
    
    distill_mode = arg_dict.get('distill', False)

    # --- START LOGGING SETUP ---
    log_dir = arg_dict.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_name = arg_dict.get('task', 'unknown_task')
    if distill_mode:
        log_file_name = f"train_log_{task_name}_distill_{current_time_str}.txt"
    else:
        log_file_name = f"train_log_{task_name}_{current_time_str}.txt"
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

    teacher_model = None
    if distill_mode:
        log_message('===> Loading teacher model for distillation')
        device = 'cpu' if arg_dict['cpu'] else 'cuda'
        teacher_model = load_teacher_model(arg_dict, device)

    log_message('===> Building model')
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()

    if distill_mode:
        from utils.losses import DistillationLoss
        loss = DistillationLoss(
            loss_weight=arg_dict.get('loss_weight', 100.0),
            alpha=arg_dict.get('distill_alpha', 0.8)
        )
        log_message(f"Using DistillationLoss with alpha={arg_dict.get('distill_alpha', 0.8)}")
    else:
        loss = build_loss(arg_dict)
        log_message(f"Using {arg_dict.get('loss_type', 'MSELoss')}")

    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    if distill_mode:
        lr_scheduler = ThreeStageDistillLr(
            base_lr=arg_dict['lr'],
            max_iters=arg_dict['max_iters'],
            warmup_ratio=arg_dict.get('warmup_ratio', 0.05),      # 5%
            stage1_ratio=arg_dict.get('stage1_ratio', 0.05),      # 5%
            stage1_end_ratio=arg_dict.get('stage1_end_ratio', 0.8),    # Stage1结束时80%
            stage2_start_ratio=arg_dict.get('stage2_start_ratio', 0.5), # Stage2开始时50%
            min_lr_ratio=arg_dict.get('min_lr_ratio', 0.01)       # 最小1%
        )
        lr_scheduler.set_init_lr(optimizer)
        
        log_message(f"Three-stage distillation learning rate schedule:")
        log_message(f"  Warmup ({lr_scheduler.warmup_iters} iters): {arg_dict['lr']:.6f}")
        log_message(f"  Stage1 ({lr_scheduler.stage1_iters} iters): {lr_scheduler.stage1_start_lr:.6f} → {lr_scheduler.stage1_end_lr:.6f}")
        log_message(f"  Stage2 ({lr_scheduler.stage2_iters} iters): {lr_scheduler.stage2_start_lr:.6f} → {lr_scheduler.min_lr:.6f}")
    else:
        lr_scheduler = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
        lr_scheduler.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000

    log_message(f"Starting training for {arg_dict['max_iters']} iterations.", console_print=True)
    log_message(f"Logging training progress every {print_freq} iterations.", console_print=False)
    log_message(f"Saving checkpoints every {save_freq} iterations to {arg_dict['save_path']}", console_print=False)

    try:
        while iter_num < arg_dict['max_iters']:
            if distill_mode:
                stage_info = lr_scheduler.get_stage_info(iter_num)
                current_stage = stage_info['stage']
                
                if iter_num == lr_scheduler.warmup_iters:
                    log_message("🔄 Switching from Warmup to Stage 1: Pure distillation")
                elif iter_num == lr_scheduler.warmup_iters + lr_scheduler.stage1_iters:
                    log_message("🔄 Switching to Stage 2: Mixed training (teacher + ground truth)")

            with tqdm(total=print_freq, desc=f"Iter {iter_num}/{arg_dict['max_iters']}") as bar:
                for feature, label, _ in dataset:        
                    if arg_dict['cpu']:
                        input_data, target_data = feature, label
                    else:
                        input_data, target_data = feature.cuda(), label.cuda()

                    regular_lr_list = lr_scheduler.get_regular_lr(iter_num)
                    lr_scheduler._set_lr(optimizer, regular_lr_list)
                    current_lr_for_log = optimizer.param_groups[0]['lr']

                    prediction = model(input_data)
                    optimizer.zero_grad()
                    
                    if distill_mode:
                        if current_stage == 0:  # Warmup
                            pixel_loss = loss(prediction, teacher_output=None, ground_truth=target_data)
                        else:
                            with torch.no_grad():
                                teacher_prediction = teacher_model(input_data)
                            if current_stage == 1:  # Stage1
                                pixel_loss = loss(prediction, teacher_output=teacher_prediction, ground_truth=None)
                            else:  # Stage2
                                pixel_loss = loss(prediction, teacher_output=teacher_prediction, ground_truth=target_data)
                    else:
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
            
            log_message(f"===> Iters[{iter_num}]({iter_num}/{arg_dict['max_iters']}): Avg Loss: {avg_loss_for_period:.4f}, Current LR: {current_lr_for_log:.6e}", console_print=True)
            
            if iter_num % save_freq == 0:
                checkpoint(model, iter_num, arg_dict['save_path'])
                log_message(f"Checkpoint saved for iter {iter_num:,}.", console_print=False)
            
            epoch_loss = 0

        log_message("Training finished successfully.", console_print=True)

    except Exception as e:
        log_message(f"AN ERROR OCCURRED DURING TRAINING: {str(e)}", console_print=True)
        import traceback
        log_message(traceback.format_exc(), console_print=False)
    finally:
        if _log_file_handle:
            log_message("Closing log file.", console_print=False)
            _log_file_handle.close()


if __name__ == "__main__":
    train()