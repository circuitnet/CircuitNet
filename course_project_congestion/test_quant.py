from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np

import copy
import torch

from tqdm import tqdm
from datetime import datetime

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser

from utils.quant import get_default_quant_config, convert_to_quantized_model, calibrate_model_no_activation_data

def test(model, device, dataset, arg_dict):
    model = model.to(device)
    model.eval()

    # Build metrics
    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if device.type == 'cpu':
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            prediction = model(input)
            for metric, metric_func in metrics.items():
                if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                    avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

            bar.update(1)
    
    # for metric, avg_metric in avg_metrics.items():
    #     print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset))) 
    return {metric: avg_metric / len(dataset) for metric, avg_metric in avg_metrics.items()}


def test_quant():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test'] 
    arg_dict['test_mode'] = True

    # --- START LOGGING SETUP ---
    log_dir = arg_dict.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"quant_log_{arg_dict.get('task', 'unknown_task')}_{current_time_str}.txt"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    _log_file_handle = open(log_file_path, 'a')

    def log_message(message, console_print=True):
        timestamped_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        if console_print:
            print(message)
        _log_file_handle.write(timestamped_message + "\n")
        _log_file_handle.flush()
    # --- END LOGGING SETUP ---
    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    log_message('===> Building model')
    # Initialize model parameters
    fp32_model = build_model(arg_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    fp32_model = fp32_model.to(device)

    # # --- Test FP32 model once for baseline ---
    # log_message("\n--- Testing FP32 Baseline Model ---")
    # results = test(fp32_model, device, dataset, arg_dict)
    # log_message("FP32 Baseline Model Results:")
    # for metric, value in results.items():
    #     log_message(f"FP32 Baseline Model - {metric}: {value:.4f}")

    # --- Define Experiments (same as before, ensure paths are correct for GPDL) ---
    # QUANT_BITS = 8
    QUANT_BITS = arg_dict.get('quant_bits', 8)  # Use provided quant_bits or default to 8
    
    norm_layer_paths = [
        'encoder.c1.main.1', 'encoder.c1.main.4', 'encoder.c2.main.1', 'encoder.c2.main.4', 'encoder.c3.1',
        'decoder.conv1.main.1', 'decoder.conv1.main.4', 'decoder.upc1.main.1',
        'decoder.conv2.main.1', 'decoder.conv2.main.4', 'decoder.upc2.main.1'
    ]
    conv_layer_paths = [
        'encoder.c1.main.0', 'encoder.c1.main.3', 'encoder.c2.main.0', 'encoder.c2.main.3', 'encoder.c3.0',
        'decoder.conv1.main.0', 'decoder.conv1.main.3', 'decoder.conv2.main.0', 'decoder.conv2.main.3', 'decoder.conv3.0'
    ]
    conv_transpose_layer_paths = ['decoder.upc1.main.0', 'decoder.upc2.main.0']

    experiment_definitions = [
        {"name": f"1_Encoder_W{QUANT_BITS}P{QUANT_BITS}_A32", "config_overrides": {'weight_bits': QUANT_BITS, 'input_bits': 32, 'param_bits': QUANT_BITS}, "target_paths": ['encoder']},
        {"name": f"2_Decoder_W{QUANT_BITS}P{QUANT_BITS}_A32", "config_overrides": {'weight_bits': QUANT_BITS, 'input_bits': 32, 'param_bits': QUANT_BITS}, "target_paths": ['decoder']},
        {"name": f"3_EncoderDecoder_W{QUANT_BITS}P{QUANT_BITS}_A32", "config_overrides": {'weight_bits': QUANT_BITS, 'input_bits': 32, 'param_bits': QUANT_BITS}, "target_paths": ['encoder', 'decoder']},
        {"name": f"4_Activations_A{QUANT_BITS}_W32P32", "config_overrides": {'weight_bits': 32, 'input_bits': QUANT_BITS, 'param_bits': 32}, "target_paths": None},
        {"name": f"5_All_W{QUANT_BITS}P{QUANT_BITS}A{QUANT_BITS}", "config_overrides": {'weight_bits': QUANT_BITS, 'input_bits': QUANT_BITS, 'param_bits': QUANT_BITS}, "target_paths": None},
        {"name": f"6_Only_Conv2D_Weights_W{QUANT_BITS}", "config_overrides": {'weight_bits': QUANT_BITS, 'input_bits': 32, 'param_bits': 32}, "target_paths": conv_layer_paths},
        {"name": f"7_Only_ConvTranspose2D_Weights_W{QUANT_BITS}", "config_overrides": {'weight_bits': QUANT_BITS, 'input_bits': 32, 'param_bits': 32}, "target_paths": conv_transpose_layer_paths},
        {"name": f"8_Only_Norm_Params_P{QUANT_BITS}", "config_overrides": {'weight_bits': 32, 'input_bits': 32, 'param_bits': QUANT_BITS}, "target_paths": norm_layer_paths}
    ]

    # --- Run Experiments ---
    for exp_def in experiment_definitions:
        current_experiment_name = exp_def["name"]
        log_message(f"\n--- Running Experiment: {current_experiment_name} ---")

        quantized_model = copy.deepcopy(fp32_model) # Start from fresh FP32
        quantized_model.to(device)

        # Define quant_config for this specific experiment
        quant_config = get_default_quant_config( # Start with FP32 defaults
            weight_bits=32, input_bits=32, param_bits=32,
            learnable_clip=False # Usually False for PTQ
        )
        quant_config.update(exp_def["config_overrides"]) # Apply experiment specifics
        target_module_paths_for_exp = exp_def["target_paths"]

        log_message(f"  Quantization Config: {quant_config}")
        log_message(f"  Target Module Paths: {target_module_paths_for_exp}")

        convert_to_quantized_model(quantized_model, quant_config,
                                   target_module_paths=target_module_paths_for_exp)
        quantized_model.to(device)
        # log_message(quantized_model) # Optional: to verify structure

        calibrate_model_no_activation_data(quantized_model, device, quant_config)

        # Test the quantized model for this experiment
        log_message(f"\n--- Testing Quantized Model: {current_experiment_name} ---")
        result = test(quantized_model, device, dataset, arg_dict)
        for metric, value in result.items():
            log_message(f"  {metric}: {value:.4f}")

    log_message("-" * 50)
    log_message("Quantization Test Script Finished.")

if __name__ == "__main__":
    test_quant()