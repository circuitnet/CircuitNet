# Copyright 2022 CircuitNet. All rights reserved.

from functools import wraps
from inspect import getfullargspec

import os
import os.path as osp
import cv2
import numpy as np
import torch
import multiprocessing as mul
import uuid
import psutil
import time
import csv
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from scipy.interpolate import make_interp_spline
from functools import partial
from mmcv import scandir

from scipy.stats import wasserstein_distance
from skimage.metrics import normalized_root_mse
import math
import metrics

__all__ = ['psnr', 'ssim', 'nrms', 'emd']

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def input_converter(apply_to=None):
    def input_converter_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(tensor2img(args[i]))
                    else:
                        new_args.append(args[i])

            return old_func(*new_args)
        return new_func

    return input_converter_wrapper


@input_converter(apply_to=('img1', 'img2'))
def psnr(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse_value))


def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


@input_converter(apply_to=('img1', 'img2'))
def ssim(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


@input_converter(apply_to=('img1', 'img2'))
def nrms(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    nrmse_value = normalized_root_mse(img1.flatten(), img2.flatten(),normalization='min-max')
    if math.isinf(nrmse_value):
        return 0.05
    return nrmse_value



def get_histogram(img):
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / float(h * w)


def normalize_exposure(img):
    img = img.astype(int)
    hist = get_histogram(img)
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    sk = np.uint8(255 * cdf)
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


@input_converter(apply_to=('img1', 'img2'))
def emd(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    # change here
    img1 = normalize_exposure(np.squeeze(img1, axis = 2))
    img2 = normalize_exposure(np.squeeze(img2, axis = 2))
    hist_1 = get_histogram(img1)
    hist_2 = get_histogram(img2)

    emd_value = wasserstein_distance(hist_1, hist_2)
    return emd_value

def tpr(tp, fn):
    return tp/(tp+fn)

def fpr(fp, tn):
    return fp/(fp+tn)

def precision(tp, fp):
    return tp/(tp+fp)

def calculate_all(csv_path):
    tpr_sum_List = []
    fpr_sum_List = []
    precision_sum_List = []
    threshold_remain_list = []
    num = 0
    tpr_sum = 0
    fpr_sum = 0 
    precision_sum = 0

    csv_file = open(os.path.join(csv_path), 'r')

    first_flag = False
    for line in csv_file:
        threshold, idx, tn, fp, fn, tp = line.strip().split(',')
        if threshold not in threshold_remain_list:
            if first_flag:
                if num !=0:
                    tpr_sum_List.append(tpr_sum/num)
                    fpr_sum_List.append(fpr_sum/num)
                    precision_sum_List.append(precision_sum/num)
            threshold_remain_list.append(threshold)
            tpr_sum = 0
            fpr_sum = 0
            precision_sum = 0
            num = 0
            first_flag = True

        if int(fp)==0 and int(tn)==0:
            continue
        elif int(tp)==0 and int(fn)==0:
            continue
        elif int(tp)==0 and int(fp)==0:
            continue
        else:
            tpr_sum += tpr(int(tp), int(fn))
            fpr_sum += fpr(int(fp), int(tn))
            precision_sum += precision(int(tp), int(fp))
            num += 1
    if num !=0:
        tpr_sum_List.append(tpr_sum/num)
        fpr_sum_List.append(fpr_sum/num)
        precision_sum_List.append(precision_sum/num)
        

    return tpr_sum_List, fpr_sum_List, precision_sum_List


def calculated_score(threshold_idx=None, 
                     temp_path=None, 
                     label_path=None,
                     save_path=None,
                     threshold_label=None, 
                     preds=None):
    file = open(os.path.join(temp_path, f'tpr_fpr_{threshold_idx}.csv'),'w')
    f_csv = csv.writer(file, delimiter=',')
    for idx, pred in enumerate(preds):
        target_test = np.load(os.path.join(label_path, pred)).reshape(-1)
        target_probabilities = np.load(os.path.join(save_path, 'test_result', pred)).reshape(-1)

        target_test[target_test>=threshold_label] = 1
        target_test[target_test<threshold_label] = 0

        target_probabilities[target_probabilities>=threshold_idx] = 1
        target_probabilities[target_probabilities<threshold_idx] = 0

        if np.sum(target_probabilities == 0)==0 and np.sum(target_test == 0)==0:
            tn = 256*256
            tp, fn, fp = 0,0,0
        elif np.sum(target_probabilities == 1)==0 and np.sum(target_test == 1)==0:
            tp = 256*256
            tn, fn, fp = 0,0,0
        else:
            tn, fp, fn, tp = confusion_matrix(target_test, target_probabilities).ravel()

        f_csv.writerow([str(threshold_idx)]+[str(i) for i in [idx, tn, fp, fn, tp]])

    # f_csv.writerow([str(threshold_idx)]+[str(i) for i in [1, 2, 3, 4, 5]]) # only for test

    print(f'{threshold_idx}-done')

def multi_process_score(out_name=None, threashold=0.0, label_path=None, save_path=None):
    uid = str(uuid.uuid4())
    suid = ''.join(uid.split('-'))
    temp_path = f'./{suid}'

    psutil.cpu_percent(None)
    time.sleep(0.5)
    # sys.exit(0)
    pool = mul.Pool(int(mul.cpu_count()*(1-psutil.cpu_percent(None)/100.0)))
    # pool = mul.Pool(1)

    preds = scandir(os.path.join(save_path, 'test_result'), suffix='npy', recursive=True)
    preds = [v for v in preds]

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    threshold_list = np.linspace(0, 1, endpoint=False, num=200)
    
    calculated_score_parital = partial(calculated_score, temp_path=temp_path, 
                                        label_path=label_path, save_path=save_path, threshold_label=threashold, preds=preds)
    rel = pool.map(calculated_score_parital, threshold_list)
    
    print(f'{suid}')

    for list_i in threshold_list:
        fr=open(os.path.join(temp_path, f'tpr_fpr_{list_i}.csv'), 'r').read()
        with open(os.path.join(temp_path, f'{out_name}'), 'a') as f:
            f.write(fr)
        f.close()

    # if not os.path.exists(os.path.join(os.getcwd(), 'out')):
    #     os.makedirs(os.path.join(os.getcwd(), 'out'))

    # print('copying')
    # os.system('cp {} {}'.format(os.path.join(temp_path, f'{out_name}'), os.path.join(os.path.join(os.getcwd(), 'out'), f'{out_name}')))

    print('copying')
    os.system('cp {} {}'.format(os.path.join(temp_path, f'{out_name}'), os.path.join(os.path.join(os.getcwd(), save_path), f'{out_name}')))
    
    print('remove temp files')
    os.system(f'rm -rf {temp_path}')

def get_sorted_list(fpr_sum_List,tpr_sum_List):
    fpr_list = []
    tpr_list = []
    for i, j in zip(fpr_sum_List, tpr_sum_List):
        if i not in fpr_list:
            fpr_list.append(i)
            tpr_list.append(j)

    fpr_list.reverse()
    tpr_list.reverse()
    fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))
    return fpr_list, tpr_list


def roc_prc(save_path):
    tpr_sum_List, fpr_sum_List, precision_sum_List = calculate_all(os.path.join(os.getcwd(), save_path, 'roc_prc.csv'))

    fpr_list, tpr_list = get_sorted_list(fpr_sum_List,tpr_sum_List)
    fpr_list = list(fpr_list)
    fpr_list.extend([1])

    tpr_list = list(tpr_list)
    tpr_list.extend([1])

    roc_numerator = 0
    for i in range(len(tpr_list)-1):
        roc_numerator += (tpr_list[i]+tpr_list[i+1])*(fpr_list[i+1]-fpr_list[i])/2

    tpr_list, p_list = get_sorted_list(tpr_sum_List, precision_sum_List)
    x_smooth = np.linspace(0, 1, 25)
    y_smooth = make_interp_spline(tpr_list, p_list, k=3)(x_smooth)

    prc_numerator = 0
    for i in range(len(y_smooth)-1):
        prc_numerator += (y_smooth[i]+y_smooth[i+1])*(x_smooth[i+1]-x_smooth[i])/2

    return roc_numerator, prc_numerator



def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()

        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[:, :, :], (2, 0, 1))
            # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()[..., None]
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


def build_metric(metric_name):
    return metrics.__dict__[metric_name.lower()]


def build_roc_prc_metric(threashold=None, dataroot=None, ann_file=None, save_path=None, **kwargs):
    if ann_file:
        with open(ann_file, 'r') as fin:
            for line in fin:
                if len(line.strip().split(',')) == 2: 
                    feature, label = line.strip().split(',')
                else:
                    label = line.strip().split(',')[-1]
                break

        label_name = label.split('/')[0]
    else:
        raise FileExistsError
    print(os.path.join(dataroot, label_name))
    multi_process_score(out_name='roc_prc.csv', threashold=threashold, label_path=os.path.join(dataroot, label_name), save_path=os.path.join('.', save_path))
    
    return roc_prc(save_path)
