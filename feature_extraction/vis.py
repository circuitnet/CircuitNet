import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Process
from typing import List
from src.util import divide_n

def get_sub_path(path):
    sub_path = []
    if isinstance(path, List):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path

def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
    return result

def std(input):
    if input.max() == 0:
        return input
    else:
        result = (input-input.min()) / (input.max()-input.min())
        return result

def vis_data(args, path_list):
        feature_save_path = os.path.join(args.save_path)
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)
        for path in path_list:
            name = os.path.basename(path)
            feature = np.rot90(255 * std(np.load(path)), 1)
            cv2.imwrite('%s/%s.jpg'%(feature_save_path,name), feature)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default = './out/cell_density', type=str, help = 'path to the dataset')
    parser.add_argument("--save_path", default = './images/cell_density', type=str, help = 'path to save')

    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args = parse_args()

    path_list = get_sub_path(args.data_path)
    print('processing %s files' % len(path_list))

    nlist = divide_n(path_list, 10 )
    process = []
    for list in nlist:
        p = Process(target=vis_data, args=(args, list))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()


    





