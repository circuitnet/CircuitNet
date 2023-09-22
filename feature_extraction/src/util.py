import os
import numpy as np

def save(root_path, dir_name, save_name, data):
    save_path = os.path.join(root_path, dir_name, save_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, data)

def divide_n(list_in, n): # divide a list to n parts
    list_out = [ [] for i in range(n)]
    for i,e in enumerate(list_in):
        list_out[i%n].append(e)
    return list_out

    