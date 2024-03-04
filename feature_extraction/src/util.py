import os
import numpy as np
import binascii

def instance_direction_rect(line): # used when we only need bounding box (rect) of the cell.

    if 'N' in line or 'S' in line:
        m_direction = (1, 0, 0, 1)
    elif 'W' in line or 'E' in line:
        m_direction = (0, 1, 1, 0)
    else:
        raise ValueError('read_macro_direction_wrong')
    return m_direction
    
def instance_direction_bottom_left(direction): # used when we need to get the bottom left corner of the cell.
    if   direction == 'N':
        i_direction = (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)
    elif direction == 'S':
        i_direction = (-1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 1)
    elif direction ==  'W':
        i_direction = (0, -1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)
    elif direction ==  'E':
        i_direction = (0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    elif direction == 'FN':
        i_direction = (-1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0)
    elif direction == 'FS':
        i_direction = (0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 1)
    elif direction == 'FW':
        i_direction = (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0)
    elif direction == 'FE':
        i_direction = (0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0)
    else:
        raise ValueError('read_macro_direction_wrong')
    return i_direction

def my_range(start, end):
    if start == end:
        return [start]
    if start != end:
        return range(start, end)

def save(root_path, dir_name, save_name, data):
    save_path = os.path.join(root_path, dir_name, save_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, data)

def save_npz(root_path, dir_name, save_name, data):
    save_path = os.path.join(root_path, dir_name, save_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savez_compressed(save_path, data)

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]
        
def divide_n(list_in, n):
    list_out = [ [] for i in range(n)]
    for i,e in enumerate(list_in):
        list_out[i%n].append(e)
    return list_out

def is_gzip_file(file_path):
    with open(file_path, 'rb') as file:
        header = file.read(2)
    hex_header = binascii.hexlify(header).decode('utf-8')
    if hex_header == '1f8b':
        return True
    else:
        return False
    