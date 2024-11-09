import numpy as np
import pickle
import os
import sys
import torch
import dgl
from dgl.transforms import add_self_loop, metis_partition
from multiprocessing import Process

def run(run_list, save_root_path):
    for i in run_list:
        print(i)
        gen_cell(i[0],i[1],i[2],save_root_path)

def gen_cell(name,
             gcell_size=1,
             root=None,
             save_root_path=None):
    print(f'process {name}')
    if name.split('-')[2] == 'FPU':
        base_name = '-'.join(name.split('-')[1:6])
    elif name.split('-')[1] == 'zero':
        base_name = '-'.join(name.split('-')[1:6])
    else:
        base_name = '-'.join(name.split('-')[1:5])
    # load data
    place_path = f'{root}/instance_placement_gcell/{name}'
    path_net_attr = f'{root}/graph_information/net_attr/{base_name}_net_attr.npy'
    path_node_attr = f'{root}/graph_information/node_attr/{base_name}_node_attr.npy'
    path_pin_attr = f'{root}/graph_information/pin_attr/{base_name}_pin_attr.npy'
    out_instance_placement = np.load(place_path, allow_pickle=True).item()
    out_net_attr = np.load(path_net_attr, allow_pickle=True)[0]
    out_node_attr = np.load(path_node_attr, allow_pickle=True)[0]
    out_pin_attr = np.load(path_pin_attr, allow_pickle=True)

    # take the intersection between graph_information and instance_placement, see FAQ #4
    node_name_fail_list = []
    fail_nodes = 0
    for index in range(out_pin_attr.shape[1]):
        net_name = out_pin_attr[1][index]
        node_name = out_pin_attr[2][index]
        if node_name in node_name_fail_list:
            continue
        if out_node_attr[node_name] not in out_instance_placement:
            fail_nodes +=1
            node_name_fail_list.append(node_name)
            continue

    # build a new mapping for node and net after taking intersection
    node_map = {}
    node_unique_idx = np.unique(out_pin_attr[2], return_counts=True)[0]
    node_unique = np.unique(out_pin_attr[2], return_counts=True)[1]
    count = 0
    for index in node_unique_idx:
        if index not in node_name_fail_list:
            node_map[index] = count
            count += 1
    num_nodes = len(node_unique_idx) - fail_nodes
    assert count == num_nodes
    net_name_list = []
    for index in range(out_pin_attr.shape[1]):
        net_name = out_pin_attr[1][index]
        node_name = out_pin_attr[2][index]
        if node_name in node_name_fail_list:
            continue
        if isinstance(net_name, list):
            net_name_list.extend(net_name)
        elif isinstance(net_name, int): 
            net_name_list.append(net_name)

    net_map = {}
    net_name_list = np.unique(net_name_list)
    net_name_list.sort()
    count = 0
    for index in range(len(net_name_list)):
        net_name = net_name_list[index]
        if net_name not in net_map:
            net_map[net_name] = count
            count += 1
    assert len(net_name_list) == count

    # read node position from instance_placement as node_feature
    net_set = {}
    for index in range(out_pin_attr.shape[1]):
        net_name = out_pin_attr[1][index]
        node_name = out_pin_attr[2][index]
        if node_name in node_name_fail_list:
            continue
        node_name_mapped = node_map[node_name]
        pin_count = node_unique[node_name_mapped]
        if isinstance(net_name, list):
            for net_name_index in net_name:
                net_name_index_mapped = net_map[net_name_index]
                if net_name_index_mapped not in net_set:
                    net_set.setdefault(net_name_index_mapped, [])
                node_position_left = out_instance_placement[out_node_attr[node_name]][0]
                node_position_bottom = out_instance_placement[out_node_attr[node_name]][1]
                node_position_right = out_instance_placement[out_node_attr[node_name]][2]
                node_position_top = out_instance_placement[out_node_attr[node_name]][3]
                center_x_node = (node_position_right-node_position_left)/2.0
                center_y_node = (node_position_top-node_position_bottom)/2.0
                net_set[net_name_index_mapped].append([node_name_mapped] + [center_x_node,center_y_node]+[node_position_left,node_position_bottom,node_position_right,node_position_top]+[pin_count])
        elif isinstance(net_name, int): 
            net_name_mapped = net_map[net_name]
            if net_name_mapped not in net_set:
                net_set.setdefault(net_name_mapped, [])
            node_position_left = out_instance_placement[out_node_attr[node_name]][0]
            node_position_bottom = out_instance_placement[out_node_attr[node_name]][1]
            node_position_right = out_instance_placement[out_node_attr[node_name]][2]
            node_position_top = out_instance_placement[out_node_attr[node_name]][3]
            center_x_node = (node_position_right-node_position_left)/2.0
            center_y_node = (node_position_top-node_position_bottom)/2.0
            net_set[net_name_mapped].append([node_name_mapped]+[center_x_node,center_y_node]+[node_position_left, node_position_bottom, node_position_right, node_position_top]+[pin_count])
        else:
            raise ValueError('net_name must be list or int')

    # build graph
    us, vs = [], []
    for net, list_node_feats in net_set.items():
        nodes = [node_feats[0] for node_feats in list_node_feats]
        us_, vs_ = node_pairs_among(nodes, max_cap=8)
        us.extend(us_)
        vs.extend(vs_)
    homo_graph = dgl.graph((us, vs))

    dgl.save_graphs(os.path.join(save_root_path, base_name),homo_graph)

def node_pairs_among(nodes, max_cap=-1):
    us = []
    vs = []
    if max_cap == -1 or len(nodes) <= max_cap:
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                us.append(u)
                vs.append(v)
    else:
        for u in nodes:
            vs_ = np.random.permutation(nodes)
            left = max_cap - 1
            for v_ in vs_:
                if left == 0:
                    break
                if u == v_:
                    continue
                us.append(u)
                vs.append(v_)
                left -= 1
    return us, vs

def divide_n(list_in, n):
    list_out = [ [] for i in range(n)]
    for i,e in enumerate(list_in):
        list_out[i%n].append(e)
    return list_out
    
def read_csv(file):
    infos = []
    with open(file, 'r') as fin:
        for line in fin:
            name = line.strip()
            infos.append(name)
    return infos

if __name__ == "__main__":
    circuitnet_root = '.'
    save_root_path = './N28'
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    name_list = read_csv('./selected.csv')
    name_list = [(name, 1, circuitnet_root) for name in name_list]
    nlist = divide_n(name_list, 50) 

    process = []
    for name_idx in nlist:
        p = Process(target=run, args=(name_idx,save_root_path))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
