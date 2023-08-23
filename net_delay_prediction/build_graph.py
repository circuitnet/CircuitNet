import os
import argparse
import numpy as np
import dgl
import torch
from multiprocessing import Process
from typing import List

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

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

def build_graph(args, path_list):
    for path in path_list:
        name = os.path.basename(path)
        net_edges = np.load(os.path.join(args.data_path, 'net_edges', name))['net_edges']
        nodes = np.load(os.path.join(args.data_path, 'nodes', name))['nodes']
        pin_positions = np.load(os.path.join(args.data_path, 'pin_positions', name), allow_pickle=True)['pin_positions'].item()
        g = dgl.heterograph({
        ('node', 'net_out', 'node'): (net_edges[:,0], net_edges[:,1]),
        ('node', 'net_in', 'node'): (net_edges[:,1], net_edges[:,0]),
        })

        g.edges['net_out'].data['net_delay'] = torch.tensor(net_edges[:,2:]).type(torch.float32)
        g.ndata['nf'] = torch.tensor([pin_positions[nodes[i.item()].replace('\\','')][0:4] for i in g.nodes()]).type(torch.float32)
        dgl.save_graphs('{}/{}.bin'.format(args.save_path, name), g)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
                                                             
    parser.add_argument("--task", default = 'net_delay', type=str, help = 'only support net delay for now' )
    parser.add_argument("--data_path", default = './post_route', type=str, help = 'path to the parent dir of features: nodes, net_edges, pin_positions')
    parser.add_argument("--save_path", default = './graph', type=str, help = 'path to save')

    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args = parse_args()
    feature_list = ['nodes', 'net_edges', 'pin_positions']

    name_list = get_sub_path(os.path.join(args.data_path, feature_list[0]))
    print('processing {} files'.format(len(name_list)))

    nlist = divide_list(name_list, 500 )
    process = []
    for list in nlist:
        p = Process(target=build_graph, args=(args, list))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
