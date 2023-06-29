import torch
import dgl
import random
import os 
random.seed(8026728)

def load_data(args):
    available_data = os.listdir(args.data_path)

    available_data_temp = []
    for i in available_data:
        if not 'zero' in i:
            available_data_temp.append(i)
    train_data_keys = random.sample(available_data_temp, args.train_data_number)
    test_data_keys = [ i for i in available_data if i not in train_data_keys ]
    test_data_keys = random.sample(test_data_keys, args.test_data_number)
    data = {}
    for k in available_data:
        g = dgl.load_graphs('graph/{}'.format(k))[0][0].to('cuda')
        g.edges['net_out'].data['net_delays_log'] = (torch.log(0.0001 + g.edges['net_out'].data['net_delay']) + 9.211) # log(0.0001) ≈ -9.2103
        data[k] = g
    data_train = {k: t for k, t in data.items() if k in train_data_keys}
    data_test = {k: t for k, t in data.items() if k in test_data_keys}
    return data_train, data_test
