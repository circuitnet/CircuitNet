import torch
import numpy as np
import dgl
import torch.nn.functional as F
import random
import pdb
import time
import argparse
import os
from sklearn.metrics import r2_score

from data_graph import load_data
from model import TimingGCN

parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_iter', type=int,
    help='If specified, test a saved model instead of training')
parser.add_argument(
    '--checkpoint', type=str,
    help='If specified, the log and model would be saved to/loaded from that checkpoint directory')
parser.add_argument(
    '--data_path', type=str, default='./graph',
    help='Path to graphs.')
parser.add_argument(
    '--train_data_number', type=int, default=20,
    help='Specify number of data used for training')
parser.add_argument(
    '--test_data_number', type=int, default=20,
    help='Specify number of data used for testing')
parser.add_argument(
    '--batch_size', type=int, default=8,
    help='Specify batch size for training')
parser.add_argument(
    '--iteration', type=int, default=100000,
    help='Specify learning rate for training')
parser.add_argument(
    '--lr', type=float, default=0.0005,
    help='Specify learning rate for training')
torch.set_default_dtype(torch.float32)
model = TimingGCN()

model.cuda()

def test_netdelay(model):
    data_train, data_test = load_data(args)
    
    model.eval()
    with torch.no_grad():
        def test_dict(data):
            for k, g in data.items():
                pred = model(g)
                truth = g.edges['net_out'].data['net_delays_log']
                r2 = r2_score(pred.cpu().numpy().reshape(-1),
                              truth.cpu().numpy().reshape(-1))
                print('{:15} {}'.format(k, r2))
                
        print('======= Training dataset ======')
        test_dict(data_train)
        print('======= Test dataset ======')
        test_dict(data_test)

def train(model, args):
    data_train, data_test = load_data(args)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.iteration):
        model.train()
        train_loss_tot_net_delays = 0
        optimizer.zero_grad()

        for k, g in random.sample(data_train.items(), args.batch_size):
            pred_net_delays= model(g)
            loss_net_delays = 0

            loss_net_delays = F.mse_loss(pred_net_delays, g.edges['net_out'].data['net_delays_log'])
            train_loss_tot_net_delays += loss_net_delays.item()
            loss_net_delays.backward()
            
        optimizer.step()

        if e == 0 or e % 20 == 19:
            with torch.no_grad():
                model.eval()
                test_loss_tot_net_delays= 0
                for k, g in data_test.items():
                    pred_net_delays= model(g)

                    test_loss_tot_net_delays += F.mse_loss(pred_net_delays, g.edges['net_out'].data['net_delays_log']).item()

                print('Epoch {}, net delay {:.6f}/{:.6f})'.format(
                    e,
                    train_loss_tot_net_delays / args.batch_size,
                    test_loss_tot_net_delays / len(data_test)
                    )
                    )

            if e == 0 or e % 200 == 199 or (e > 6000 and test_loss_tot_net_delays / len(data_test) < 6):
                if args.checkpoint:
                    save_path = './checkpoints/{}/{}.pth'.format(args.checkpoint, e)
                    torch.save(model.state_dict(), save_path)
                    print('saved model to', save_path)
                try:
                    test_netdelay(model)
                except ValueError as e:
                    print(e)
                    print('Error testing, but ignored')

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.test_iter:
        assert args.checkpoint, 'no checkpoint dir specified'
        model.load_state_dict(torch.load('./checkpoints/{}/{}.pth'.format(args.checkpoint, args.test_iter)))
        test_netdelay(model)
        
    else:
        if args.checkpoint:
            print('saving logs and models to ./checkpoints/{}'.format(args.checkpoint))
            os.system('mkdir -p ./checkpoints/{}'.format(args.checkpoint))
            stdout_f = './checkpoints/{}/stdout.log'.format(args.checkpoint)
            stderr_f = './checkpoints/{}/stderr.log'.format(args.checkpoint)
            train(model, args)
        else:
            print('No checkpoint is specified. abandoning all model checkpoints and logs')
            train(model, args)

