import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class NetConv(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf, h1=16, h2=16):
        super().__init__()
        self.in_nf = in_nf
        self.in_ef = in_ef
        self.out_nf = out_nf
        self.h1 = h1
        self.h2 = h2
        self.MLP_msg_i2o = MLP(self.in_nf * 2 , 32, 32, 32, 1 + self.h1 + self.h2)
        self.MLP_reduce_o = MLP(self.in_nf + self.h1 + self.h2, 32, 32, 32, self.out_nf)
        self.MLP_msg_o2i = MLP(self.in_nf * 2, 32, 32, 32, 32, self.out_nf)
        self.MLP_readout = MLP(self.in_nf * 2, 32, 32, 32, 32, self.out_nf)


    def edge_readout(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf']], dim=1)  # source destination
        x = self.MLP_readout(x)
        return {'nef': x}
    
    def edge_msg_i(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf']], dim=1)
        x = self.MLP_msg_o2i(x)
        return {'efi': x}
    
    def edge_msg_o(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf']], dim=1)
        x = self.MLP_msg_i2o(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'efo1': f1 * k, 'efo2': f2 * k}

    def node_reduce_o(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfo1'], nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_o(x)
        return {'new_nf': x}

    def forward(self, g, nf):
        with g.local_scope():
            g.ndata['nf'] = nf
            # input nodes
            g.apply_edges(self.edge_readout, etype='net_out')
            g.update_all(self.edge_msg_i, fn.sum('efi', 'new_nf'), etype='net_out')
            # output nodes
            g.apply_edges(self.edge_msg_o, etype='net_in')
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.sum('efo1', 'nfo1'), etype='net_in')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.max('efo2', 'nfo2'), etype='net_in')
            g.apply_nodes(self.node_reduce_o)
            
            return g.ndata['new_nf'], g.edges['net_out'].data['nef'] 

class TimingGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc1 = NetConv(4, 0, 16)
        self.nc2 = NetConv(16, 0, 16)
        self.nc3 = NetConv(16, 0, 4)

    def forward(self, g):
        nf0 = g.ndata['nf']
        x, _ = self.nc1(g, nf0)
        x, _ = self.nc2(g, x)
        _, net_delays = self.nc3(g, x)
        return net_delays