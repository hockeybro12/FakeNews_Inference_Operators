def deregister_torch_ipc():
    from multiprocessing.reduction import ForkingPickler
    import torch
    ForkingPickler._exi_reducers.pop(torch.cuda.Event)
    for t in torch._storage_classes:
        ForkingPickler._extra_reducers.pop(t)
    for t in torch._tensor_classes:
        ForkingPickler._extra_reducers.pop(t)
    ForkingPickler._extra_reducers.pop(torch.Tensor)
    ForkingPickler._extra_reducers.pop(torch.nn.parameter.Parameter)
import torch
import math  
import sys 
torch.manual_seed(0)
import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

# from yellowbrick.text import TSNEVisualizer
# from yellowbrick.datasets import load_hobbies
# deregister_torch_ipc()

# Contruct a two-layer GNN model
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('once')


import torch
torch.manual_seed(0)
sk_learn_seed = 5
import numpy as np

np.random.seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# dgl.random.seed(0)


class FakeNewsRGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, canonical_etypes, num_workers, n_layers=2, dropout=0.25, activation=None, conv_type='gcn'):
        super(FakeNewsRGCN, self).__init__()

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_workers = num_workers
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.num_heads = 1

        self.layers = nn.ModuleList()
        if self.conv_type == 'gcn':
            self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.GraphConv(in_feats[utype], hid_feats, norm='right', bias=True, weight=True) for utype, etype, vtype in canonical_etypes}))
        elif self.conv_type == 'sageconv':
            self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.SAGEConv(in_feats[utype], hid_feats, feat_drop=0.25, bias=True, activation=None, aggregator_type='gcn') for utype, etype, vtype in canonical_etypes}))
        elif self.conv_type == 'gatconv':
            self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.GATConv(in_feats[utype], hid_feats, feat_drop=0.25, attn_drop=0.25, activation=None, num_heads=self.num_heads) for utype, etype, vtype in canonical_etypes}))
        self.dropout = nn.Dropout(p=0.35)
        for i in range(1, n_layers):
            if i == (n_layers - 1):
                if self.conv_type == 'gcn':
                    self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.GraphConv(hid_feats, out_feats, norm='right') for _, etype, _ in canonical_etypes}))
                elif self.conv_type == 'sageconv':
                    self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.SAGEConv(hid_feats, out_feats, feat_drop=0.25, bias=True, activation=None, aggregator_type='gcn') for _, etype, _ in canonical_etypes}))
                elif self.conv_type == 'gatconv':
                    self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.GATConv(hid_feats*self.num_heads, out_feats, feat_drop=0.25, attn_drop=0.25, activation=None, num_heads=self.num_heads) for _, etype, _ in canonical_etypes}))
            else:
                if self.conv_type == 'gcn':
                    self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.GraphConv(hid_feats, hid_feats, norm='right')for _, etype, _ in canonical_etypes}))
                elif self.conv_type == 'sageconv':
                    self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.SAGEConv(hid_feats, hid_feats, feat_drop=0.25, bias=True, activation=None, aggregator_type='gcn') for _, etype, _ in canonical_etypes}))
                elif self.conv_type == 'gatconv':
                    self.layers.append(dglnn.HeteroGraphConv({etype : dglnn.GATConv(hid_feats*self.num_heads, hid_feats, feat_drop=0.25, attn_drop=0.25, activation=None, num_heads=self.num_heads) for _, etype, _ in canonical_etypes}))


    def forward(self, blocks, inputs):
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                sys.stdout.flush()
                x = layer(blocks[0], inputs)
                if self.conv_type == 'gatconv':
                    # reshape them for GATCONV
                    x = {ntype : h.view(-1, self.num_heads*self.hid_feats) for ntype, h in x.items()}
                # RELU activation
                x = {ntype : F.relu(h) for ntype, h in x.items()}
            else:
                if self.conv_type == 'gcn':
                    x = {ntype : self.dropout(h) for ntype, h in x.items()}
                x = layer(blocks[i], x)
                if self.conv_type == 'gatconv':
                    # reshape them for GATCONV
                    if i != (self.n_layers - 1):
                        output_feat_size = self.hid_feats
                    else:
                        output_feat_size = self.out_feats
                    x = {ntype : h.view(-1, self.num_heads*output_feat_size) for ntype, h in x.items()}
                if i != (self.n_layers - 1):
                    # RELU activation
                    x = {ntype : F.relu(h) for ntype, h in x.items()}
        return x

    def inference(self, curr_g, x, batch_size, sampler):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        nodes = torch.arange(curr_g.number_of_nodes())
        curr_g = curr_g.to('cpu')

        for l, layer in enumerate(self.layers[:-1]):
            y = {k: torch.zeros(curr_g.number_of_nodes(k), self.hid_feats if l != self.n_layers - 1 else self.out_feats) for k in curr_g.ntypes}

            new_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                curr_g, {k: torch.arange(curr_g.number_of_nodes(k)) for k in curr_g.ntypes}, new_sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                block = blocks[0].to(torch.device('cuda'))

                h = {k: x[k][input_nodes[k].type(torch.LongTensor)].to(torch.device('cuda')) for k in input_nodes.keys() if k in x}
                
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k].type(torch.LongTensor)] = h[k].cpu()

            x = y

        return y

    def inference_all_layers(self, curr_g, x, batch_size, sampler):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        nodes = torch.arange(curr_g.number_of_nodes())
        curr_g = curr_g.to('cpu')

        # [self.conv1, self.conv2, self.conv3]
        # print((self.layers[:-1]))
        for l, layer in enumerate(self.layers):
            y = {k: torch.zeros(curr_g.number_of_nodes(k), self.hid_feats if l != self.n_layers - 1 else self.out_feats) for k in curr_g.ntypes}

            new_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                curr_g, {k: torch.arange(curr_g.number_of_nodes(k)) for k in curr_g.ntypes}, new_sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                block = blocks[0].to(torch.device('cuda'))

                h = {k: x[k][input_nodes[k].type(torch.LongTensor)].to(torch.device('cuda')) for k in input_nodes.keys() if k in x}
                
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k].type(torch.LongTensor)] = h[k].cpu()

            x = y

        return y




# for link prediction
class HeteroScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = x
            for etype in edge_subgraph.canonical_etypes:
                if edge_subgraph.num_edges(etype) <= 0:
                    continue
                edge_subgraph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']

class FakeNewsModel(nn.Module):
    # here we have a model that first computes the representation and then predicts the scores for the edges
    def __init__(self, in_features, hidden_features, out_features, canonical_etypes, num_workers, n_layers, conv_type):
        super().__init__()
        self.sage = FakeNewsRGCN(in_features, hidden_features, out_features, canonical_etypes, num_workers, n_layers=n_layers, conv_type=conv_type)
        self.pred = HeteroScorePredictor()


    def forward(self, blocks, x, g, neg_g):

        # positive_graph, negative_graph, input_features, ('source', 'has_follower', 'user')
        # run both part of the graph through it given the input feature
        x = self.sage(blocks, x)
        if g is None:
            return x
        pos_score = self.pred(g, x)
        neg_score = self.pred(neg_g, x)
        return pos_score, neg_score 

    # def forward(self, blocks, x):
    #     return self.sage(blocks, x)

    def pred_score_edges(self, blocks, x):
        x = self.sage(blocks, x)
        pos_score = self.pred(g, x)
        return pos_score

    def inference(self, g, x, batch_size, inference):
        return self.sage.inference(g, x, batch_size, inference)

    def inference_all_layers(self, g, x, batch_size, inference):
        return self.sage.inference_all_layers(g, x, batch_size, inference)


def compute_loss(pos_score, neg_score, canonical_etypes):
    # Margin loss
    all_losses = []
    for given_type in canonical_etypes:
        if given_type not in pos_score:
            continue
        n_edges = pos_score[given_type].shape[0]
        if n_edges == 0:
            continue
        all_losses.append((1 + neg_score[given_type].view(n_edges, -1) - pos_score[given_type].unsqueeze(1)).clamp(min=0).mean())
    return torch.stack(all_losses, dim=0).mean()

def cross_entropy_custom(x, labels):
    epsilon = 1 - math.log(2)
    y = torch.nn.functional.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50