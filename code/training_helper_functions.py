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
import pickle
from random import randint
import subprocess
import math
import faiss   
import sys 
torch.manual_seed(0)
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.parallel import DistributedDataParallel
import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)
import scipy
import datetime

# from yellowbrick.text import TSNEVisualizer
# from yellowbrick.datasets import load_hobbies
# deregister_torch_ipc()

# Contruct a two-layer GNN model
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import graph_helper_functions
import dgl.function as fn
from collections import defaultdict
import sklearn.linear_model as lm
import sklearn.metrics as skm
import sklearn
import json
import os
from tqdm.auto import tqdm
# import torch.multiprocessing as mp
# import multiprocessing as mp
import torch.multiprocessing as mp
from _thread import start_new_thread
from functools import wraps
import traceback
from collections import Counter
from torch.utils.data import DataLoader
import scipy
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('once')
from sklearn.neural_network import MLPClassifier
import random
# from yellowbrick.style import set_palette
from torch.nn.utils import clip_grad_norm_
import glob

import torch
torch.manual_seed(0)
sk_learn_seed = 5
import numpy as np
from sklearn import metrics

from sklearn.mixture import GaussianMixture
np.random.seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')


def get_features_given_graph(g, args):
    '''given a graph, get the features that can be used to train the model'''

    source_feats = g.nodes['source'].data['source_embedding'].to(torch.device('cuda'))
    user_feats = g.nodes['user'].data['user_embedding'].to(torch.device('cuda'))
    article_feats = g.nodes['article'].data['article_embedding'].to(torch.device('cuda'))
    node_features_for_inference = {'source': source_feats, 'user': user_feats, 'article': article_feats}

    return node_features_for_inference


def get_features_given_blocks(g, args, blocks):
    '''given blocks, get features that can be used to train the graph'''
    node_features = {'source': blocks[0].srcdata['source_embedding']['source'].to(torch.device('cuda')), 'user': blocks[0].srcdata['user_embedding']['user'].to(torch.device('cuda')), 'article': blocks[0].srcdata['article_embedding']['article'].to(torch.device('cuda'))}

    return node_features


def get_train_mask_nids(args, overall_graph, training_set_to_use, curr_data_split, dev_set_to_use, graph_style, use_dev_set, curr_data_split_key=None):
    '''given a graph, get the training/test/dev masks and node ID's that we are going to train/test/dev on. For FANG it's document nodes, otherwise it's sources'''


    train_mask = np.zeros(overall_graph._g[0].number_of_nodes(ntype='source'))
    dev_mask = np.zeros(overall_graph._g[0].number_of_nodes(ntype='source'))
    test_mask = np.zeros(overall_graph._g[0].number_of_nodes(ntype='source'))
    train_nids = []
    dev_nids = []
    test_nids = []
    sources_used = 0

    for given_source_identifier_combo, given_source_id in overall_graph.sources_mapping_dict.items():
        given_source = given_source_identifier_combo.replace(overall_graph.source_name_identifier, '')

        if given_source in training_set_to_use:
            test_mask[given_source_id-1] = 0
            train_mask[given_source_id-1] = 1
            if use_dev_set:
                dev_mask[given_source_id-1] = 0
            train_nids.append(given_source_id-1)
            sources_used += 1
        elif given_source in curr_data_split['test']:
            test_mask[given_source_id-1] = 1
            train_mask[given_source_id-1] = 0
            if use_dev_set:
                dev_mask[given_source_id-1] = 0
            test_nids.append(given_source_id-1)
            sources_used += 1
        elif use_dev_set and given_source in dev_set_to_use:
            test_mask[given_source_id-1] = 0
            train_mask[given_source_id-1] = 0
            if use_dev_set:
                dev_mask[given_source_id-1] = 1
                dev_nids.append(given_source_id-1)
            sources_used += 1
        else:
            # print(str(given_source) + " cannot be found.")
            train_mask[given_source_id-1] = 0

    return train_mask, dev_mask, test_mask, train_nids, dev_nids, test_nids
