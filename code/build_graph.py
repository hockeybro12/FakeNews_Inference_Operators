import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import graph_helper_functions
import dgl.function as fn
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='build_dgl_graph_and_save')
parser.add_argument('--save_path', type=str, default='PaperGraph_test_release_2/')
parser.add_argument('--users_that_follow_each_other_path', type=str, default='dataset_release/users_that_follow_each_other.npy')
parser.add_argument('--dataset_corpus', type=str, default='News-Media-Reliability/data/acl2020/corpus.tsv')
parser.add_argument('--user_id_to_representation_file', type=str, default='dataset_release/user_id_to_representation.tsv')
parser.add_argument('--source_name_to_representation_file', type=str, default='dataset_release/source_name_to_representation.tsv')
parser.add_argument('--article_to_representation_file', type=str, default='dataset_release/article_id_to_representation.tsv')
parser.add_argument('--article_name_to_id_file', type=str, default='dataset_release/article_name_to_id.npy')
parser.add_argument('--followers_dict_path', type=str, default='dataset_release/source_followers_dict.npy')
parser.add_argument('--articles_per_directory_path', type=str, default='dataset_release/articles_per_directory.npy')
parser.add_argument('--article_user_mapping_path', type=str, default='dataset_release/article_user_talkers.npy')



parser.add_argument('--building_original_graph_first', action='store_true', help="")

parser.add_argument('--path_where_data_is', type=str, default='News-Media-Reliability')

# TODO: release
parser.add_argument('--source_username_mapping_dict_path', type=str, default='dataset_release/domain_twitter_triplets_dict.npy')


args = parser.parse_args()

overall_graph = graph_helper_functions.FakeNewsDataset(save_path=args.save_path, users_that_follow_each_other_path=args.users_that_follow_each_other_path, dataset_corpus=args.dataset_corpus, followers_dict_path=args.followers_dict_path, source_username_mapping_dict_path=args.source_username_mapping_dict_path, building_original_graph_first=args.building_original_graph_first, path_where_data_is=args.path_where_data_is, user_id_to_representation_file=args.user_id_to_representation_file, source_name_to_representation_file=args.source_name_to_representation_file, article_to_representation_file=args.article_to_representation_file, articles_per_directory_path=args.articles_per_directory_path, article_user_mapping_path=args.article_user_mapping_path, article_name_to_id_file=args.article_name_to_id_file)


overall_graph.process()
overall_graph.save()