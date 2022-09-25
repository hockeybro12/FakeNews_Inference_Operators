import torch
import faiss   
import sys 
torch.manual_seed(0)

# Contruct a two-layer GNN model
import numpy as np
import os
from tqdm.auto import tqdm
# import torch.multiprocessing as mp
# import multiprocessing as mp
from collections import Counter

import warnings
warnings.filterwarnings('once')

import torch
torch.manual_seed(0)
sk_learn_seed = 5
import numpy as np

np.random.seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_articles_dict(args, overall_graph, graph_style, training_set_to_use, curr_data_split_key, labels_dict, add_everything=False, recompute_with_new_graph=False, recompute_and_save=False, use_bias=False, computing_based_on_predicted_labels=False, labels_dict_predicted=None, dev_set_to_use=None, test_set_to_use=None):
    '''determine labels for articles by back propagating from sources'''

    g = overall_graph._g[0]

    articles_we_didnt_have = None

    
    training_set_to_use_dict = {}
    for x in training_set_to_use:
        training_set_to_use_dict[x] = 1
            

    article_name_labels_dict = {}
    article_source_mapping = {}
    article_id_name_mapping = {} 
    article_id_labels_dict = {}

    if add_everything:
        print("The article dict will include everything")
        sys.stdout.flush()
        # decide the path given the flags
        article_labels_dict_path = args.path_where_graph_is + str(curr_data_split_key) + 'article_id_labels_dict_everything.npy'
        if computing_based_on_predicted_labels:
            print("Computing articles dict based on predicted labels")
            article_labels_dict_path = args.path_where_graph_is + str(curr_data_split_key) + 'article_id_labels_dict_predicted_labels.npy'
        if not os.path.isfile(article_labels_dict_path) or recompute_with_new_graph or use_bias:
            print("We don't have the dict!")
            sys.stdout.flush()
            # go through each article
            for given_article, given_article_id in tqdm(overall_graph.articles_mapping_dict.items()):
                article_id_name_mapping[given_article_id-1] = given_article
                # determine which sources this article is connected to
                given_article_id = np.int64(given_article_id)
                edges_connecting_article_to_source = list(g.out_edges(given_article_id-1, etype='is_published_by')[1].cpu().numpy())
                most_common_article_label = []
                for given_source, given_source_id in overall_graph.sources_mapping_dict.items():
                    if 'good_source' in given_source or 'low_source' in given_source or 'mid_source' in given_source:
                        continue
                    # determine the label of those sources
                    given_source_wo_identifier = given_source.replace(overall_graph.source_name_identifier, '')
                    
                    if (given_source_id-1) in edges_connecting_article_to_source and given_source_wo_identifier in labels_dict:
                        # if the article connects to the source
                        if use_bias:
                            curr_source_label = labels_dict[given_source_wo_identifier][1]
                        else:
                            curr_source_label = labels_dict[given_source_wo_identifier][0]
                        most_common_article_label.append(curr_source_label)
                        article_source_mapping[given_article_id-1] = given_source_wo_identifier
                    elif given_source_wo_identifier not in labels_dict:
                        if dev_set_to_use is not None:
                            if given_source_wo_identifier in dev_set_to_use or given_source_wo_identifier in test_set_to_use or given_source_wo_identifier in training_set_to_use:
                                # this is a problem situation since we can't find the source
                                print(str(given_source_wo_identifier) + " source could not be found in the labels dict when computing the articles")
                        else:
                            # this is a problem situation since we can't find the source
                            print(str(given_source_wo_identifier) + " source could not be found in the labels dict when computing the articles")
                # append the most common label of this article
                if len(most_common_article_label) > 0:
                    article_id_labels_dict[given_article_id-1] = Counter(most_common_article_label).most_common(1)[0][0] 
                    article_name_labels_dict[given_article.replace(overall_graph.article_name_identifier, '')] = Counter(most_common_article_label).most_common(1)[0][0]
                elif articles_we_didnt_have is not None and given_article_id-1 in articles_we_didnt_have:
                    sys.stdout.flush()
                    print("We didn't have this article but it's an article for these many sources" + str(len(edges_connecting_article_to_source)))
                sys.stdout.flush()

            print("The dict is done!")
            sys.stdout.flush()
            # save the dict if necessary
            if not recompute_with_new_graph or recompute_and_save and not use_bias:
                print("Saving and the length is " + str(len(article_id_labels_dict)))

                if articles_we_didnt_have is not None:
                    for given_article_we_needed in articles_we_didnt_have:
                        if given_article_we_needed in article_id_labels_dict:
                            print("It's there now")
                        else:
                            print("NO")

                sys.stdout.flush()
                np.save(article_labels_dict_path , np.asarray(dict(article_id_labels_dict)))
                if computing_based_on_predicted_labels:
                    np.save(args.path_where_graph_is + str(curr_data_split_key) + 'article_source_mapping_predicted_labels.npy', np.asarray(dict(article_source_mapping)))
                    np.save(args.path_where_graph_is + str(curr_data_split_key) + 'article_id_name_mapping_predicted_labels.npy', np.asarray(dict(article_id_name_mapping)))
                np.save(args.path_where_graph_is + str(curr_data_split_key) + 'article_source_mapping.npy', np.asarray(dict(article_source_mapping)))                    
                np.save(args.path_where_graph_is + str(curr_data_split_key) + 'article_id_name_mapping.npy', np.asarray(dict(article_id_name_mapping)))
        else:
            # the dict exists, so just load it. This avoids recomputing every time
            article_id_labels_dict = overall_graph.load_dict(article_labels_dict_path)
            if computing_based_on_predicted_labels:
                article_source_mapping = overall_graph.load_dict(args.path_where_graph_is + str(curr_data_split_key) + 'article_source_mapping_predicted_labels.npy')
                article_id_name_mapping = overall_graph.load_dict(args.path_where_graph_is + str(curr_data_split_key) + 'article_id_name_mapping_predicted_labels.npy')
            article_source_mapping = overall_graph.load_dict(args.path_where_graph_is + str(curr_data_split_key) + 'article_source_mapping.npy')
            article_id_name_mapping = overall_graph.load_dict(args.path_where_graph_is + str(curr_data_split_key) + 'article_id_name_mapping.npy')

    return article_id_labels_dict, article_source_mapping, article_id_name_mapping

