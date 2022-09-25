from curses import newpad
import numpy as np 
import pandas as pd 
import pickle
import json
from collections import defaultdict
from tqdm.auto import tqdm
import tldextract
import random
import sys
import joblib
import glob
import os
import re

# for BERT
import torch
from pytorch_transformers import *

# DGL imports 
# to build a dataset
import dgl
from dgl import save_graphs, load_graphs
from dgl.data import DGLBuiltinDataset


class FakeNewsDataset(DGLBuiltinDataset):
    """FakeNewsDataset graph
    """

    def __init__(self, save_path=None, users_that_follow_each_other_path=None, dataset_corpus=None, followers_dict_path=None, source_username_mapping_dict_path=None, building_original_graph_first=False, path_where_data_is=None, source_name_to_representation_file=None, user_id_to_representation_file=None, articles_per_directory_path=None, article_to_representation_file=None, article_user_mapping_path=None, article_name_to_id_file=None):

        self.mode = 'graph'
        self.building_original_graph_first = building_original_graph_first
        self.dataset_corpus = dataset_corpus 
        self._save_dir = save_path
        self.followers_dict_path = followers_dict_path
        self.source_username_mapping_dict_path = source_username_mapping_dict_path
        self.users_that_follow_each_other_path = users_that_follow_each_other_path
        self.source_name_to_representation_file = source_name_to_representation_file
        self.user_id_to_representation_file = user_id_to_representation_file
        self.articles_per_directory_path = articles_per_directory_path
        self.article_to_representation_file = article_to_representation_file
        self.article_user_mapping_path = article_user_mapping_path
        self.article_name_to_id_file = article_name_to_id_file


        self.sources_in_corpus_domains = {}
        self.source_name_identifier = '__source_name'
        self.article_name_identifier = '__article_name'
        self.user_profile_identifier = '__user_profile'
        self.date_name_identifier = '__date_name'

        self.labels = {}
        self.corpus_df = pd.read_csv(self.dataset_corpus, sep='\t')

        self.path_where_data_is = path_where_data_is

    def load_dict(self, dict_path):
        out_dict = defaultdict(list)
        old_out_dict = np.load(dict_path, allow_pickle=True)
        out_dict.update(old_out_dict.item())
        return out_dict

    def process(self):

        print("Loading in the process function")
        sys.stdout.flush()

        print("Loading representations")
        if self.building_original_graph_first:
            # if we are building the graph and adding the sources we will need the representations
            self.source_name_to_representation = {}
            with open(self.source_name_to_representation_file, 'r+') as f:
                for line in f:
                    parts = line.split('\t')
                    new_parts = parts[1].replace('[', '').replace(']', '').replace('\n', '')
                    new_parts = " ".join(new_parts.split())
                    new_parts = re.sub(' +', ' ', new_parts).split(' ')
                    self.source_name_to_representation[parts[0]] = torch.tensor(list(map(float, new_parts)), dtype=torch.float32)
            self.user_id_to_representation = {}
            with open(self.user_id_to_representation_file, 'r+') as f:
                for line in tqdm(f):
                    parts = line.split('\t')
                    new_parts = parts[1].replace('[', '').replace(']', '').replace('\n', '')
                    new_parts = " ".join(new_parts.split())
                    new_parts = re.sub(' +', ' ', new_parts).split(' ')
                    self.user_id_to_representation[parts[0]] = torch.tensor(list(map(float, new_parts)), dtype=torch.float32)
            self.article_to_representation = {}
            with open(self.article_to_representation_file, 'r+') as f:
                for line in tqdm(f):
                    parts = line.split('\t')
                    new_parts = parts[1].replace('[', '').replace(']', '').replace('\n', '')
                    new_parts = " ".join(new_parts.split())
                    new_parts = re.sub(' +', ' ', new_parts).split(' ')
                    self.article_to_representation[int(parts[0])] = torch.tensor(list(map(float, new_parts)), dtype=torch.float32)
        self.articles_per_directory = self.load_dict(self.articles_per_directory_path)
        self.article_user_mapping = self.load_dict(self.article_user_mapping_path)
        

        # load the sources we will use
        self.corpus_df = pd.read_csv(self.dataset_corpus, sep='\t')
        sources_in_corpus = []
        self.sources_in_corpus_normalized = []
        for index, row in self.corpus_df.iterrows():
            sources_in_corpus.append(row['source_url'])
            self.sources_in_corpus_normalized.append(row['source_url_normalized'])
            self.labels[row['source_url_normalized']] = (row['fact'], row['bias'])
        self.sources_in_corpus_domains = {}
        for x in sources_in_corpus:
            self.sources_in_corpus_domains[str(tldextract.extract(x).registered_domain)] = x

        # create dictionaries to store the metadata
        self.source_followers_dict = self.load_dict(self.followers_dict_path)
        self.source_username_mapping_dict = self.load_dict(self.source_username_mapping_dict_path)
        self.users_that_follow_each_other = self.load_dict(self.users_that_follow_each_other_path)

        self.article_name_to_id = self.load_dict(self.article_name_to_id_file)

        self.sources_mapping_dict = None
        self.users_mapping_dict = None

        with open(os.path.join(self.path_where_data_is, "data/acl2020" , f"splits.json")) as the_file:
            self.data_splits = json.load(the_file)
            self.curr_data_split = self.data_splits['1']

            self.dev_set_sources = random.sample(self.curr_data_split['train'], int(len(self.curr_data_split['train']) * 0.3))
            np.save(self._save_dir + '_dev_set_sources.npy', self.dev_set_sources)

        self._g = self._build_dgl_graph_style_m1_m2()

    def _build_dgl_graph_style_m1_m2(self):
        ''' Builds a graph where nodes are sources or users or articles.

        Sources have users that are their followers

        Sources are represented as the average of all their RoBERTa SBERT document embeddings

        Users are represented with a feature vector containing their profile embedded using SBERT and some other features

        Articles are represented by their RoBERTa embeddings. Articles are published by sources and interacted with by users.
        '''



        # define all the nodes, edges, and edge types for the graph            
        data_dict = {('source', 'has_follower', 'user'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('user', 'follows', 'source'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('source', 'has_article', 'article'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('article', 'is_published_by', 'source'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('article', 'has_talker', 'user'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('user', 'talks_about', 'article'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('user', 'user_follows', 'user'): (torch.tensor([0, 0]), torch.tensor([0, 0])) }

        # add the edges for the inference operators
        data_dict_with_extra_options = {('article', 'talks_similar_article', 'article'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('article', 'does_not_talk_similar_article', 'article'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('source', 'talks_similar_source', 'source'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('user', 'connects_with_contr', 'user'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('user', 'is_influenced_by_contr', 'user'): (torch.tensor([0, 0]), torch.tensor([0, 0])), ('article', 'talks_similar_article_neg', 'article'): (torch.tensor([0, 0]), torch.tensor([0, 0]))
                    }
        data_dict.update(data_dict_with_extra_options)

        # use DGL to build the graph and set up the initial embeddings. We will add nodes and update the embeddings later
        dgl_graph = dgl.heterograph(data_dict)
        # assign zeros for the features to start with
        source_embedding_size = 778
        user_embedding_size = 773
        article_embedding_size = 768
        self.article_embedding_size = article_embedding_size
        dgl_graph.nodes['source'].data['source_embedding'] = torch.zeros(1, source_embedding_size)
        dgl_graph.nodes['source'].data['source_name'] = torch.zeros(1, 1)
        dgl_graph.nodes['source'].data['source_label'] = torch.zeros(1, 1)
        dgl_graph.nodes['user'].data['user_embedding'] = torch.zeros(1, user_embedding_size)
        dgl_graph.nodes['article'].data['article_embedding'] = torch.zeros(1, article_embedding_size)
        dgl_graph.nodes['article'].data['article_label'] = torch.zeros(1, 1)

        # create placeholder dictionaries to keep track of the string of each node in the graph
        sources_mapping_dict = {}
        users_mapping_dict = {}
        articles_mapping_dict = {}
        
        def add_directory_to_graph(curr_directory):
            '''Helper function to add a source to the graph. It will add the source, it's articles, all users that interact with (or it's articles), and the respective embeddings/edges'''

            print("Adding the source " + str(curr_directory))

            # make the graph variables global
            nonlocal dgl_graph


            # get the label for the source
            current_source_label = self.labels[curr_directory][0]
            if current_source_label == 'low':
                current_source_label = 1
            elif current_source_label == 'mixed':
                current_source_label = 2
            elif current_source_label == 'high':
                current_source_label = 3
            else:
                print("Label is unknown")
                print(current_source_label)
            if current_source_label is None:
                return

            # first, let's add the sources sources that have followers
            # add the source node first
            sid = dgl_graph.number_of_nodes(ntype='source')
            # add the root node
            dgl_graph.add_nodes(1, ntype='source')
            # add it to the mapping dictionary
            if curr_directory + self.source_name_identifier not in sources_mapping_dict:
                sources_mapping_dict[curr_directory + self.source_name_identifier] = sid
            else:
                sid = sources_mapping_dict[curr_directory + self.source_name_identifier]
            
            # add the document representation for this source
            current_source_tensor = self.source_name_to_representation[curr_directory]
            # source embedding
            dgl_graph.nodes['source'].data['source_embedding'][sid] = current_source_tensor
            dgl_graph.nodes['source'].data['source_name'][sid] = torch.FloatTensor([sid])

                
            if current_source_label is not None:
                dgl_graph.nodes['source'].data['source_label'][sid] = torch.ones(1, 1) * current_source_label
            else:
                print("What the heck there is no label!")

                
            # check to see if the source has any followers
            if curr_directory in self.source_username_mapping_dict:
                # get their twitter handle
                given_source_username = self.source_username_mapping_dict[curr_directory]
                print("Getting followers for source " + str(given_source_username))
                if given_source_username not in self.source_followers_dict:
                    print("The source username doesnt have followers")
                # get the users that follow this source
                users_that_follow_this_source = self.source_followers_dict[given_source_username]

                print("There are so many followers " + str(len(users_that_follow_this_source)))
                source_followers_added = 0
                # go through all the users that follow
                for given_follower_username in tqdm(users_that_follow_this_source):

                    # flag to see if we were able to add a user this time
                    follower_added = False
                    # get the user and add it
                    given_follower_username = str(given_follower_username)

                    if (given_follower_username + self.user_profile_identifier) not in users_mapping_dict:

                        # get the user representation
                        if given_follower_username in self.user_id_to_representation:
                            follower_added = True
                            given_follower_embedding = self.user_id_to_representation[given_follower_username]
                            # add the node for the follower
                            fid = dgl_graph.number_of_nodes(ntype='user')
                            dgl_graph.add_nodes(1, ntype='user')
                            # add the user representation
                            dgl_graph.nodes['user'].data['user_embedding'][fid] = given_follower_embedding
                            users_mapping_dict[given_follower_username + self.user_profile_identifier] = fid
                        else:
                            print("Couldn't find the representation for the user " + str(given_follower_username))
                            continue

                    else:
                        follower_added = True
                        fid = users_mapping_dict[given_follower_username + self.user_profile_identifier]

                    if follower_added:
                        source_followers_added += 1
                        # if we were able to add a follower let's link it to the source
                        # the source would have been added earlier
                        # add an edge from the source to the follower
                        dgl_graph.add_edges(torch.tensor([sid]), torch.tensor([fid]), etype='has_follower')
                        dgl_graph.add_edges(torch.tensor([fid]), torch.tensor([sid]), etype='follows')

                print("Got these many followers " + str(source_followers_added))
                    
            # get the data for the articles
            articles_added = 0
            articles_added_with_users = 0

            print("Adding articles of which there are these many " + str(len(self.articles_per_directory[curr_directory])))

            articles_added = 0
            articles_added_with_users = 0
            for article_name in self.articles_per_directory[curr_directory]:

                if article_name + self.article_name_identifier not in articles_mapping_dict:
                    # we haven't added this article yet and it is ok to add it
                    # add the article node
                    aid = dgl_graph.number_of_nodes(ntype='article')
                    dgl_graph.add_nodes(1, ntype='article')
                    articles_mapping_dict[article_name + self.article_name_identifier] = aid
                    # add the article representation
                    current_article_tensor = self.article_to_representation[int(self.article_name_to_id[article_name])]
                    dgl_graph.nodes['article'].data['article_embedding'][aid] = current_article_tensor  
                    dgl_graph.nodes['article'].data['article_label'][aid] = torch.ones(1, 1) * current_source_label 
                else:
                    # article already exists, load it
                    aid = articles_mapping_dict[article_name + self.article_name_identifier]

                # link the source to the article
                # make the link only if the article is not already linked to the source. so check for that first
                found_edge = False
                try:
                    edge_there = dgl_graph.edge_ids(torch.tensor([sid]), torch.tensor([aid]), etype='has_article')
                    found_edge = True
                except Exception as e:
                    found_edge = False
                if not found_edge:
                    dgl_graph.add_edge(torch.tensor([sid]), torch.tensor([aid]), etype='has_article')
                    dgl_graph.add_edge(torch.tensor([aid]), torch.tensor([sid]), etype='is_published_by')
            
                # increment the counter that we added the article
                articles_added += 1

                article_name_id = self.article_name_to_id[article_name]

                # see if the article has any users propagating it. it could have a multitude of names in the dictionary, so check for that
                # add the users that talk about this articles to the graph
                if article_name_id in self.article_user_mapping:
                    # flag to see if we added a user that talks about this article or not
                    added_tweeter_article = False
                    for given_tweeter_username in self.article_user_mapping[article_name_id]:
                        # see if this user already exists in the graph
                        given_tweeter_username = str(given_tweeter_username)
                        if given_tweeter_username + self.user_profile_identifier not in users_mapping_dict: 

                            # if we have downloaded the profile for this username
                            if given_tweeter_username in self.user_id_to_representation:
                                given_tweeter_embedding = self.user_id_to_representation[given_tweeter_username]

                                # add the node for the tweeter
                                atid = dgl_graph.number_of_nodes(ntype='user')
                                dgl_graph.add_nodes(1, ntype='user')
                                # add the user representation
                                dgl_graph.nodes['user'].data['user_embedding'][atid] = given_tweeter_embedding
                                users_mapping_dict[given_tweeter_username + self.user_profile_identifier] = atid

                                # add an edge from the article to the user if it doesn't already exist
                                found_edge = False
                                try:
                                    edge_there = dgl_graph.edge_ids(torch.tensor([aid]), torch.tensor([atid]), etype='has_talker')
                                    found_edge = True
                                except Exception as e:
                                    found_edge = False
                                if not found_edge:
                                    dgl_graph.add_edges(torch.tensor([aid]), torch.tensor([atid]), etype='has_talker')
                                    dgl_graph.add_edges(torch.tensor([atid]), torch.tensor([aid]), etype='talks_about')                       
                        else:
                            # article exists
                            atid = users_mapping_dict[given_tweeter_username + self.user_profile_identifier]
                            # add edge if not already exists
                            found_edge = False
                            try:
                                edge_there = dgl_graph.edge_ids(torch.tensor([aid]), torch.tensor([atid]), etype='has_talker')
                                found_edge = True
                            except Exception as e:
                                found_edge = False
                            if not found_edge:
                                # add an edge from the article to the tweeter
                                dgl_graph.add_edges(torch.tensor([aid]), torch.tensor([atid]), etype='has_talker')
                                dgl_graph.add_edges(torch.tensor([atid]), torch.tensor([aid]), etype='talks_about')
                        added_tweeter_article = True
                    if added_tweeter_article:
                        articles_added_with_users += 1 
                else:
                    print("not able to add an article " + str(article_name))

            print("We added these many articles " + str(articles_added) + " and these many articles with users " + str(articles_added_with_users))
        


        def connect_users_that_follow_each_other(given_users_dict):
            '''Connect pairs of users that follow each other based on the given dictionary'''

            print("In follow each other function")
            print(dgl_graph.num_edges(etype='user_follows'))
            counter_now = 0

            # these lists will store the edges we want to connect to make it quicker
            ids_to_add_to_graph = []
            second_ids_to_add_to_graph = []
            backward_ids_to_add = []
            second_backward_ids_to_add = []
            # connect users that follow each other only if they already exist in the graph
            for curr_random_user, all_random_users_followed in tqdm(given_users_dict.items()):
                curr_random_user = str(curr_random_user)

                if counter_now >= 5000:
                    # every 5000, add the edges -> this is for efficiency
                    counter_now = 0
                    dgl_graph.add_edges(torch.tensor(ids_to_add_to_graph), torch.tensor(second_ids_to_add_to_graph), etype='user_follows')
                    dgl_graph.add_edges(torch.tensor(backward_ids_to_add), torch.tensor(second_backward_ids_to_add), etype='user_follows')
                    ids_to_add_to_graph = []
                    second_ids_to_add_to_graph = []
                    backward_ids_to_add = []
                    second_backward_ids_to_add = []

                for given_user_followed in all_random_users_followed:
                    given_user_followed = str(given_user_followed)

                    if counter_now >= 5000:
                        # every 5000, add the edges -> this is for efficiency
                        counter_now = 0
                        dgl_graph.add_edges(torch.tensor(ids_to_add_to_graph), torch.tensor(second_ids_to_add_to_graph), etype='user_follows')
                        dgl_graph.add_edges(torch.tensor(backward_ids_to_add), torch.tensor(second_backward_ids_to_add), etype='user_follows')
                        ids_to_add_to_graph = []
                        second_ids_to_add_to_graph = []
                        backward_ids_to_add = []
                        second_backward_ids_to_add = []

                    try:
                        f1_id = users_mapping_dict[curr_random_user + self.user_profile_identifier]
                        f2_id = users_mapping_dict[given_user_followed + self.user_profile_identifier]

                        ids_to_add_to_graph.append(f1_id)
                        second_ids_to_add_to_graph.append(f2_id)
                        backward_ids_to_add.append(f2_id)
                        second_backward_ids_to_add.append(f1_id)
                        counter_now += 1

                    except Exception as e:
                        # print(e)
                        # exit(1)
                        continue

            if len(ids_to_add_to_graph) > 0:
                dgl_graph.add_edges(torch.tensor(ids_to_add_to_graph), torch.tensor(second_ids_to_add_to_graph), etype='user_follows')
                dgl_graph.add_edges(torch.tensor(backward_ids_to_add), torch.tensor(second_backward_ids_to_add), etype='user_follows')

            print(dgl_graph.num_edges(etype='user_follows'))

                            
        # add all the unknown sources to the graph
        # now, add the sources that we have articles for but no entities/google news
        if not self.building_original_graph_first:
            print("Loading the graph from before to add the sources from the directory")
            self.load()
            sources_mapping_dict = self.sources_mapping_dict
            users_mapping_dict = self.users_mapping_dict
            articles_mapping_dict = self.articles_mapping_dict
            dgl_graph = self._g[0]

        for index, row in tqdm(self.corpus_df.iterrows(), total=len(self.corpus_df)):
            curr_directory = row['source_url_normalized']

            print(curr_directory)
            sys.stdout.flush()

            if (curr_directory + self.source_name_identifier) in sources_mapping_dict:
                # we added this source before or at least tried to, don't add again
                print("Directory exists, don't add again")
                continue

            print("Adding directory " + str(curr_directory))
            sys.stdout.flush()
            add_directory_to_graph(curr_directory)

            # if curr_directory == 'crooked.com':
            #     break

        print("Saving the graph")
        self._g = dgl_graph
        self.sources_mapping_dict = sources_mapping_dict
        self.users_mapping_dict = users_mapping_dict
        self.articles_mapping_dict = articles_mapping_dict
        self.save()
        print("Graph saved")

        if self.building_original_graph_first:
            exit(1)

        print("Now we are going to do the users that follow each other")
        connect_users_that_follow_each_other(self.users_that_follow_each_other)

        print("Saving the graph")
        self._g = dgl_graph
        self.sources_mapping_dict = sources_mapping_dict
        self.users_mapping_dict = users_mapping_dict
        self.articles_mapping_dict = articles_mapping_dict
        self.save()
        print("Graph saved")
        
        # we need to remove the 0 node since it was a placeholder when we created the graph
        print("Removing the node")
        dgl_graph.remove_nodes(0, ntype='source')
        dgl_graph.remove_nodes(0, ntype='user')
        dgl_graph.remove_nodes(0, ntype='article')
        self.sources_mapping_dict = sources_mapping_dict
        self.users_mapping_dict = users_mapping_dict
        self.articles_mapping_dict = articles_mapping_dict
        self._g = dgl_graph
        self.save()

        return dgl_graph

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    def save(self, new_save_dir=None):
        # save graphs and labels
        if new_save_dir is not None:
            directory_to_save = new_save_dir
        else:
            directory_to_save = self._save_dir
        graph_path = os.path.join(directory_to_save, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self._g)
        # save other information in python dict
        info_path = os.path.join(directory_to_save, self.mode + '_info.pkl')

        with open(directory_to_save + 'sources_mapping_dict.pkl', 'wb') as outfile:
            pickle.dump(self.sources_mapping_dict, outfile)
        with open(directory_to_save + 'users_mapping_dict.pkl', 'wb') as outfile:
            pickle.dump(self.users_mapping_dict, outfile)
        with open(directory_to_save + 'articles_mapping_dict.pkl', 'wb') as outfile:
            pickle.dump(self.articles_mapping_dict, outfile)

    def load(self, new_save_dir=None):
        # load processed data from directory `self.save_path`
        if new_save_dir is not None:
            directory_to_save = new_save_dir
        else:
            directory_to_save = self._save_dir
        graph_path = os.path.join(directory_to_save, self.mode + '_dgl_graph.bin')
        self._g, label_dict = load_graphs(graph_path)
        info_path = os.path.join(directory_to_save, self.mode + '_info.pkl')
        with open(directory_to_save + 'sources_mapping_dict.pkl', 'rb') as infile:
            self.sources_mapping_dict = pickle.load(infile)
        with open(directory_to_save + 'users_mapping_dict.pkl', 'rb') as infile:
            self.users_mapping_dict = pickle.load(infile)
        with open(directory_to_save + 'articles_mapping_dict.pkl', 'rb') as infile:
            self.articles_mapping_dict = pickle.load(infile)
        

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self._save_dir, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self._save_dir, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)











