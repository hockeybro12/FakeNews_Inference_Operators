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
import faiss   
import sys 
torch.manual_seed(0)
import timeit
from torch.nn.parallel import DistributedDataParallel

from GNN_model_architecture import FakeNewsModel

import graph_helper_functions
# To make the GNN we use DGL
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args

from GNN_evaluation import do_evaluation


import sklearn
import json
import os
import torch.multiprocessing as mp
from _thread import start_new_thread
from functools import wraps
import traceback
import pandas as pd
import warnings
warnings.filterwarnings('once')
# from yellowbrick.style import set_palette
from torch.nn.utils import clip_grad_norm_
 
import torch
torch.manual_seed(0)
sk_learn_seed = 5
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dgl.random.seed(0)

from training_helper_functions import get_train_mask_nids, get_features_given_blocks, get_features_given_graph
from inference_operator_helper_functions import *


def running_code(proc_id, n_gpus, args, devices, overall_graph):
    '''This is the function that will run the code. It is possible to run the code on multiple devices but only one GPU is necessary to train this.'''
    device = devices[proc_id]
    print("Device " + str(device))
    sys.stdout.flush()

    # the port we are currently running on -> used to handle distributed training
    if args.curr_port is not None:
        curr_port_tru = str(args.curr_port)
    else:
        curr_port_tru = '25295'
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port=str(curr_port_tru))
    world_size = n_gpus
    torch.distributed.init_process_group(backend="nccl", init_method=dist_init_method, world_size=world_size, rank=proc_id)
    torch.cuda.set_device(device)


    # get the graph
    g = overall_graph._g[0]
    print("Loaded the graph")
    sys.stdout.flush()

    # get the labels
    labels_dict = {}
    corpus_df = pd.read_csv(overall_graph.dataset_corpus, sep='\t')
    sources_in_corpus = []
    for index, row in corpus_df.iterrows():
        sources_in_corpus.append(row['source_url'])
        labels_dict[row['source_url_normalized']] = (row['fact'], row['bias'])

    # load the data splits
    with open(os.path.join(args.path_where_data_is, "data/acl2020" , f"splits.json")) as the_file:
        data_splits = json.load(the_file)
        n_nodes_sources = g.number_of_nodes(ntype='source')
   
    # print graph statistics
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    n_nodes_users = g.number_of_nodes(ntype='user')
    n_nodes_articles = g.number_of_nodes(ntype='article')
    n_edges_articles_talkers = g.number_of_edges(etype='has_talker')
    print("""----Data statistics------'
      #Edges %d
      #Nodes %d
      #Source Nodes %d
      #User Nodes %d
      #Article Nodes %d
      #Article Talker Edges %d""" %
          (n_edges, n_nodes, n_nodes_sources, n_nodes_users, n_nodes_articles, n_edges_articles_talkers))
    sys.stdout.flush()

    # out features is 3 for 3-class classification and set the embedding size
    out_features = 3
    user_embedding_size = 773
    article_embedding_size = 768

        
    # set up the model
    cuda = True
    torch.cuda.set_device(device)
    print("use cuda:", args.gpu)
    print("Need to push graph to GPU")
    if cuda:
        g = g.int()#.to(args.gpu)
    # set up the model
    model = FakeNewsModel(in_features={'source':778, 'user':user_embedding_size, 'article': article_embedding_size}, hidden_features=args.hidden_features, out_features=out_features, canonical_etypes=g.canonical_etypes, num_workers=args.num_workers, n_layers=args.n_layers, conv_type='gcn')
    # set up the model for distributed training
    if cuda:
        # model.cuda()
        model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    if args.load_model:
        print("Loading the model")
        model.module.load_state_dict(torch.load(args.path_to_save_model))
        model.eval()
    model.eval()

    # set up the  optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decayRate = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-4)


    # initialize graph
    dur = []
    best_acc = 0.0
        
    # which inference operators are we going to run. For now, we support users-users (2) and users-articles (8)
    em_options_to_run = args.em_options_to_run_list.split(':')
    em_options_to_run = [int(x) for x in em_options_to_run]
    print("These are the EM options we are going to run")
    print(em_options_to_run)

    if args.run_multiple_data_splits:
        all_the_data_splits_we_want_to_run_now = data_splits.keys()
    else:
        all_the_data_splits_we_want_to_run_now = args.data_splits_key_to_run_when_doing_one.split(':')
    print("Running these many data splits " + str(all_the_data_splits_we_want_to_run_now))

    best_test_acc_average = []
    best_test_acc_at_dev_average = []
    path_to_save_model_old = args.path_to_save_model

    iterations_before_em = 500
    
    for curr_data_split_key in all_the_data_splits_we_want_to_run_now:
        # go through each data split
        print("Data split is " + str(curr_data_split_key))
        # keep track of what edges inference operators added.
        em_edges_added = {}
        args.path_to_save_model = path_to_save_model_old + curr_data_split_key
        print(args.path_to_save_model)
        print("Running " + str(curr_data_split_key))

        # set up the model again for this split
        model = FakeNewsModel(in_features={'source':778, 'user':user_embedding_size, 'article': article_embedding_size}, hidden_features=args.hidden_features, out_features=out_features, canonical_etypes=g.canonical_etypes, num_workers=args.num_workers, n_layers=args.n_layers, conv_type='gcn')
        model = model.to(torch.device('cuda'))
        model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        decayRate = 0.96
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-4)
        # store the best performance
        best_acc = 0.0
        best_dev_acc = 0.0
        bes_acc_at_dev = 0.0
        best_train_acc = 0.0
        print("Working with data split " + str(curr_data_split_key))
        # we are training each data split
        curr_data_split = data_splits[curr_data_split_key]

        training_set_to_use = curr_data_split['train']
        # create a dev set
        if not os.path.isfile(args.path_where_graph_is + '/dev_set_' + curr_data_split_key + '.npy'):
            # if we don't have the dev set yet, create it and save it to load from later
            training_set_to_use, dev_set_to_use = sklearn.model_selection.train_test_split(curr_data_split['train'], test_size=args.percentage_of_dev_to_use)
            np.save(args.path_where_graph_is + '/dev_set_' + curr_data_split_key + '.npy', dev_set_to_use)
            np.save(args.path_where_graph_is + '/train_set_' + curr_data_split_key + '.npy', training_set_to_use)
        else:
            dev_set_to_use = np.load(args.path_where_graph_is + '/dev_set_' + curr_data_split_key + '.npy')
            training_set_to_use = np.load(args.path_where_graph_is + '/train_set_' + curr_data_split_key + '.npy')
        print("Length of the new train set: " + str(len(training_set_to_use)))
        print("Length of the new dev set: " + str(len(dev_set_to_use)))
        sys.stdout.flush()  

        graph_style = 'm2'                  

        # set up the masks for what to train on 
        train_mask, dev_mask, test_mask, train_nids, dev_nids, test_nids = get_train_mask_nids(args, overall_graph, training_set_to_use, curr_data_split, dev_set_to_use, graph_style=graph_style, use_dev_set=True, curr_data_split_key=curr_data_split_key)
        train_mask_tensor = torch.from_numpy(train_mask)
        train_idx = torch.nonzero(train_mask_tensor).squeeze()
        test_mask_tensor = torch.from_numpy(test_mask)
        test_idx = torch.nonzero(test_mask_tensor).squeeze()
        dev_mask_tensor = torch.from_numpy(dev_mask)
        dev_idx = torch.nonzero(dev_mask_tensor).squeeze()
        train_labels_idx = None

        curr_g = overall_graph._g[0]

        # set up the samplers and dataloaders for DGL
        negative_sampler_to_use = dgl.dataloading.negative_sampler.Uniform(5)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
        # for MULTI-GPU training, split the NID's
        # then set up the dataaloaoders for node classification
        train_nid = torch.split(train_idx, math.ceil(len(train_idx) / n_gpus))[proc_id]
        test_nid = torch.split(test_idx, math.ceil(len(test_idx) / n_gpus))[proc_id]
        dataloader_nc = dgl.dataloading.NodeDataLoader(curr_g, {'source': train_nid}, sampler, batch_size=64, shuffle=True, drop_last=False, num_workers=args.num_workers)
        test_sampler = sampler
        dataloader_test = dgl.dataloading.NodeDataLoader(curr_g, {'source': test_idx}, test_sampler, batch_size=64, shuffle=True, drop_last=False, num_workers=args.num_workers)
        dev_sampler = sampler
        dataloader_dev = dgl.dataloading.NodeDataLoader(curr_g, {'source': dev_idx}, dev_sampler, batch_size=64, shuffle=True, drop_last=False, num_workers=args.num_workers)

        
        any_em_done = False
        best_train_acc = 0.0

        graph_number_entity_advice_purity = {}

        first_iteration = True
        em_iterations_done_so_far = 0
        # determine the help article mapping dictionaries (article -> ID mapping, article -> label, article -> source)
        # if we have computed it before already it will just return
        article_id_labels_dict_all, article_source_mapping_all, article_id_name_mapping_all = compute_articles_dict(args, overall_graph, graph_style, training_set_to_use, curr_data_split_key, labels_dict, add_everything=True, computing_based_on_predicted_labels=False, labels_dict_predicted=None)

        for em_epoch in range(args.n_epochs):
            start_time = time.time()
            if args.just_train_nc:
                iterations_before_em = 500

            # save the node classification model before any inference opertators are run
            if proc_id == 0 and not args.use_loaded_model_for_secondary_em:
                torch.save(model.module.state_dict(), args.path_to_save_model + str(curr_data_split_key) + '_em_option_after_lp_nc' + str(em_options_to_run))

        
            print("Training the node classifier")
            sys.stdout.flush()

            loss_fcn = nn.CrossEntropyLoss()
            loss_fcn = loss_fcn.to(torch.device('cuda'))

            sys.stdout.flush()

            model.train()
            optimizer.zero_grad()
            sys.stdout.flush()
            
            for iteration, (input_nodes, output_nodes, blocks) in enumerate(dataloader_nc):

                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata     # returns a dict
                if graph_style == 'fang':
                    output_labels = blocks[-1].dstdata['article_label']['article']    # returns a dict
                else:
                    output_labels = blocks[-1].dstdata['source_label']['source']    # returns a dict
                output_labels = (output_labels - 1).long()
                output_labels = torch.squeeze(output_labels)
                sys.stdout.flush()

                node_features = get_features_given_blocks(curr_g, args, blocks)
                output_predictions = model(blocks, node_features, g=None, neg_g=None)['source']


                loss = loss_fcn(output_predictions, output_labels)
                loss.backward()
                clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step(loss)
                sys.stdout.flush()

                acc = (torch.sum(output_predictions.argmax(dim=1) == output_labels.long()).item()) / len(output_predictions)

                if iteration % 10 == 0 and proc_id == 0:
                    print("Epoch {:05d} | Loss {:.4f} | Acc {:.4f}".format(em_epoch, loss.item(), acc))

            torch.cuda.empty_cache()

            # compute the accuracy on process 0
            if proc_id == 0:
                model.eval()

                train_acc, train_loss, train_f1 = do_evaluation(model, curr_g, overall_graph, args, test_dataloader=dataloader_nc, loss_fcn=loss_fcn)
                print("Epoch " + str(em_epoch) + " Train Set Accuracy classification: " + str(train_acc) + " Loss: " + str(train_loss) + " test f1 " + str(train_f1))
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                print("Best train accuracy " + str(best_train_acc))

                test_acc, test_loss, test_f1 = do_evaluation(model, curr_g, overall_graph, args, test_dataloader=dataloader_test, loss_fcn=loss_fcn)
                print("Epoch " + str(em_epoch) + " Test Accuracy classification: " + str(test_acc) + " Loss: " + str(test_loss) + " test f1 " + str(test_f1))
                dev_acc, dev_loss, dev_f1 = do_evaluation(model, curr_g, overall_graph, args, test_dataloader=dataloader_dev, loss_fcn=loss_fcn)
                print("Epoch " + str(em_epoch) + " Dev Set Accuracy classification: " + str(dev_acc) + " Loss: " + str(dev_loss) + " test f1 " + str(dev_f1))


                save_the_model = False
                if not args.save_only_at_good_dev:
                    if test_acc > best_acc or (args.not_run_test and dev_acc > best_dev_acc) or args.save_always:
                        save_the_model = True
                else:
                    if dev_acc > best_dev_acc:
                        print("Dev acc higher sso we should save")
                        save_the_model = True
                if args.save_always:
                    save_the_model = True
                if save_the_model:
                    print("Saving the model at save_the_model")
                    best_acc = test_acc
                    torch.save(model.module.state_dict(), args.path_to_save_model + str(curr_data_split_key) + '_em_iter1_option_nc_' + str(em_options_to_run))

                    

                print("Best accuracy classification: " + str(best_acc))
                if dev_acc > best_dev_acc:
                    bes_acc_at_dev = test_acc
                    best_dev_acc = dev_acc
                print("Best accuracy classification at the dev set: " + str(bes_acc_at_dev))
                print("Best dev accuracy classification: " + str(best_dev_acc))
                torch.save(model.module.state_dict(), args.path_to_save_model + str(curr_data_split_key) + 'best_at_dev' + str(em_options_to_run))
                sys.stdout.flush()

            torch.cuda.empty_cache()

            # update the learning rate
            print('Learning rate at this epoch is: ', optimizer.param_groups[0]['lr'], '\n')
            sys.stdout.flush()

            iterations_before_em = iterations_before_em - 1
            print(str(iterations_before_em) + " before we run another EM step")
            print('Learning rate at this epoch is: ', optimizer.param_groups[0]['lr'], '\n')
            sys.stdout.flush()
               
            end_time = time.time()
            print("total time taken this loop: " + str(end_time - start_time))

        print("Done training this data split " + str(curr_data_split_key))
        print("Best Test Accuracy {:.4f}".format(best_acc))
        best_test_acc_average.append(best_acc)
        best_test_acc_at_dev_average.append(best_test_acc_at_dev_average)

    print("Done training")
    print("Best average test accuracy " + str(np.mean(best_test_acc_average)))
    print("Best dev set average test accuracy " + str(np.mean(best_test_acc_at_dev_average)))

def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--batch-size", type=int, default=7000,
                        help="Batch size")
    parser.add_argument("--hidden_features", type=int, default=512,
                        help="Number of hidden features in graph model")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Worker count")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")

    parser.add_argument('--path_to_save_model', nargs='?', type=str, default='', help='Path to save the trained model.')
    parser.add_argument('--load_model', action='store_true', help="True if you want to load the model")
    parser.add_argument('--use_loaded_model_for_secondary_em', action='store_true', help="True if you want to load the model for inference operators second iteration")
    

    parser.add_argument("--data_splits_key_to_run_when_doing_one", type=str, default='0',
                        help="If we are only running one data split meaning only_run_one_data_split is set, then which data split are we running")
    parser.add_argument('--only_run_one_data_split', action='store_true', help="True if you want to only run one data split.")
    parser.add_argument('--run_multiple_data_splits', action='store_true', help="True if you want to reset multiple data splits at once in the EM stage.")
    parser.add_argument("--curr_port", type=str, default=None, help='The port to run on. Useful for multi-GPU training')
    
    
    parser.add_argument("--save_only_at_good_dev", action='store_true', help="True if you only want to save at good performance on the dev set.")
    parser.add_argument('--save_always', action='store_true', help="True you always want to save the model.")

    parser.add_argument("--just_train_nc", action='store_true', help="True you just want to train node classification. Otherwise we may end up doing inference operators (in the future).")
    parser.add_argument("--path_to_load_model_from", type=str, default=None,
                        help="if you want to load the model from a specific path, put it here")


    parser.add_argument('--path_where_graph_is', type=str, default='PaperGraph_test_release/')
    parser.add_argument('--users_that_follow_each_other_path', type=str, default='dataset_release/users_that_follow_each_other.npy')
    parser.add_argument('--followers_dict_path', type=str, default='dataset_release/source_followers_dict.npy')
    parser.add_argument('--source_username_mapping_dict_path', type=str, default='dataset_release/domain_twitter_triplets_dict.npy')

    parser.add_argument('--percentage_of_dev_to_use', type=float, default=0.20, help='How much of the training set we should train on')
    parser.add_argument('--path_where_data_is', type=str, default='News-Media-Reliability')
    parser.add_argument('--dataset_corpus', type=str, default='News-Media-Reliability/data/acl2020/corpus.tsv')

    ## these arguments are not implemented yet but are there for the future once we add inference operators to this code-base
    parser.add_argument("--em_options_to_run_list", type=str, default='2', help="after how many iterations should we do the inference operator process")
    
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    print(args)

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # load and preprocess dataset
    overall_graph = graph_helper_functions.FakeNewsDataset(save_path=args.path_where_graph_is, users_that_follow_each_other_path=args.users_that_follow_each_other_path, dataset_corpus=args.dataset_corpus, followers_dict_path=args.followers_dict_path, source_username_mapping_dict_path=args.source_username_mapping_dict_path, building_original_graph_first=False)
    overall_graph.load()

    # overall_graph._g[0].remove_nodes(0, ntype='source')


    procs = []
    if n_gpus == 1:
        ctx = mp.get_context('spawn')
        p = ctx.Process(target=running_code, args=(0, n_gpus, args, devices, overall_graph))
        p.start()
        procs.append(p)
        for p in procs:
            p.join()
    else:
        print("We have multiple GPUs")
        sys.stdout.flush()
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=running_code, args=(proc_id, n_gpus, args, devices, overall_graph))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
