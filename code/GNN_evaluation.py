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
import sys 
torch.manual_seed(0)


from sklearn.metrics import roc_auc_score
import numpy as np
import sklearn.metrics as skm
from collections import Counter

import warnings
warnings.filterwarnings('once')

import torch
torch.manual_seed(0)
np.random.seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from training_helper_functions import get_features_given_graph, get_features_given_blocks


def do_evaluation(model, g, overall_graph, args, test_dataloader=None, loss_fcn=None):

    if test_dataloader is None:
        return 0.0, 0.0, 0.0

    g = overall_graph._g[0]

    node_features_for_inference = get_features_given_graph(g, args)

    model.eval()
    total_acc = 0
    count = 0
    total_loss = 0

    wrong_sources = {}

    all_labels = []
    all_predictions = []
    all_predictions_auc = []

    low_confidence_sources_probs = {}

    with torch.no_grad():
        for input_nodes, seeds, blocks in test_dataloader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            output_labels = blocks[-1].dstdata['source_label']['source']
                
            if len(output_labels) == 0:
                continue

            sys.stdout.flush()

            output_labels = (output_labels - 1).long()
            output_labels = torch.squeeze(output_labels)

            node_features = get_features_given_blocks(g, args, blocks)
            output_predictions = model(blocks, node_features, g=None, neg_g=None)['source']

            # this is just to handle odd batch sizes
            try:
                list(output_labels.long().cpu().numpy())
            except Exception as e:
                output_labels = torch.unsqueeze(output_labels, dim=0)

            if loss_fcn is not None:
                # compute loss if provided
                try:
                    loss = loss_fcn(output_predictions, output_labels)
                except Exception as e:
                    print("Error computing loss")
                    print(output_predictions)
                    print(output_labels)
                    exit(1)

            # compute accuracy
            acc = (torch.sum(output_predictions.argmax(dim=1) == output_labels.long()).item())
            source_ids = blocks[-1].dstnodes['source'].data['_ID']

            # this handles a situation where there is only one element in output_labels due to the batch being only one element long
            try:
                output_labels_to_use = list(output_labels.long().cpu().numpy())
            except Exception as e:
                output_labels_to_use = [output_labels.long().cpu().numpy()]

            all_labels.extend(output_labels_to_use)
            all_predictions.extend(list(output_predictions.argmax(dim=1).cpu().numpy()))

            total_acc += acc

            if loss_fcn is not None:
                total_loss += loss.item() * len(output_predictions)
            count += len(output_predictions)

        f1_score = skm.f1_score(all_labels, all_predictions, average='macro')

    
        if loss_fcn is not None:
            return total_acc / count, total_loss / count, f1_score
        else:
            return total_acc / count, 100, f1_score

