This folder contains the code to build the graph, train it on Node Classification, and run Inference Operators.


## Building The Graph
First, you must build the graph, which is done in multiple steps. 

### Initial Graph
(1) Create a directory where you want the graph to be saved: `mkdir PaperGraph_release/`
(2) Make sure you have the data in this directory. You should have the Folders `dataset_release`, and `News-Media-Reliability`. `News-Media-Reliability` is the repository from (https://github.com/ramybaly/News-Media-Reliability)[href]. `dataset_release` comes with this folder. You need to read the README there in order to download the embeddings and other files from Google Drive that are too large. 
(3) Build the graph. If you are using the embeddings we released, you don't have to do anything except for have `dataset_release` set up correctly. If not, you need to update `source_name_to_representation_file`, `article_to_representation_file`, and `user_id_to_representation_file` appropriately. You can set it up to have the same format as what is in `dataset_release`, with the key being the graph node ID and the value being the representation. Since everything else for creating the graph is based on Graph Node IDs (like what sources have what articles and what articles have users propagating them), you will have to update all of that as well.
(4) To build the initial graph: `python build_graph.py --save_path PaperGraph_release/ --building_original_graph_first`. This will add all the sources and their data.

### Adding Extra Connections
Then, we must add the extra connections (user-user following edges/etc.). For this:

(1) So that you don't have to run the previoous Initial Graph building step again, make a new directory and copy the graph there. `mkdir PaperGraph_release_connected/` and then `cp PaperGraph_release/* PaperGraph_release_connected/`
(2) Then, build the graph so it is connected. `python build_graph.py --save_path PaperGraph_release_connected/`

## Training for Node Classification
To train the graph for node classification:
```
python -u GNN_model.py ----path_where_graph_is PaperGraph_release_connected/  --path_to_save_model PaperGraph_test_release_connected_model_path --hidden_features 128 --lr 0.001 --batch-size 128 --data_splits_key_to_run_when_doing_one 0 --gpu 0 --n-layers 5 --num-workers 0 --just_train_nc --curr_port 22181 --n-epochs 100 --save_only_at_good_dev
```

## Running the Inference Operators
Coming soon

## Using Your Own Embeddings
This is possible now, but the tutorial is coming soon.