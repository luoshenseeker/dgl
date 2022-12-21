#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pickle
import numpy as np
import scipy.sparse
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import argparse
import copy
import networkx as nx
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

out_put_size = 1

def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)

#%%
import platform

sysstr = platform.system()
if sysstr == "Linux":
    # paths
    data_path = "/home/zhouxk/data/"
elif sysstr == "Windows":
    # paths
    data_path = "C:\\Users\\luoshenseeker\\home\\work\\科研\\workbentch\\misc\\data\\data_with_label\\"
else:
    data_path = "/home/zhouxk/data/"

with open(f"{data_path}data_dict_sparse_update.pkl", "rb") as f:
    data_dict_sparse = pickle.load(f)

with open(f"{data_path}matrix_data_have_node.pkl", "rb") as f:
    matrix_data_have_node = pickle.load(f)

with open(f"{data_path}ipaddress_dict.pkl", "rb") as f:
    ipaddress_dict = pickle.load(f)

with open(f"{data_path}matrix_label.pkl", "rb") as f:
    matrix_label = pickle.load(f)

def make_graph(start:int, end:int, transform, device):
    rows = []
    cols = []
    values = []
    for i in range(start, end+1):
        rows.extend(data_dict_sparse[str(i)][0])
        cols.extend(data_dict_sparse[str(i)][1])
        values.extend(data_dict_sparse[str(i)][2])

    # make the graph no direction
    tmp_rows = copy.deepcopy(rows)
    rows.extend(cols)
    cols.extend(tmp_rows)
    values.extend(values)
    del tmp_rows

    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)
    sparseM_v = scipy.sparse.coo_matrix((values, (rows, cols)))

    # label_position = 0
    # for idx, date in enumerate(matrix_data_have_node[0]):
    #     if date > label_num:
    #         label_position = idx
    #         break
    # rows_date  = np.array(matrix_data_have_node[0][:label_position]) - 1
    # cols_date  = np.array(matrix_data_have_node[1][:label_position])
    # values_date = np.array(matrix_data_have_node[2][:label_position])

    # sparseM_date = scipy.sparse.coo_matrix((values_date, (rows_date, cols_date)))
    G = dgl.from_scipy(sparseM_v)
    G = transform(G)
    print(G)

    G = G.to(device)
    nodes_number = G.num_nodes()

    feature = [1 for _ in range(500)]
    features = [feature for _ in range(nodes_number)]
    features = torch.tensor(features).float().to(device)
    labels_tot_num = matrix_label[1][end : end + 1]
    labels = [labels_tot_num for _ in range(nodes_number)]
    labels = torch.tensor(labels).float().to(device)
    return [[G, features], labels]

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
def evaluate(model, val_set):
    model.eval()
    with torch.no_grad():
        input_g = val_set[0][0]
        input_f = val_set[0][1]
        labels = val_set[1]
        logits = model(input_g, input_f)
        test_metricses = []
        r2_scores = []

        mape_score = mean_absolute_percentage_error(labels.cpu().detach().numpy(), logits.cpu().detach().numpy())
        r2_score_single = r2_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy())
        test_metricses.append(mape_score)
        r2_scores.append(r2_score_single)

        print(f"Output status: min {logits.cpu().detach().numpy().min()}, max {logits.cpu().detach().numpy().max()}, mean {logits.cpu().detach().numpy().mean()}")

        mape_score_test = np.mean(test_metricses)            
        r2_score_single_test = np.mean(r2_scores)
        torch.cuda.empty_cache()
        return mape_score_test, r2_score_single_test

def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)

def train(model, train_set):
    # define train/val samples, loss function and optimizer
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_hist = []
    # training loop
    for epoch in range(300):
        model.train()
        loss_single = 0
        for batch in train_set:
            input_g = batch[0][0]
            input_f = batch[0][1]
            labels = batch[1]
            logits = model(input_g, input_f)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_single += loss.item()
            loss_hist.append(loss_single)
        # acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f}"
              . format(epoch, loss_single))
    save_model(model)
    save_pkl(loss_hist, "loss", f"{get_model_number()}")

model_path = "output/models"

def get_model_number():
    names = os.listdir(model_path)
    tot = 0
    for name in names:
        if name.startswith("saved_model"):
            tot += 1
    return tot

def save_model(model):
    model_number = get_model_number()
    torch.save(model.state_dict(), f"{model_path}/saved_model{model_number+1}")

def load_model(model_number=0):
    if model_number == 0:
        model_number = get_model_number()
        return torch.load(f"{model_path}/saved_model{model_number}")
    else:
        return torch.load(f"{model_path}/saved_model{model_number}")

train_status = True

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    if train_status:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="cora",
                            help="Dataset name ('cora', 'citeseer', 'pubmed').")
        args = parser.parse_args()
        print(f'Training with DGL built-in GraphConv module.')
    
        use_my_dataset = True
        # load and preprocess dataset
        
        if not use_my_dataset and args.dataset == 'cora':
            data = CoraGraphDataset(transform=transform)
        elif not use_my_dataset and args.dataset == 'citeseer':
            data = CiteseerGraphDataset(transform=transform)
        elif not use_my_dataset and args.dataset == 'pubmed':
            data = PubmedGraphDataset(transform=transform)
        # else:
        #     raise ValueError('Unknown dataset: {}'.format(args.dataset))
        
        if not use_my_dataset:
            g = data[0]
        # g = make_graph(1, 7, transform=transform, device=device)
            g = g.int().to(device)
            features = g.ndata['feat']
            labels = g.ndata['label']
            masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
            
        # make features and labels, features must be the same
        
        # adjs = map(lambda x: nx.adjacency_matrix(x), g) # this can make a list of nx.adjacency_matrix
        # feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
        #     x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
        # Identity matrix above, can't be used

        # normalization
            degs = g.in_degrees().float()
            norm = torch.pow(degs, -0.5).to(device)
            norm[torch.isinf(norm)] = 0
            g.ndata['norm'] = norm.unsqueeze(1)

        # create GCN model    
        in_size = 500
        out_size = out_put_size
        model = GCN(in_size, 16, out_size).to(device)

        train_set = [
            make_graph(1, 7, transform=transform, device=device), 
            make_graph(2, 8, transform=transform, device=device),
            make_graph(3, 9, transform=transform, device=device),
            make_graph(4, 10, transform=transform, device=device),
            make_graph(5, 11, transform=transform, device=device)]

        # model training
        print('Training...')
        train(model, train_set)

        
    
    # create GCN model    
    in_size = 500
    out_size = out_put_size
    model = GCN(in_size, 16, out_size).to(device)
    model.load_state_dict(load_model(0))
    test_set = make_graph(6, 12, transform=transform, device=device)
    # test the model
    print('Testing...')
    mape, r2 = evaluate(model, test_set)
    print("Test mape {:.4f}, r2 {:.4f}".format(mape, r2))
