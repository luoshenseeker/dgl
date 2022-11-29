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

out_put_size = 1

def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


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

def make_graph(start:int, end:int, transform):
    rows = []
    cols = []
    values = []
    for i in range(start, end+1):
        rows.extend(data_dict_sparse[str(i)][0])
        cols.extend(data_dict_sparse[str(i)][1])
        values.extend(data_dict_sparse[str(i)][2])
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

    labels = matrix_label[1][end : end + 1]
    labels = torch.tensor(labels).float()
    return [G, labels]

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
    
def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              . format(epoch, loss.item(), acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    args = parser.parse_args()
    print(f'Training with DGL built-in GraphConv module.')
 
    # load and preprocess dataset
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    # if args.dataset == 'cora':
    #     data = CoraGraphDataset(transform=transform)
    # elif args.dataset == 'citeseer':
    #     data = CiteseerGraphDataset(transform=transform)
    # elif args.dataset == 'pubmed':
    #     data = PubmedGraphDataset(transform=transform)
    # else:
    #     raise ValueError('Unknown dataset: {}'.format(args.dataset))
    # g = data[0]
    g = make_graph(1, 7, transform=transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
        
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5).to(device)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model    
    in_size = features.shape[1]
    out_size = out_put_size
    model = GCN(in_size, 16, out_size).to(device)

    # model training
    print('Training...')
    train(g, features, labels, masks, model)
    
    # test the model
    print('Testing...')
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
