
# coding: utf-8

# In[1]:


import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import scipy.sparse
from model import *
import argparse
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)

torch.manual_seed(0)
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = './ACM.mat'

run_my_code = True

# urllib.request.urlretrieve(data_url, data_file_path)
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

if run_my_code:
    with open(f"{data_path}data_dict_sparse_update.pkl", "rb") as f:
        data_dict_sparse = pickle.load(f)

    with open(f"{data_path}matrix_data_have_node.pkl", "rb") as f:
        matrix_data_have_node = pickle.load(f)

    with open(f"{data_path}ipaddress_dict.pkl", "rb") as f:
        ipaddress_dict = pickle.load(f)

    predict_days = 1
    graph_num = 8
    label_num = graph_num + predict_days

    rows = np.array(data_dict_sparse[str(graph_num)][0])
    cols = np.array(data_dict_sparse[str(graph_num)][1])
    values = np.array(data_dict_sparse[str(graph_num)][2])

    sparseM_v = scipy.sparse.coo_matrix((values, (rows, cols)))

    label_position = 0
    for idx, date in enumerate(matrix_data_have_node[0]):
        if date > label_num:
            label_position = idx
            break
    rows_date  = np.array(matrix_data_have_node[0][:label_position]) - 1
    cols_date  = np.array(matrix_data_have_node[1][:label_position])
    values_date = np.array(matrix_data_have_node[2][:label_position])

    sparseM_date = scipy.sparse.coo_matrix((values_date, (rows_date, cols_date)))
else:
    data = scipy.io.loadmat(data_file_path)
#TODO: recover time stamp

parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')



parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_hid',   type=int, default=256)
parser.add_argument('--n_inp',   type=int, default=256)
parser.add_argument('--clip',    type=int, default=1.0) 
# parser.add_argument('--max_lr',  type=float, default=1e-3) 
if run_my_code:
    parser.add_argument('--max_lr',  type=float, default=1e-1) 
else:
    parser.add_argument('--max_lr',  type=float, default=1e-3) 

args = parser.parse_args()
filename = f"lr{args.max_lr}_n{args.n_epoch}"

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

loss_func = nn.MSELoss()

def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    mape_hist = []
    r2_hist = []
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        if run_my_code:
            logits = model(G, 'date')
            # loss = F.cross_entropy(logits.view(-1)[train_idx], labels[train_idx].to(device))
            # loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
            loss = loss_func(logits[train_idx], labels[train_idx].to(device))
        else:
            logits = model(G, 'paper')
            loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
            # loss = torch.nn.MSELoss(logits[train_idx], labels[train_idx].to(device))
        # The loss is computed only for labeled nodes.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 1 == 0:
            model.eval()
            if run_my_code:
                logits = model(G, 'date')
            else:
                logits = model(G, 'paper')
            test_metricses = []
            r2_scores = []
            for i in range(len(logits)):
                test_metrics = mean_absolute_percentage_error(labels, logits[i].cpu().detach().numpy())
                r2_score_single = r2_score(labels, logits[i].cpu().detach().numpy())
                # print(test_metrics)
                test_metricses.append(test_metrics)
                mape_hist.append(test_metrics)
                r2_scores.append(r2_score_single)
                r2_hist.append(r2_score_single)
            test_metrics = np.mean(test_metricses)
            
            r2_score_single = np.mean(r2_scores)
            # print("mean:", test_metrics)
            # pred   = logits.argmax(1).cpu()
            # print(logits, labels)
            # train_acc = (pred[train_idx] == labels[train_idx]).float().mean() ##TODO: metrics
            # val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
            # test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
            # if best_val_acc < val_acc:
            #     best_val_acc = val_acc
            #     best_test_acc = test_acc
            print('Epoch: %d LR: %.5f Loss %.4f, MAPE %.4f, R2 %.4f' % (
                epoch,
                optimizer.param_groups[0]['lr'], 
                loss.item(),
                test_metrics,
                r2_score_single
            ))
    save_pkl(mape_hist, "mape", filename)
    save_pkl(r2_hist, "r2", filename)

device = torch.device("cuda:0")

if run_my_code:
    G = dgl.heterograph({
            ('node', 'forward-relation', 'node') : sparseM_v.nonzero(),
            ('node', 'backward-relation', 'node') : sparseM_v.transpose().nonzero(),
            ('date', 'have-node', 'node') : sparseM_date.nonzero(),
            ('node', 'in-date', 'date') : sparseM_date.transpose().nonzero(),
        })
    print(G)
else:
    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        })
    print(G)

if run_my_code:
    with open(f"{data_path}matrix_label.pkl", "rb") as f:
        matrix_label = pickle.load(f)
    rows_label   = np.array(matrix_label[0][:label_num]) - 1
    cols_label   = np.array(matrix_label[1][:label_num])
    values_label = np.array(matrix_label[2][:label_num])

    sparseM_label = scipy.sparse.coo_matrix((values_label, (rows_label, cols_label)))
    pvc = sparseM_label.tocsr()
else:
    pvc = data['PvsC'].tocsr()
p_selected = pvc.tocoo()
# generate labels
labels = pvc.indices
if run_my_code:
    labels = torch.tensor(labels).float()
else:
    labels = torch.tensor(labels).long()

if run_my_code:
    # generate train/val/test split
    # pid = p_selected.col
    pid = np.array(range(label_num))
    shuffle = np.random.permutation(pid)
    # train_idx = torch.tensor(shuffle[:-1]).long()
    # val_idx = torch.tensor(shuffle[-4:-3]).long()
    # test_idx = torch.tensor(shuffle[-3:]).long()
    train_idx = pid
    val_idx = pid
    test_idx = pid
else:
    # generate train/val/test split
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()


node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

#     Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad = False)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb

G = G.to(device)

model = HGT(G,
            node_dict, edge_dict,
            n_inp=args.n_inp,
            n_hid=args.n_hid,
            # n_out=labels.max().item()+1,  ## target bug point
            n_out=label_num,  ## target bug point
            n_layers=2,
            n_heads=4,
            use_norm = True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
print('Training HGT with #param: %d' % (get_n_params(model)))
train(model, G)




# model = HeteroRGCN(G,
#                    in_size=args.n_inp,
#                    hidden_size=args.n_hid,
#                    out_size=labels.max().item()+1).to(device)
# optimizer = torch.optim.AdamW(model.parameters())
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
# print('Training RGCN with #param: %d' % (get_n_params(model)))
# train(model, G)



# model = HGT(G,
#             node_dict, edge_dict,
#             n_inp=args.n_inp,
#             n_hid=args.n_hid,
#             n_out=labels.max().item()+1,
#             n_layers=0,
#             n_heads=4).to(device)
# optimizer = torch.optim.AdamW(model.parameters())
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
# print('Training MLP with #param: %d' % (get_n_params(model)))
# train(model, G)
