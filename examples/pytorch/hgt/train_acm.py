
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
test_string = ""
# test_string = "_test"  ## show test mode, affect the hist file name

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

with open(f"{data_path}data_dict_sparse_update.pkl", "rb") as f:
    data_dict_sparse = pickle.load(f)

with open(f"{data_path}matrix_data_have_node.pkl", "rb") as f:
    matrix_data_have_node = pickle.load(f)

with open(f"{data_path}ipaddress_dict.pkl", "rb") as f:
    ipaddress_dict = pickle.load(f)

with open(f"{data_path}matrix_label.pkl", "rb") as f:
    matrix_label = pickle.load(f)

def make_graph(start:int, end:int):
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
    G = dgl.heterograph({
        ('node', 'forward-relation', 'node') : sparseM_v.nonzero(),
        ('node', 'backward-relation', 'node') : sparseM_v.transpose().nonzero(),
        # ('date', 'have-node', 'node') : sparseM_date.nonzero(),
        # ('node', 'in-date', 'date') : sparseM_date.transpose().nonzero(),
    })
    print(G)

    global node_dict
    global edge_dict
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

    labels = matrix_label[1][end : end + 1]
    labels = torch.tensor(labels).float()
    return [G, labels]

if not run_my_code:
    data = scipy.io.loadmat(data_file_path)
#TODO: recover time stamp

parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')



parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_hid',   type=int, default=256)
parser.add_argument('--n_inp',   type=int, default=256)
parser.add_argument('--clip',    type=int, default=1.0) 
# parser.add_argument('--max_lr',  type=float, default=1e-3) 
if run_my_code:
    parser.add_argument('--max_lr',  type=float, default=1) 
else:
    parser.add_argument('--max_lr',  type=float, default=1e-3) 

args = parser.parse_args()
filename = f"lr{args.max_lr}_n{args.n_epoch}{test_string}"
filename_test = f"test_lr{args.max_lr}_n{args.n_epoch}{test_string}"
filename_val = f"val_lr{args.max_lr}_n{args.n_epoch}{test_string}"

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

loss_func = nn.MSELoss()

def train(model, train_set, test_set, val_set):
    train_step = torch.tensor(0)
    mape_hist = []
    r2_hist = []
    mape_val_hist = []
    r2_val_hist = []
    loss_hist = []
    tot_mape_hist = []
    tot_r2_hist = []
    tot_mape_val_hist = []
    tot_r2_val_hist = []
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        loss_single = 0
        for batch in train_set:
            input = batch[0]
            labels = batch[1]
            if run_my_code:
                # logits = model(G, 'date')
                logits = model(input, 'node')
                # loss = F.cross_entropy(logits.view(-1)[train_idx], labels[train_idx].to(device))
                # loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
                loss = loss_func(logits, labels.to(device))
            else:
                logits = model(G, 'paper')
                loss = F.cross_entropy(input, labels.to(device))
                # loss = torch.nn.MSELoss(logits[train_idx], labels[train_idx].to(device))
            # The loss is computed only for labeled nodes.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss_single += loss.item()
            print("1:{}".format(humanize.naturalsize(torch.cuda.memory_allocated(0))))
        train_step += 1
        loss_hist.append(loss_single / len(train_set))
        scheduler.step(train_step)
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            val_input = val_set[0]
            val_label = val_set[1]
            logits = model(val_input, 'node')
            print("3:{}".format(humanize.naturalsize(torch.cuda.memory_allocated(0))))
            test_metricses = []
            r2_scores = []
            for i in range(len(logits)):
                mape_score = mean_absolute_percentage_error(val_label, logits[i].cpu().detach().numpy())
                r2_score_single = r2_score(val_label, logits[i].cpu().detach().numpy())
                # print(test_metrics)
                test_metricses.append(mape_score)
                mape_val_hist.append(mape_score)
                r2_scores.append(r2_score_single)
                r2_val_hist.append(r2_score_single)
            mape_score_val = np.mean(test_metricses)            
            r2_score_single_val = np.mean(r2_scores)
            print('Epoch: %d LR: %.5f Loss %.4f, valMAPE %.4f, valR2 %.4f' % (
                epoch,
                optimizer.param_groups[0]['lr'], 
                loss.item(),
                mape_score_val,
                r2_score_single_val,
            ))
            tot_mape_val_hist.append(mape_score_val)
            tot_r2_val_hist.append(r2_score_single_val)
            torch.cuda.empty_cache()
        
        # test
        model.eval()
        test_input = test_set[0]
        test_label = test_set[1]
        logits = model(test_input, 'node')
        print("2:{}".format(humanize.naturalsize(torch.cuda.memory_allocated(0))))
        test_metricses = []
        r2_scores = []
        for i in range(len(logits)):
            mape_score = mean_absolute_percentage_error(test_label, logits[i].cpu().detach().numpy())
            r2_score_single = r2_score(test_label, logits[i].cpu().detach().numpy())
            # print(test_metrics)
            test_metricses.append(mape_score)
            mape_hist.append(mape_score)
            r2_scores.append(r2_score_single)
            r2_hist.append(r2_score_single)
        mape_score_test = np.mean(test_metricses)            
        r2_score_single_test = np.mean(r2_scores)
        torch.cuda.empty_cache()
        tot_mape_hist.append(mape_score_test)
        tot_r2_hist.append(r2_score_single_test)

    save_pkl(tot_mape_hist, "mape", filename_test)
    save_pkl(tot_mape_val_hist, "r2", filename_test)
    save_pkl(tot_r2_hist, "mape", filename_val)
    save_pkl(tot_r2_val_hist, "r2", filename_val)
    save_pkl(loss_hist, "loss", filename)

device = torch.device("cuda:0")

if run_my_code:
    pass
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
    pass
else:
    pvc = data['PvsC'].tocsr()
# p_selected = pvc.tocoo()
# # generate labels
# labels = pvc.indices
# if run_my_code:
#     pass
#     # labels = torch.tensor(labels).float()
# else:
#     labels = torch.tensor(labels).long()

# if run_my_code:
#     pass
# else:
#     # generate train/val/test split
#     pid = p_selected.row
#     shuffle = np.random.permutation(pid)
#     train_idx = torch.tensor(shuffle[0:800]).long()
#     val_idx = torch.tensor(shuffle[800:900]).long()
#     test_idx = torch.tensor(shuffle[900:]).long()

train_set = [
    make_graph(1, 7), 
    make_graph(2, 8),
    make_graph(3, 9),
    make_graph(4, 10),
    make_graph(5, 11)]
test_set = make_graph(6, 12)
# val_set = make_graph(6, 12)
val_set = []

model = HGT(train_set[0][0],
            node_dict, edge_dict,
            n_inp=args.n_inp,
            n_hid=args.n_hid,
            # n_out=labels.max().item()+1,  ## target bug point
            n_out=7,  ## target bug point
            n_layers=2,
            n_heads=4,
            use_norm = True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
print('Training HGT with #param: %d' % (get_n_params(model)))
train(model, train_set, test_set, val_set)




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
