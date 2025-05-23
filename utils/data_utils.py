"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import gc
from sklearn.preprocessing import LabelEncoder



def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, datapath, args.use_feats, args.data_split, args.val_prop, args.test_prop, args.split_seed, args.normalize_adj, args.normalize_feats, args.neg_sampling)
        
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, use_feats, normalize_feats):
    if use_feats:
        if normalize_feats:
            features = normalize(features)
        
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0], dtype=adj.dtype))
        
    def get_torch_dtype_from_np_dtype(dtype):
        return torch.from_numpy(np.array([], dtype=dtype)).dtype
    
    features = sparse_mx_to_torch_sparse_tensor(features, dtype = get_torch_dtype_from_np_dtype(features.dtype))
    adj = sparse_mx_to_torch_sparse_tensor(adj, dtype = get_torch_dtype_from_np_dtype(adj.dtype))
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx, dtype=torch.float):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col))
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges_rnd(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = adj.nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.csr_matrix(1. - adj.toarray()).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)   
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]

    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape, dtype=adj.dtype)
    return adj_train, torch.IntTensor(train_edges), torch.IntTensor(train_edges_false), torch.IntTensor(val_edges), torch.IntTensor(val_edges_false), torch.IntTensor(test_edges), torch.IntTensor(test_edges_false)  

def mask_edges(adj, edges, neg_sampling):
    try:
        train_idx = edges[edges['split'] == 'train'].index
        valid_idx = edges[edges['split'] == 'valid'].index
        test_idx = edges[edges['split'] == 'test'].index
    except KeyError:
        raise KeyError("The edges DataFrame does not contain the 'split' column. Please ensure the DataFrame has been properly prepared with train/valid/test splits.")

    # Extract subject-object pairs as numpy arrays
    train_edges = edges.loc[train_idx, ['subject', 'object']].values
    val_edges = edges.loc[valid_idx, ['subject', 'object']].values
    test_edges = edges.loc[test_idx, ['subject', 'object']].values

    if neg_sampling==-1:
        train_edges_false = np.empty((0, 2), dtype=np.bool)
        val_edges_false = np.empty((0, 2), dtype=np.bool)
        test_edges_false = np.empty((0, 2), dtype=np.bool)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape, dtype=adj.dtype)
    
    return adj_train, torch.IntTensor(train_edges), torch.IntTensor(train_edges_false), torch.IntTensor(val_edges), torch.IntTensor(val_edges_false), torch.IntTensor(test_edges), torch.IntTensor(test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, data_path, use_feats, data_split, val_prop, test_prop, split_seed, normalize_adj, normalize_feats, neg_sampling):
    nodes, edges = None, None
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset.startswith(('tr', 'Disease', 'GO', 'Genomic', 'Phen')):
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset.startswith('split'):
        nodes, edges = id_string_to_numerical(dataset, data_path)
        adj, features = load_data_kg(nodes, edges, use_feats)[:2]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'features': features}  
                  
    if data_split:
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges_rnd(
                    adj, val_prop, test_prop, split_seed
            )
    else:        
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, edges, neg_sampling
            )
    
    del adj
    torch.cuda.empty_cache()
    gc.collect()
    data['adj_train'] = adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
    data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train'], data['features'] = process(
        data['adj_train'], data['features'], normalize_adj, use_feats, normalize_feats
    )
    if dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset.startswith('transitive'):
            adj, features, labels = load_data_kg(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'kg':
            adj, features, labels = load_data_kg(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph), dtype = np.bool)
    # Convert adjacency matrix to CSR format with appropriate shape and dtype
    adj = sp.csr_matrix(adj, dtype=np.bool)
    if not use_feats:
        features = sp.eye(adj.shape[0], dtype=np.bool)
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()        
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        #adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
        
    else:
        features = sp.eye(adj.shape[0], dtype=np.bool)
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)), allow_pickle=True)
    return sp.csr_matrix(adj, dtype=np.bool), features, labels

def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features
    
def id_string_to_numerical(dataset_str, data_path):
    edges = pd.read_csv(os.path.join(data_path, f"{dataset_str}.edges.csv"), low_memory=True)
    nodes = pd.read_csv(os.path.join(data_path, f"{dataset_str}.nodes.csv"), low_memory=True)
    name_to_index_map = {v: k for k, v in nodes['name'].to_dict().items()}
    edges['subject'] = edges['subject'].map(name_to_index_map)
    edges['object'] = edges['object'].map(name_to_index_map)
    nodes['name'] = nodes['name'].map(name_to_index_map)
    return nodes, edges
    
def load_data_kg(nodes, edges, use_feats=False):    
    n_nodes = len(nodes)
    nodes = nodes.sort_values(by='name')
    labels = nodes['type'].values.flatten()
    del nodes
    rows = edges['subject'].values.flatten()
    cols = edges['object'].values.flatten()
    adj = sp.csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.bool)
    if use_feats: 
        def strings_to_categorical(strings):
            label_encoder = LabelEncoder()
            categories = label_encoder.fit_transform(strings)
            return dict(zip(strings, categories))
        
        predicate_to_category_map = strings_to_categorical(edges['predicate'].unique())
        edges['predicate'] = edges['predicate'].map(predicate_to_category_map)
        data = edges['predicate'].values.flatten()
        features = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    else:
        features = sp.eye(n_nodes, dtype=np.bool, format='csr')          
    return adj, features, labels
