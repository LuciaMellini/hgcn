"""Data utils functions for pre-processing and data loading."""
import os

import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
import gc
import logging
from torch_sparse import SparseTensor
import warnings

def get_edge_index_from_df(df, node_mapping):
    """
    Convert a DataFrame of edges to a PyTorch edge index tensor.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'subject' and 'object' columns.
        node_mapping (dict): Mapping from node names to indices.
        
    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges].
    """
    edge_index_values = df.apply(
        lambda row: [node_mapping[row['subject']], node_mapping[row['object']]], 
        axis=1
    ).tolist()
    return torch.tensor(edge_index_values).t()

def get_data_object_from_df(df, node_mapping):
    """
    Convert a DataFrame of edges to a PyTorch geometric Data object.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'subject' and 'object' columns.
        node_mapping (dict): Mapping from node names to indices.
        
    Returns:
        Data: PyTorch geometric Data object with edge_index and num_nodes.
    """
    edge_index = get_edge_index_from_df(df, node_mapping)
    num_nodes = len(node_mapping)
    
    return Data(edge_index=edge_index, num_nodes=num_nodes, node_mapping=node_mapping)

def apply_negative_sampling(data, neg_sampling_ratio=1.0):
    """
    Apply negative sampling to the edge index of a PyTorch geometric Data object.
    
    Args:
        data (Data): PyTorch geometric Data object with edge_index.
        neg_sampling_ratio (float): Ratio of negative samples to positive samples.
        
    Returns:
        Data: Updated Data object with negative edges added.
    """
    if neg_sampling_ratio <= 0:
        return data
    
    # Generate negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=int(neg_sampling_ratio * data.num_edges),
    )
    
    data.neg_edge_index = neg_edge_index.t()    
    return data

def load_data(args, datapath):

    edges = pd.read_csv(os.path.join(datapath, "{}.edges.csv".format(args.dataset)), 
                       low_memory=True)
    nodes = pd.read_csv(os.path.join(datapath, "{}.nodes.csv".format(args.dataset)), 
                       low_memory=True)
    
    # Report data sizes
    
    node_mapping = {node_id: i for i, node_id in enumerate(nodes['name'])}

    data_split = {}
    if args.rnd_split:    
        data = get_data_object_from_df(edges, node_mapping)
        
        # Randomly split edges into train, validation, and test sets
        splitter = RandomLinkSplit(
            num_val=args.val_prop,
            num_test=args.test_prop,
        )
        data_split['train'], data_split['valid'], data_split['test'] = splitter(data)
        
    else:
        data_split = {}
        for split in ['train', 'valid', 'test']:
            data_split[split] = get_data_object_from_df(
                edges[edges['split'] ==  split], node_mapping
            )
            
    for split in ['train', 'valid', 'test']:
        if not split == 'train':
            data_split[split] = apply_negative_sampling(data_split[split], args.neg_sampling)

        data_split[split].validate(raise_on_error=True)
  
    return data_split['train'], data_split['valid'], data_split['test']


def get_global_indices_from_batch(batch):
    batch_indices = batch.n_id  # Original global indices of all nodes in the batch
    return batch_indices
  
def data_to_adj(data, max_num_nodes):
    adj = torch.sparse_coo_tensor(
        data.edge_index,
        torch.ones(data.num_edges, dtype=torch.bool),
        (max_num_nodes, max_num_nodes),
        dtype=torch.bool
    )
    
    return adj
    
def data_to_pos_neg_indices(data):
    
    pos_indices = data.edge_index
    neg_indices = data.neg_edge_index if hasattr(data, 'neg_edge_index') else torch.empty((2, 0), dtype=torch.long)  
    
    return pos_indices.clone().detach().to(torch.int).t(), neg_indices.clone().detach().to(torch.int).t()


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
