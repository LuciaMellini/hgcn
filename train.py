from __future__ import division
from __future__ import print_function

import os
os.environ.update({
    'MALLOC_CHECK_': '0',
    'PYTHONMALLOC': 'malloc',
    'GLIBC_TUNABLES': 'glibc.malloc.tcache_count=0',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'CUDA_LAUNCH_BLOCKING': '1'
})



import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data, data_to_adj, data_to_pos_neg_indices, get_global_indices_from_batch
from utils.train_utils import get_dir_name, format_metrics
from torch.amp import autocast, GradScaler

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import batched_negative_sampling
from tqdm import tqdm


def train(args):
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    train_data, valid_data, test_data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))

    args.n_nodes = train_data.num_nodes + valid_data.num_nodes + test_data.num_nodes
    args.num_neighbours = [25, 10]
    # Calculate feat_dim as the product of all elements in num_neighbours list
    args.feat_dim = int(np.prod(args.num_neighbours)*args.batch_size)
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['node'].x.max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = args.neg_sampling*(train_data.num_edges + valid_data.num_edges + test_data.num_edges)
        args.nb_edges = train_data.num_edges + valid_data.num_edges + test_data.num_edges
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1
    
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
    
    if args.save:
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        
    # Model
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    scaler = GradScaler('cuda')
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
          
        # for key, value in data.items():
        #     if isinstance(value, torch.Tensor):
        #         data[key] = value.to(args.device)
            #logging.info(f"Data tensor {x} has dtype {data[x].dtype}")

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
        
    train_loader = NeighborLoader(
        train_data,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbours,  # Use all neighbors (like full-batch)
        shuffle=True,
        # input_nodes=torch.arange(1000, dtype=torch.long, device=args.device),  # Use all nodes for training
        drop_last=True,  # Drop the last incomplete batch
    )
  
    for epoch in range(args.epochs):
        #logging.info("Before training: "+torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        t = time.time()
        model.train()    
        #with torch.autograd.set_detect_anomaly(True):
        total_loss = 0
        for batch in train_loader:
            data = {}
            data['adj_train'] = data_to_adj(batch, args.feat_dim)
            data['features'] = torch.eye(args.feat_dim, dtype=torch.bool)
            for key, value in data.items():
                data[key] = value.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'): 
                batch_embeddings = model.encode(data['features'], data['adj_train'])
                #logging.info("After model encode: "+torch.cuda.memory_summary())
                neg_edge_index = batched_negative_sampling(
                    edge_index=batch.edge_index,
                    batch = torch.ones(batch.num_nodes, dtype=torch.int64),
                    num_neg_samples=int(args.neg_sampling * batch.num_edges),
                )
                batch.neg_edge_index = neg_edge_index    
                data['train_edges'], data['train_edges_false'] = data_to_pos_neg_indices(batch)   
                train_metrics = model.compute_metrics(batch_embeddings, data, 'train')
                loss = train_metrics['loss']
                #logging.info("After compute metrics: " + torch.cuda.memory_summary())

            # del embeddings
            # gc.collect()
            # torch.cuda.empty_cache()
            #logging.info("After delete embeddings: " + torch.cuda.memory_summary())
            scaler.scale(loss).backward()


            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            break
        
        inference_loader = NeighborLoader(
            train_data,
            num_neighbors=[-1,-1],  # Use all neighbors (like full-batch)
            batch_size=args.batch_size*5,           # Process in batches if GPU memory is limited
            shuffle=False,           # No shuffling for inference
            # input_nodes=torch.arange(1000, dtype=torch.long, device=args.device),  # Use all nodes for inference
            drop_last=True,  # Do not drop the last batch
            # disjoint=True,  # Ensure disjoint batches for inference
        )    
        
        model.eval()
        embeddings = torch.zeros((args.n_nodes, args.dim), dtype=torch.bfloat16, device=args.device)
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                for batch in inference_loader:
                    data = {}
                    data['adj_train'] = data_to_adj(batch, args.feat_dim)
                    data['features'] = torch.eye(args.feat_dim, dtype=torch.bool)
                    for key, value in data.items():
                        data[key] = value.to(args.device)
                    idx_mapping = get_global_indices_from_batch(batch)
                    seed_indices = batch.input_id  # Original global indices of all nodes in the batch
                    # Find positions of seed_indices in idx_mapping
                    positions = []
                    for idx in seed_indices:
                        positions_for_idx = torch.where(idx_mapping == idx)[0]
                        positions.append(positions_for_idx[0] if positions_for_idx.size(0) > 0 else None)
                    positions = [p for p in positions if p is not None]
                    positions = torch.tensor(positions, device=args.device)
                    # logging.info("None positions: " + str(torch.sum(positions == None)))
                    #logging.info("Before model encode: "+torch.cuda.memory_summary())
                    local_embeddings = model.encode(data['features'], data['adj_train'])
                    # Check if embeddings at seed_indices are non-zero
                    is_nonzero = torch.any(embeddings[seed_indices] != 0, dim=1)
                    nonzero_count = torch.sum(is_nonzero).item()
                    embeddings[seed_indices] = local_embeddings.to(torch.bfloat16)[positions, :]
                    
                    
                    
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                format_metrics(train_metrics, 'train'),
                                'time: {:.4f}s'.format(time.time() - t)
                                ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad(): 
                with autocast(device_type='cuda',dtype=torch.bfloat16):
                    data['valid_edges'], data['valid_edges_false'] = data_to_pos_neg_indices(valid_data)
                    data['valid_edges_false'] = data['valid_edges_false'].t()
                    val_metrics = model.compute_metrics(embeddings, data, 'valid')
                    if (epoch + 1) % args.log_freq == 0:
                        logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                    if model.has_improved(best_val_metrics, val_metrics):
                        data['test_edges'], data['test_edges_false'] = data_to_pos_neg_indices(test_data)
                        data['test_edges_false'] = data['test_edges_false'].t()

                        best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                        best_emb = embeddings.cpu()
                        if args.save:
                            np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.float().detach().numpy())
                        best_val_metrics = val_metrics
                        counter = 0
                    else:
                        counter += 1
                        if counter == args.patience and epoch > args.min_epochs:
                            logging.info("Early stopping")
                            break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'valid')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.float().cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

if __name__ == '__main__':
    args = parser.parse_args()    
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train(args)