# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import sys
import os
import time
import ast
import os.path as osp

import graphlearn_torch as glt
import torch
import torch_geometric.transforms as T
from ogb.lsc import MAG240MDataset
from dgl.partition import get_peak_mem
from torch_geometric.utils import from_dgl
import torch_geometric as pyg

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from load_graph import load_dataset



def load_mag240m():
  g, num_classes, predict_category, list_of_metapaths, reverse_edge_type_prefix = load_dataset('mag240m', root='../dataset')
  g = g.formats('coo')
  pyg_g = from_dgl(g)

  # torch.save(pyg_g, '../dataset/donor2/graph.pt')
  return pyg_g, num_classes, predict_category


def partition_dataset(path: str,
                      out_path: str,
                      num_partitions: int,
                      chunk_size: int,
                      edge_assign_strategy: str='by_src',
                      cache_ratio: float=0.25):
  print(f'-- Loading {path} ...')
  dataset = 'mag240m'

  print(f"Peak memory usage: {get_peak_mem():.2f} GB")
  start = time.time()
  data, num_classes, predict_category = load_mag240m()
  print(f'-- Loading {path} takes {time.time() - start} seconds')
  print(f"Peak memory usage: {get_peak_mem():.2f} GB")
  start = time.time()
  node_num = {k : v.shape[0] for k, v in data.feat_dict.items()}

  print('-- Saving label ...')
  label_dir = osp.join(out_path, f'{dataset}-label')
  glt.utils.ensure_dir(label_dir)
  torch.save(data[predict_category].label.squeeze(), osp.join(label_dir, 'label.pt'))


  train_mask = data[predict_category].train_mask.squeeze()
  val_mask = data[predict_category].val_mask.squeeze()
  test_mask = data[predict_category].test_mask.squeeze()
  print('-- Partitioning training idx ...')
  train_idx = train_mask.nonzero(as_tuple=False).squeeze()
  train_idx = train_idx.split(train_idx.size(0) // num_partitions)
  train_idx_partitions_dir = osp.join(out_path, f'{dataset}-train-partitions')
  glt.utils.ensure_dir(train_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(train_idx[pidx], osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning validation idx ...')
  val_idx = val_mask.nonzero(as_tuple=False).squeeze()
  val_idx = val_idx.split(val_idx.size(0) // num_partitions)
  val_idx_partitions_dir = osp.join(out_path, f'{dataset}-val-partitions')
  glt.utils.ensure_dir(val_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(val_idx[pidx], osp.join(val_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning test idx ...')
  test_idx = test_mask.nonzero(as_tuple=False).squeeze()
  test_idx = test_idx.split(test_idx.size(0) // num_partitions)
  test_idx_partitions_dir = osp.join(out_path, f'{dataset}-test-partitions')
  glt.utils.ensure_dir(test_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(test_idx[pidx], osp.join(test_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Initializing graph ...')
  csr_topo = glt.data.Topology(edge_index=data.edge_index, input_layout='COO')
  graph = glt.data.Graph(csr_topo, mode='ZERO_COPY')

  print('-- Sampling hotness ...')
  glt_sampler = glt.sampler.NeighborSampler(graph, args.num_nbrs)
  node_probs = []
  for pidx in range(num_partitions):
    seeds = train_idx[pidx]
    prob = glt_sampler.sample_prob(seeds, node_num)
    node_probs.append(prob.cpu())

  print('-- Partitioning graph and features ...')
  print(f"Peak memory usage: {get_peak_mem():.2f} GB")
  partitions_dir = osp.join(out_path, f'{dataset}-partitions')
  partitioner = glt.partition.FrequencyPartitioner(
    output_dir=partitions_dir,
    num_parts=num_partitions,
    num_nodes=node_num,
    edge_index=data.edge_index_dict,
    probs=node_probs,
    node_feat=data.feat_dict,
    node_feat_dtype=torch.float16,
    edge_assign_strategy=edge_assign_strategy,
    chunk_size=chunk_size,
    cache_ratio=cache_ratio
  )
  partitioner.partition()
  print(f"Peak memory usage: {get_peak_mem():.2f} GB")
  print(f'-- Partitioning graph and features takes {time.time() - start} seconds')


if __name__ == '__main__':
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'dataset', 'donor2')
  glt.utils.ensure_dir(root)
  parser = argparse.ArgumentParser(description="Arguments for partitioning ogbn datasets.")
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--out_path', type=str, default=root,
      help='output path')
  parser.add_argument('--dataset', type=str, default='mag240m',
      help='dataset')
  parser.add_argument("--num_partitions", type=int, default=2,
      help="Number of partitions")
  parser.add_argument("--chunk_size", type=int, default=10000,
      help="Chunk size for feature partitioning.")
  parser.add_argument("--edge_assign_strategy", type=str, default='by_src',
      help="edge assign strategy can be either 'by_src' or 'by_dst'")
  parser.add_argument(
    "--num_nbrs",
    type=ast.literal_eval,
    default='[25,20]',
    help="The number of neighbors to sample hotness for feature caching.",
  )
  parser.add_argument(
    "--cache_ratio",
    type=float,
    default=0.25,
    help="The proportion to cache features per partition.",
  )
  args = parser.parse_args()

  if args.num_partitions > 2:
    args.out_path = osp.join(args.out_path, args.dataset + f'_{args.num_partitions}')
  else:
    args.out_path = osp.join(args.out_path, args.dataset) 
  os.makedirs(args.out_path, exist_ok=True)

  partition_dataset(
    args.path,
    args.out_path,
    num_partitions=args.num_partitions,
    chunk_size=args.chunk_size,
    edge_assign_strategy=args.edge_assign_strategy,
    cache_ratio=args.cache_ratio
  )
