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
import os.path as osp

import graphlearn_torch as glt
import torch
from torch_geometric.utils import from_dgl
import dgl
import torch_geometric.transforms as T

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from load_graph import load_dataset


def load_donor():
  g, num_classes, predict_category, list_of_metapaths, reverse_edge_type_prefix = load_dataset('donor', root='../dataset')
  g = g.formats('coo')
  pyg_g = from_dgl(g)

  pyg_g = T.ToUndirected()(pyg_g)
  pyg_g = T.AddSelfLoops()(pyg_g)

  # torch.save(pyg_g, '../dataset/donor2/graph.pt')
  return pyg_g, num_classes, predict_category


def partition_dataset(path: str,
                      out_path: str,
                      num_partitions: int,
                      chunk_size: int,
                      edge_assign_strategy: str='by_src'):
  print(f'-- Loading {path} ...')
  dataset = 'donor'

  data, num_classes, predict_category = load_donor()
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

  print('-- Partitioning graph and features ...')
  partitions_dir = osp.join(out_path, f'{dataset}-partitions')
  partitioner = glt.partition.RandomPartitioner(
    output_dir=partitions_dir,
    num_parts=num_partitions,
    num_nodes=node_num,
    edge_index=data.edge_index_dict,
    node_feat=data.feat_dict,
    node_feat_dtype=torch.float64,
    edge_assign_strategy=edge_assign_strategy,
    chunk_size=chunk_size,
  )
  partitioner.partition()


if __name__ == '__main__':
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'dataset', 'donor2')
  glt.utils.ensure_dir(root)
  parser = argparse.ArgumentParser(description="Arguments for partitioning ogbn datasets.")
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--out_path', type=str, default=root,
      help='output path')
  parser.add_argument('--dataset', type=str, default='donor',
      help='dataset')
  parser.add_argument("--num_partitions", type=int, default=2,
      help="Number of partitions")
  parser.add_argument("--chunk_size", type=int, default=10000,
      help="Chunk size for feature partitioning.")
  parser.add_argument("--edge_assign_strategy", type=str, default='by_src',
      help="edge assign strategy can be either 'by_src' or 'by_dst'")

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
  )
