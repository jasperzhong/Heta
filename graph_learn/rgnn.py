# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GATConv, GCNConv, SAGEConv, HGTConv, GraphConv


class RGNN(torch.nn.Module):
  r""" [Relational GNN model](https://arxiv.org/abs/1703.06103).

  Args:
    etypes: edge types.
    in_dim: input size.
    h_dim: Dimension of hidden layer.
    out_dim: Output dimension.
    num_layers: Number of conv layers.
    dropout: Dropout probability for hidden layers.
    model: "rsage" or "rgat".
    heads: Number of multi-head-attentions for GAT.
    node_type: The predict node type for node classification.

  """
  def __init__(self, ntypes, etypes, ntypes_to_in_dim, h_dim, out_dim, num_layers=2,
               dropout=0.2, model='rgat', heads=4, node_type=None):
    super().__init__()
    self.node_type = node_type
    if node_type is not None:
      self.lin = torch.nn.Linear(h_dim, out_dim)

    self.node_projs = torch.nn.ModuleDict()
    for ntype in ntypes:
      self.node_projs[ntype] = torch.nn.Linear(ntypes_to_in_dim[ntype], h_dim)

    self.convs = torch.nn.ModuleList()
    for i in range(num_layers):
      in_dim = h_dim
      h_dim = out_dim if (i == (num_layers - 1) and node_type is None) else h_dim
      if model == 'rsage':
        self.convs.append(HeteroConv({
            etype: SAGEConv(in_dim, h_dim, root_weight=False)
            for etype in etypes}))
      elif model == 'rgat':
        self.convs.append(HeteroConv({
            etype: GATConv(in_dim, h_dim // heads, heads=heads, add_self_loops=False)
            for etype in etypes}))
      elif model == 'rgcn':
        self.convs.append(HeteroConv({
          etype: GraphConv(in_dim, h_dim, add_self_loops=False)
          for etype in etypes}))
      elif model == 'hgt':
        self.convs.append(HGTConv(in_dim, h_dim, metadata=[ntypes, etypes], heads=heads))
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x_dict, edge_index_dict):
    # perform projection layer
    x_dict = {key: self.node_projs[key](x.type(torch.float32)) for key, x in x_dict.items()}

    for i, conv in enumerate(self.convs):
      
      for key in list(edge_index_dict.keys()):
        if key[0] not in x_dict or key[-1] not in x_dict:
          del edge_index_dict[key]
          
      x_dict = conv(x_dict, edge_index_dict)
      if i != len(self.convs) - 1:
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
    if hasattr(self, 'lin'): # for node classification
      return self.lin(x_dict[self.node_type])
    else:
      return x_dict
