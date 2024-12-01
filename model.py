"""hgnn model (R-GCN, R-GAT, and HGT)"""
import math
import time
from typing import Dict, List
import os

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
from dgl.distributed.graph_partition_book import NodePartitionPolicy
from dgl.distributed.kvstore import get_kvstore
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv, GraphConv, HeteroGraphConv

from gpu_cache import GPUCache


def _broadcast_layers(layers: List[nn.Module], src=0):
    """Function to broadcast the parameters from the given source rank.
    """
    if not dist.is_initialized():
        # no need to broadcast if not using distributed training
        return 

    for layer in layers:
        if isinstance(layer, nn.Parameter):
            dist.broadcast(layer.data, src=src)
        else:
            for p in layer.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)

def init_emb(shape, dtype):
    print("init emb shape: ", shape)
    arr = torch.zeros(shape, dtype=dtype)
    nn.init.uniform_(arr, -1.0, 1.0)
    return arr


class DistEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    g : DistGraph
        training graph
    embed_size : int
        Output embed size
    dgl_sparse_emb: bool
        Whether to use DGL sparse embedding
        Default: False
    embed_name : str, optional
        Embed name
    """

    def __init__(
        self,
        dev_id,
        g,
        embed_size,
        ntypes_w_feat,
        dataset,
        dgl_sparse_emb=False,
        feat_name="feat",
        embed_name="node_emb",
        partition_book=None,
        predict_category=None, 
        cache_method='none',
        args=None
    ):
        super(DistEmbedLayer, self).__init__()
        self.dev_id = dev_id
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.feat_name = feat_name
        self.g = g
        self.ntypes_w_feat = ntypes_w_feat
        self.predict_category = predict_category
        self.dgl_sparse_emb = dgl_sparse_emb
        self.dataset = dataset

        self.node_projs = nn.ModuleDict()
        ntypes = partition_book.ntypes if partition_book is not None else g.ntypes
        ntypes_wo_feat = set(ntypes) - set(ntypes_w_feat)
        ntypes_w_feat = sorted(set(ntypes_w_feat) & set(g.ntypes))
        print(f"rank {dist.get_rank()} ntypes_w_feat: {ntypes_w_feat}")

        self._proj_time = 0
        self._fetch_feat_time = 0

        self._cache_hit_rate = {ntype: [] for ntype in ntypes}
        self._local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self._machine_id = dist.get_rank() // self._local_world_size
        self._gpu_id = dist.get_rank() % self._local_world_size
        self._gpu_caches = {}
        self.feat = {}

        label = 'dgl' if partition_book is None else 'Heta'
        for ntype in ntypes_w_feat:
            if partition_book is not None:
                part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
            else:
                part_policy = g.get_node_partition_policy(ntype) 
            
            if dataset == 'mag240m' and label != 'dgl':
                self.feat[ntype] = np.load('/dev/shm/paper.npy', mmap_mode='r')
                self.node_projs[ntype] = nn.Linear(
                    self.feat[ntype].shape[1], embed_size
                )
            elif ntype in g.ntypes and feat_name in g.nodes[ntype].data:
                self.node_projs[ntype] = nn.Linear(
                    g.nodes[ntype].data[feat_name].shape[1], embed_size
                )
            elif dataset == 'igb-het':
                self.node_projs[ntype] = nn.Linear(
                    1024, embed_size
                )

            nn.init.xavier_uniform_(self.node_projs[ntype].weight)
            print("node {} has data {}".format(ntype, feat_name))
            if cache_method != 'none':
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype) 
                part_size = part_policy.get_part_size()
                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")
                if dataset == 'mag240m' and label != 'dgl':
                    feat_size = self.feat[ntype].shape[1]
                    feat_dtype = torch.float16
                else:
                    feat_size = g.nodes[ntype].data[feat_name].shape[1]
                    feat_dtype = g.nodes[ntype].data[feat_name].dtype

                try:
                    cache_nodes = np.load(f"cache/{label}/{dataset}_{cache_method}/part{self._machine_id}/{ntype}.npy")
                except FileNotFoundError:
                    print(f"Rank {dist.get_rank()} cache file for {ntype} not found")
                    continue
                if dataset == 'donor':
                    cache = GPUCache(len(cache_nodes), g.number_of_nodes(ntype), feat_size, feat_dtype, dev_id)
                else:
                    cache = GPUCache(len(cache_nodes), part_policy.get_size(), feat_size, feat_dtype, dev_id)
                if dataset == 'mag240m' and label != 'dgl':
                    cache.init_cache(cache_nodes, g, ntype, init_data=self.feat[ntype])
                else:
                    cache.init_cache(cache_nodes, g, ntype)
                self._gpu_caches[ntype] = cache


        if dgl_sparse_emb:
            self.node_embeds = {}
            for ntype in sorted(ntypes_wo_feat):
                # We only create embeddings for nodes without node features.
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype) 
                part_size = part_policy.get_part_size()
                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")

                gpu_cache = None
                if cache_method != 'none' and part_size > 0 and label != 'dgl':
                    cache_nodes = np.load(f"cache/{label}/{dataset}_{cache_method}/part{self._machine_id}/{ntype}.npy")
                    local_cache_nodes = cache_nodes[cache_nodes % self._local_world_size == self._gpu_id]
                    gpu_cache = GPUCache(len(local_cache_nodes), part_policy.get_size(), embed_size, torch.float32, dev_id)
                    gpu_cache.init_cache(local_cache_nodes, g, ntype, init_func=init_emb)
                    print(f"Rank {dist.get_rank()} init cache for {ntype}")

                emb = dgl.distributed.DistEmbedding(
                    part_policy.get_size(),
                    self.embed_size,
                    embed_name + "_" + ntype,
                    init_emb,
                    part_policy,
                    gpu_cache=gpu_cache
                )
                self.node_embeds[ntype] = emb
                print(f"Rank {dist.get_rank()} create DistEmbedding for {ntype}")
                if gpu_cache is not None:
                    self._gpu_caches[emb.name] = gpu_cache
        else:
            self.node_embeds = nn.ModuleDict()
            for ntype in sorted(ntypes_wo_feat):
                part_policy = None
                if partition_book is not None:
                    part_policy = NodePartitionPolicy(partition_book, ntype=ntype)
                else:
                    part_policy = g.get_node_partition_policy(ntype) 
                part_size = part_policy.get_part_size()

                print(f"Rank {dist.get_rank()} part_size for {ntype}: {part_size}")
                self.node_embeds[ntype] = nn.Embedding(
                    part_policy.get_size(), embed_size, sparse=True
                )
                nn.init.uniform_(self.node_embeds[ntype].weight, -1.0, 1.0)
    
    def broadcast(self):
        for ntype in sorted(self.node_projs.keys()):
            print(f"broadcast node_projs for {ntype}")
            _broadcast_layers([self.node_projs[ntype]], src=0)
        
    @property
    def cache_hit_rate(self):
        return {
            ntype: f"{np.mean(self._cache_hit_rate[ntype]):.4f}" for ntype in self._cache_hit_rate if len(self._cache_hit_rate[ntype]) > 0 
        }
    
    @property
    def gpu_caches(self):
        return self._gpu_caches

    def _fetch_feat(self, ntype, node_ids):
        """Fetch features
        """
        if ntype in self.feat:
            return torch.from_numpy(self.feat[ntype][node_ids])
        else:
            return self.g.nodes[ntype].data[self.feat_name][node_ids]

    def forward(self, node_ids):
        """Forward computation
        Parameters
        ----------
        node_ids : dict of Tensor
            node ids to generate embedding for.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = {}
        self._fetch_feat_time = 0
        self._proj_time = 0
        for ntype in node_ids:
            start = time.time()
            if ntype in self.ntypes_w_feat:
                if ntype in self._gpu_caches:
                    idx = node_ids[ntype]
                    gpu_cache = self._gpu_caches[ntype]
                    cached_feat, cache_mask = gpu_cache.get(idx)
                    self._cache_hit_rate[ntype].append(gpu_cache.cache_hit_rate)
                    cached_feat = cached_feat.type(torch.float32)
                    uncached_mask = ~cache_mask
                    uncached_idx = idx[uncached_mask].to('cpu')

                    uncached_values = (self._fetch_feat(ntype, uncached_idx)
                                       .type(torch.float32)
                                       .to(self.dev_id))
                    feat = torch.empty((idx.shape[0], gpu_cache.dim), dtype=torch.float32, device=self.dev_id)
                    feat[cache_mask] = cached_feat
                    feat[uncached_mask] = uncached_values
                else:
                    feat = (self._fetch_feat(ntype, node_ids[ntype].cpu())
                            .type(torch.float32)
                            .to(self.dev_id))
                self._fetch_feat_time += time.time() - start

                start = time.time()
                embeds[ntype] = self.node_projs[ntype](feat)
                self._proj_time += time.time() - start
            elif self.dgl_sparse_emb:
                embeds[ntype] = self.node_embeds[ntype](node_ids[ntype], self.dev_id)
                if self.node_embeds[ntype].gpu_cache is not None:
                    cache_hit_rate = self.node_embeds[ntype].gpu_cache.cache_hit_rate
                    self._cache_hit_rate[ntype].append(cache_hit_rate)
                self._fetch_feat_time += time.time() - start
            else:
                embeds[ntype] = self.node_embeds[ntype](node_ids[ntype]).to(self.dev_id)
        return embeds




class RGCN(nn.Module):
    """use dglnn.HeteroGraphConv"""

    def __init__(self, etypes: List[str], predict_category: str, in_size: int, hid_size: int,
                 out_size: int, n_layers: int = 3, dropout: float = 0.5, dist: bool = False, process_group=None):
        super().__init__()
        self.predict_category = predict_category
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = nn.ModuleList()
        print(f"created RGCN with etypes: {etypes}")
        for layer in range(n_layers):
            if layer == 0:
                layer_in_size = in_size
            else:
                layer_in_size = hid_size
            
            if layer == n_layers - 1:
                layer_out_size = out_size
            else:
                layer_out_size = hid_size

            self.layers.append(HeteroGraphConv({
                etype: GraphConv(layer_in_size, layer_out_size, norm='right') for etype in etypes
            }, aggregate='sum'))
        self.self_loop_layer = nn.Linear(layer_in_size, layer_out_size)
        self.dropout = nn.Dropout(dropout)
        self.dist = dist
        self.process_group = process_group
    
    def broadcast(self):
        _broadcast_layers([self.self_loop_layer], src=0)
    
    def forward(self, blocks: List[dgl.DGLHeteroGraph], inputs: Dict[str, torch.Tensor]):
        h = inputs
        num_seeds = blocks[-1].num_dst_nodes(self.predict_category)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            input_dst = {
                k: v[:block.number_of_dst_nodes(k)] for k, v in h.items()
            }
            
            h = layer(block, h)
            if self.dist:
                dist.all_reduce(h[self.predict_category][:num_seeds], op=dist.ReduceOp.SUM, group=self.process_group)
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        input_dst = input_dst[self.predict_category][:num_seeds]
        if self.dist:
            dist.all_reduce(input_dst, op=dist.ReduceOp.SUM, group=self.process_group)
            input_dst /= dist.get_world_size(group=self.process_group)
        h[self.predict_category] += self.self_loop_layer(input_dst)
        return h[self.predict_category]


class RGAT(nn.Module):
    """use dglnn.HeteroGraphConv"""

    def __init__(self, etypes: List[str], predict_category: str, in_size: int, hid_size: int,
                 out_size: int, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.5, dist: bool = False, process_group=None):
        super().__init__()
        self.predict_category = predict_category
        self.layers = nn.ModuleList()
        for layer in range(n_layers):
            if layer == 0:
                layer_in_size = in_size
            else:
                layer_in_size = hid_size

            self.layers.append(HeteroGraphConv({
                etype: GATConv(layer_in_size, hid_size // n_heads, n_heads) for etype in etypes
            }, aggregate='sum'))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_size, out_size)
        self.dist = dist
        self.process_group = process_group
    
    def broadcast(self):
        _broadcast_layers([self.fc], src=0)
    

    def forward(self, blocks: List[dgl.DGLHeteroGraph], inputs: Dict[str, torch.Tensor]):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )

            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        if not self.dist:
            return self.fc(h[self.predict_category])
        else:
            dist.all_reduce(h[self.predict_category], op=dist.ReduceOp.SUM, group=self.process_group)
            return self.fc(h[self.predict_category])



class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout = 0.2, use_norm = False, 
                 predict_category = None, dist: bool =False, process_group=None):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

        self.predict_category = predict_category
        self.dist = dist
        self.process_group = process_group


    def forward(self, G: dgl.DGLGraph, h: Dict[str, torch.Tensor]):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                # skip empty 
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]] 
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype][:G.num_dst_nodes(dsttype)]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype : (
                            fn.u_mul_e("v_%d" % self.edge_dict[etype], "t", "m"),
                            fn.sum("m", "t"),
                    )
                    for etype in G.etypes
                }, 
                cross_reducer = 'mean'
            )


            new_h = {}
            for ntype in G.ntypes:
                if isinstance(G.dstdata['t'], dict) and ntype not in G.dstdata['t']:
                    new_h[ntype] = h[ntype][:G.num_dst_nodes(ntype)]
                    continue
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                if isinstance(G.dstdata['t'], dict):
                    t = G.dstdata['t'][ntype].view(-1, self.out_dim)
                else:
                    t = G.dstdata['t'].view(-1, self.out_dim)

                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype][:G.num_dst_nodes(ntype)] * (1-alpha)

                if self.dist and ntype == self.predict_category:
                    dist.all_reduce(trans_out, op=dist.ReduceOp.SUM, group=self.process_group)
                    trans_out /= dist.get_world_size(group=self.process_group)

                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)



class HGT(nn.Module):
    def __init__(
        self,
        node_dict,
        edge_dict,
        predict_category,
        in_feats,
        num_hidden,
        n_classes,
        n_layers=3,
        n_heads=4,
        use_norm=True,
        dist: bool = False,
        process_group=None,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.predict_category = predict_category
        self.gcs = nn.ModuleList()
        self.n_inp = in_feats
        self.n_hid = num_hidden
        self.n_out = n_classes
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        self.use_norm = use_norm

        for i in range(n_layers):
            if i == n_layers - 1:
                self.gcs.append(
                    HGTLayer(
                        num_hidden,
                        num_hidden,
                        node_dict,
                        edge_dict,
                        n_heads,
                        use_norm=use_norm,
                        predict_category=predict_category,
                        dist=dist,
                        process_group=process_group
                    )
                )
            else:
                self.gcs.append(
                    HGTLayer(
                        num_hidden,
                        num_hidden,
                        node_dict,
                        edge_dict,
                        n_heads,
                        use_norm=use_norm,
                    )
                )

        self.fc = nn.Linear(num_hidden, n_classes)
        self.dist = dist
        self.process_group = process_group

    def broadcast(self):
        predict_category_id = self.node_dict[self.predict_category]
        layers_to_broadcast = [self.fc, self.gcs[-1].skip, self.gcs[-1].a_linears[predict_category_id]]
        if self.use_norm:
            layers_to_broadcast.append(self.gcs[-1].norms[predict_category_id])
        _broadcast_layers(layers_to_broadcast, src=0)

    def forward(self, blocks, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.gcs, blocks)):
            h = layer(block, h)
         
        return self.fc(h[self.predict_category])


def get_model(model_name: str, g: dgl.DGLGraph, predict_category: str, 
              in_feats: int, num_hidden: int,  n_classes: int,
              num_layers: int, dist: bool = False, process_group=None):
    """get model 
    """
    if model_name == "rgcn":
        model = RGCN(
            g.etypes, 
            predict_category,
            in_feats,
            num_hidden,
            n_classes,
            num_layers,
            dist=dist,
            process_group=process_group
        )
    elif model_name == "rgat":
        model = RGAT(
            g.etypes,
            predict_category,
            in_feats, 
            num_hidden, 
            n_classes,
            num_layers,
            dist=dist,
            process_group=process_group
        )
    elif model_name == "hgt":
        node_dict = {ntype: i for i, ntype in enumerate(g.ntypes)}
        edge_dict = {etype: i for i, etype in enumerate(g.etypes)}
        model = HGT(
            node_dict,
            edge_dict,
            predict_category,
            in_feats,
            num_hidden,
            n_classes,
            num_layers,
            use_norm=True,
            dist=dist,
            process_group=process_group
        )
    else:
        raise ValueError(f"model_name {model_name} not supported")
    
    return model

if __name__ == "__main__":
    # debug HGT use ogbn-mag 
    from load_graph import load_dataset
    from metatree import MetaTree

    dataset = "ogbn-mag"
    g, n_classes, predict_category = load_dataset(dataset)
    print("num_classes: ", n_classes)
    metatree = MetaTree(g, predict_category)
    rels = metatree.tree.rels
    print("rels: ", rels)
    # only keep these rels
    g = dgl.edge_type_subgraph(g, rels)


    node_dict = {ntype: i for i, ntype in enumerate(g.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(g.etypes)}
    print(node_dict, edge_dict)

    model = HGT(
        node_dict,
        edge_dict,
        predict_category,
        in_feats=128,
        num_hidden=128,
        n_classes=n_classes,
        n_layers=2,
        n_heads=4,
        use_norm=True,
    )
    train_mask = g.nodes[predict_category].data['train_mask']
    nodes = g.nodes(predict_category)
    train_nids = {predict_category: nodes[train_mask]}

    sampler = dgl.dataloading.NeighborSampler(
        [25, 10]
    )
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nids,
        sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False
    )
    
    for input_nodes, seeds, blocks in dataloader:
        batch_inputs = (
            {ntype: g.nodes[ntype].data["feat"][input_nodes[ntype]] for ntype in g.ntypes}
        )
        batch_labels = g.nodes[predict_category].data["label"][seeds[predict_category]]
    
        logits = model(blocks, batch_inputs)
        break