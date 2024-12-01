import argparse
import os
import numpy as np
import time
from collections import defaultdict

import torch as th
from tqdm import tqdm

import dgl
from load_graph import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-mag")
args = parser.parse_args()

time_per_ntype = defaultdict(float)
num_nodes_per_ntype = defaultdict(int)
read_time = 0
write_time = 0
embed_ntypes = set()
embed_dim = 64

budget = 4 * 1024 * 1024 * 1024 # 4GB

def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.

    Profile each node type time and number of nodes.
    """
    global read_time, write_time
    for ntype in g.ntypes:
        if ntype in input_nodes:
            th.cuda.synchronize()
            t0 = time.perf_counter()
            g.nodes[ntype].data["feat"][input_nodes[ntype]].to(device)
            th.cuda.synchronize()
            read_time += time.perf_counter() - t0
            time_per_ntype[ntype] += time.perf_counter() - t0
            num_nodes_per_ntype[ntype] += len(input_nodes[ntype])

            if ntype in embed_ntypes:
                # simulate read optimizer states
                th.cuda.synchronize()
                t0 = time.perf_counter()
                g.nodes[ntype].data["os1"][input_nodes[ntype]].to(device)
                g.nodes[ntype].data["os2"][input_nodes[ntype]].to(device)
                th.cuda.synchronize()
                read_time += time.perf_counter() - t0
                time_per_ntype[ntype] += time.perf_counter() - t0

                # simulate write
                new_feat = th.randn(
                    len(input_nodes[ntype]), embed_dim, device=device)
                new_os1 = th.randn(
                    len(input_nodes[ntype]), embed_dim, device=device)
                new_os2 = th.randn(
                    len(input_nodes[ntype]), embed_dim, device=device)
                th.cuda.synchronize()
                t0 = time.perf_counter()
                g.nodes[ntype].data["feat"][input_nodes[ntype]] = new_feat.cpu()
                g.nodes[ntype].data["os1"][input_nodes[ntype]] = new_os1.cpu()
                g.nodes[ntype].data["os2"][input_nodes[ntype]] = new_os2.cpu()
                th.cuda.synchronize()
                write_time += time.perf_counter() - t0
                time_per_ntype[ntype] += time.perf_counter() - t0


def save_cached_node(budget, ntype2count, cache_ratio_dict, part, method, label):
    # how many nodes can be cached
    dtype_size = 2 if args.dataset == 'MAG240M' else 4 
    for ntype, ratio in cache_ratio_dict.items():
        feat_size = g.nodes[ntype].data["feat"].shape[1] * dtype_size
        feat_size = feat_size * 3 if ntype in embed_ntypes else feat_size # embedding + optimizer states
        num_cached_nodes = int(ratio* budget / feat_size)
        count = ntype2count[ntype]
        # cache the hottest nodes
        count = count.numpy()
        # get the cached nodes
        cached_nodes = np.argsort(count)[::-1][:num_cached_nodes]
        print(f"cache {len(cached_nodes)} {ntype} nodes")
        print(f"theoretical hit rate: {np.sum(count[cached_nodes]) / np.sum(count)}")
        dir = f"cache/{label}/{args.dataset}_{method}/{part}"
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, f'{ntype}.npy'), cached_nodes)

if __name__ == "__main__":
    start = time.time()
    g, n_classes, target_node_type, _, reverse_edge_type_prefix = load_dataset(
        args.dataset, load_feat=True)
    load_time = time.time() - start
    print(f"Load {args.dataset} time: {load_time:.2f}s")

    # if does not have features, create a embedding layer
    for ntype in g.ntypes:
        if "feat" not in g.nodes[ntype].data:
            g.nodes[ntype].data["feat"] = th.randn(
                g.num_nodes(ntype), embed_dim)
            g.nodes[ntype].data["os1"] = th.randn(
                g.num_nodes(ntype), embed_dim)
            g.nodes[ntype].data["os2"] = th.randn(
                g.num_nodes(ntype), embed_dim)
            embed_ntypes.add(ntype)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ntype2count_tensor = {ntype: th.zeros(g.number_of_nodes(ntype), dtype=th.int64, device=device) for ntype in g.ntypes}

    train_nid = {target_node_type: g.nodes[target_node_type].data["train_mask"].nonzero(
        as_tuple=True)[0]}
    sampler = dgl.dataloading.NeighborSampler([25, 20])
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=2048 * 8,
        shuffle=True,
        drop_last=True,
        use_uva=False
    )

    num_batches = len(dataloader)
    print("Number of batches: ", num_batches)
    count_time = 0
    start = time.time()
    for input_nodes, seeds, blocks in tqdm(dataloader):
        count_start = time.time()
        for ntype, nid in input_nodes.items():
            ntype2count_tensor[ntype][nid] += 1
        count_time += time.time() - count_start

        load_subtensor(g, seeds, input_nodes,
                       th.device("cuda"), load_feat=True)
  
    tot_time = time.time() - start
    sample_time = tot_time - read_time - write_time - count_time
    # tot_time += load_time
    print(
        f"Total time: {tot_time:.2f}s, read time: {read_time:.2f}s, write time: {write_time:.2f}s, sample time: {sample_time:.2f}s")

    shape_dict = {}
    for ntype in g.ntypes:
        shape_dict[ntype] = g.nodes[ntype].data["feat"].shape

    miss_penalty_ratio = defaultdict(float)
    for ntype in g.ntypes:
        if ntype in embed_ntypes:
            miss_penalty_ratio[ntype] = time_per_ntype[ntype] / \
                num_nodes_per_ntype[ntype] / shape_dict[ntype][1] / \
                4 / 3  # embedding + optimizer states
        else:
            miss_penalty_ratio[ntype] = time_per_ntype[ntype] / \
                num_nodes_per_ntype[ntype] / shape_dict[ntype][1] / 4

    if device == th.device("cuda"):
        ntype2count_tensor_cpu = {ntype: count_tensor.cpu() for ntype, count_tensor in ntype2count_tensor.items()}
    else:
        ntype2count_tensor_cpu = ntype2count_tensor

    print("Time per node type: ", time_per_ntype)
    print("Number of nodes per node type: ", num_nodes_per_ntype)
    # shape
    print("Shape of each node type: ", shape_dict)
    print("Miss penalty ratio: ", miss_penalty_ratio)

    ## calculate cache size ratio for each node type
    # ogbn-mag for 2 parts
    part_dict = {
        'part0': ['paper', 'author', 'field_of_study'],
        'part1': ['paper', 'institution'],
    }
    for part, ntypes in part_dict.items():
        cache_ratio_dict = {}
        denominator = sum([ntype2count_tensor_cpu[ntype].sum() * miss_penalty_ratio[ntype] for ntype in ntypes])
        for ntype in ntypes:
            cache_ratio_dict[ntype] = ntype2count_tensor_cpu[ntype].sum() * miss_penalty_ratio[ntype] / denominator
    
        print("Cache size ratio: ", cache_ratio_dict)

        save_cached_node(budget, ntype2count_tensor_cpu, cache_ratio_dict, part, "miss_penalty", "Heta")

