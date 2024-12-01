import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import dgl
from dgl.partition import get_peak_mem
import networkx as nx
import pandas as pd

from metatree import MetaTree, Subtree, get_num_edges, get_num_nodes

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from load_graph import load_dataset
from gpu_cache import itemsize

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-mag")
parser.add_argument("--num_parts", type=int, default=2)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--out_dir", type=str, default="partitions/Heta")
args = parser.parse_args()


def get_graph_mem_size(graph: dgl.DGLGraph, target_node_type: str):
    """get graph memory size"""
    topo = (graph.number_of_nodes() + graph.number_of_edges()) * 8
    feat = 0
    for data in graph.nodes[target_node_type].data.items():
        feat += data[1].numel() * itemsize(data[1].dtype)
    return (topo + feat) / 1e9


def _balanced_k_partition(subtrees: List[Subtree], k: int) -> Dict:
    # sort the numbers in descending order
    subtrees.sort(reverse=True, key=lambda x: x.weight)
    
    # initialize k partitions
    partitions = {i:[] for i in range(k)}
    sums = [0]*k
    
    for subtree in subtrees:
        # find the partition with smallest sum
        min_sum_index = sums.index(min(sums))
        
        # add the current number to this partition
        partitions[min_sum_index].append(subtree)
        
        # add the current number to the sum of this partition
        sums[min_sum_index] += subtree.weight
    
    return partitions

def meta_partition(graph: dgl.DGLGraph, target_node_type: str, num_parts: int, out_path: str, 
                          list_of_metapaths: List[List[Tuple[str, str, str]]] = None, 
                          reverse_edge_type_prefix: str = "rev_", depth=2):
    """partition heterographs along relations 

    Args:
        graph (dgl.DGLGraph): the input graph 
        target_node_type (str): the target node type (root node type)
        num_parts (int): number of partitions
        out_path (str): output path
        list_of_metapaths (List[List[Tuple[str, str, str]]]): a list of metapaths (src ntype, etype, dst ntype)
        reverse_edge_type_prefix (str): the prefix of reverse edge type
        depth (int): the depth of the metatree
    """
    start = time.time()
    # partition graph
    metatree = MetaTree(graph, target_node_type, list_of_metapaths, reverse_edge_type_prefix, depth)
    subtrees = metatree.get_all_subtrees_from_root()

    if num_parts > len(subtrees):
        raise NotImplementedError(f'num_parts ({num_parts}) > subtrees ({len(subtrees)})')

    # partition graph
    partitions = _balanced_k_partition(subtrees, num_parts)
    part2num_rels = defaultdict(int)
    part2num_nodes = defaultdict(int)
    part2num_edges = defaultdict(int)
    part2leave_nodes = defaultdict(list)
    for part_id, subtrees in partitions.items():
        leave_nodes = []
        for subtree in subtrees:
            part2num_rels[part_id] += subtree.num_rels
            part2num_nodes[part_id] += get_num_nodes(subtree.rels, graph)
            part2num_edges[part_id] += get_num_edges(subtree.rels, graph)
            leave_nodes.extend(subtree.leave_nodes)
        leave_nodes = list(set(leave_nodes))
        part2leave_nodes[part_id] += leave_nodes

    # to pandas for easy visualization
    df = pd.DataFrame(columns=['partition', 'weight', 'num_rels', 'num_nodes', 'num_edges', 'rels'])
    df['partition'] = list(range(num_parts))
    df['weight'] = [sum([subtree.weight for subtree in partitions[part_id]]) for part_id in range(num_parts)]
    df['num_rels'] = [part2num_rels[part_id] for part_id in range(num_parts)]
    df['num_nodes'] = [part2num_nodes[part_id] for part_id in range(num_parts)]
    df['num_edges'] = [part2num_edges[part_id] for part_id in range(num_parts)]
    df['rels'] = [[subtree.rels for subtree in partitions[part_id]] for part_id in range(num_parts)]
    df['leave_nodes'] = [part2leave_nodes[part_id] for part_id in range(num_parts)]
    # df['weight'] /= df['weight'].sum()
    print(df)
    print(f"Peak memory usage: {get_peak_mem():.2f} GB")

    print("Partition Time: {:.4f}s".format(time.time()-start))
    start = time.time()

    sorted_ntypes = sorted(graph.ntypes)
    sorted_etypes = [f"{etype[0]}:{etype[1]}:{etype[2]}" for etype in sorted(graph.canonical_etypes)]
    ntypes = {
        ntype: i for i, ntype in enumerate(sorted_ntypes)
    }
    etypes = {
        etype: i for i, etype in enumerate(sorted_etypes)
    }
    # ntype: [[start, end], ...]
    node_map = {}
    for ntype in sorted_ntypes:
        node_map[ntype] = []
    # etype: [[start, end], ...]
    edge_map = {}
    for etype in sorted_etypes:
        edge_map[etype] = []
    # save 
    num_nodes = 0
    num_edges = 0
    allocated_ntypes = set()
    for part_id in range(num_parts):
        rels = []
        for subtree in partitions[part_id]:
            rels.extend(subtree.rels)

        g = dgl.edge_type_subgraph(graph, rels)

        for ntype in sorted_ntypes:
            start = num_nodes
            if ntype in part2leave_nodes[part_id] and ntype not in allocated_ntypes:
                end = start + g.number_of_nodes(ntype)
                allocated_ntypes.add(ntype)
            else:
                end = start
            node_map[ntype].append([start, end])
            num_nodes = end
        
        for etype in sorted_etypes:
            start = num_edges
            etype = tuple(etype.split(':'))
            if etype in g.canonical_etypes:
                end = start + g.number_of_edges(etype=etype)
            else:
                end = start
            etype = f"{etype[0]}:{etype[1]}:{etype[2]}"
            edge_map[etype].append([start, end])
            num_edges = end

        os.makedirs(os.path.join(out_path, f'part{part_id}'), exist_ok=True)
        dgl.save_graphs(os.path.join(out_path, f'part{part_id}/graph.dgl'), g)

    print(f"Peak memory usage: {get_peak_mem():.2f} GB")
    # generate {args.dataset}.json
    data = {
        "graph_name": args.dataset,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "part_method": "meta-partition",
        "num_parts": num_parts,
        "node_map": node_map,
        "edge_map": edge_map,
        "ntypes": ntypes,
        "etypes": etypes,
    }
    for part_id in range(num_parts):
        data[f"part-{part_id}"] = {
            "part_graph": f"part{part_id}/graph.dgl",
        }

    with open(os.path.join(out_path, f"{args.dataset}.json"), "w") as f:
        json.dump(data, f, indent=4)
 
    print(f"Peak memory usage: {get_peak_mem():.2f} GB")
    print("Saving Time: {:.4f}s".format(time.time()-start))

if __name__=="__main__":
    print(args)
    if args.num_parts > 2:
        args.out_dir = os.path.join(args.out_dir, args.dataset + f"_{args.num_parts}")
    else:
        args.out_dir = os.path.join(args.out_dir, args.dataset + f"_depth_{args.depth}")
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    g, num_classes, target_node_type, list_of_metapaths, reverse_edge_type_prefix = load_dataset(args.dataset)
    print("Load graph time: {:.4f}s".format(time.time()-start))
    print(f"Graph memory size: {get_graph_mem_size(g, target_node_type):.2f} GB")

    start = time.time()
    meta_partition(g, target_node_type, args.num_parts, args.out_dir, list_of_metapaths, reverse_edge_type_prefix, args.depth)
    print("Time: {:.4f}s".format(time.time()-start))
    