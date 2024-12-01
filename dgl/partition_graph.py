import argparse
import os
import sys
import time

import numpy as np
import torch as th

import dgl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from load_graph import load_dataset
from metatree import MetaTree

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-mag",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=2, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="partitions/dgl",
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--complete_missing_feats",
        action="store_true",
        help="Whether to complete the missing features."
    )
    args = argparser.parse_args()
    print(args)
    if args.num_parts > 2:
        args.output = os.path.join(args.output, args.dataset+"_"+args.part_method + "_"+str(args.num_parts))
    else:
        args.output = os.path.join(args.output, args.dataset+"_"+args.part_method)

    start = time.time()
    g, num_classes, target_node_type, list_of_metapaths, reverse_edge_type_prefix = load_dataset(args.dataset, complete_missing_feats=args.complete_missing_feats)
    print("num_classes: ", num_classes)
    metatree = MetaTree(g, target_node_type, list_of_metapaths, reverse_edge_type_prefix)
    rels = metatree.tree.rels
    print("rels: ", rels)
    # only keep these rels
    g = dgl.edge_type_subgraph(g, rels)
    g = g.formats(['coo'])

    print(
        "load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    if args.balance_train:
        balance_ntypes = {target_node_type: g.ndata['train_mask'][target_node_type]}
    else:
        balance_ntypes = None

    start = time.time()
    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )
    print("partitioning takes {:.3f} seconds".format(time.time() - start))