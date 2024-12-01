import argparse
import os
import socket
import sys
import time

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import dgl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser(description="GCN")
parser.add_argument("--graph_name", type=str, help="graph name")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--id", type=int, help="the partition id")
parser.add_argument(
    "--ip_config", type=str, help="The file for IP configuration"
)
parser.add_argument(
    "--part_config", type=str, help="The path to the partition config file"
)
parser.add_argument(
    "--n_classes", type=int, default=0, help="the number of classes"
)
parser.add_argument(
    "--predict_category", type=str, help="predict category"
)
parser.add_argument(
    "--backend",
    type=str,
    default="gloo",
    help="pytorch distributed backend",
)
parser.add_argument(
    "--num_gpus",
    type=int,
    default=-1,
    help="the number of GPU device. Use -1 for CPU training",
)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--num_hidden", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--fan_out", type=str, default="25,25")
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--batch_size_eval", type=int, default=1000)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument(
    "--local_rank", type=int, help="get rank of the process"
)
parser.add_argument(
    "--standalone", action="store_true", help="run in the standalone mode"
)
parser.add_argument(
    "--pad-data",
    default=False,
    action="store_true",
    help="Pad train nid to the same length across machine, to ensure num "
    "of batches to be the same.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed used in the training and validation phase.",
)
parser.add_argument(
    "--dgl-sparse",
    action="store_true",
    help="Whether to use DGL sparse embedding",
)
parser.add_argument(
    "--sparse-lr", type=float, default=0.06, help="sparse lr rate"
)
parser.add_argument(
    "--cache-method", type=str, help='cache method'
)
parser.add_argument(
    "--no-sampling", action="store_true", help="no sampling"
)
args = parser.parse_args()


def set_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)


def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        {ntype: g.nodes[ntype].data["feat"][input_nodes[ntype]].to(
            device) for ntype in g.ntypes} if load_feat else None
    )
    batch_labels = g.nodes[args.predict_category].data["label"][seeds[args.predict_category]].to(
        device)
    return batch_inputs, batch_labels


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, embed_layer, g, dataloader, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        all_preds = []
        all_labels = []
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # fetch features/labels
            # move to target device
            blocks = [block.to(device) for block in blocks]
            batch_inputs = embed_layer(input_nodes)
            batch_labels = g.nodes[args.predict_category].data["label"][seeds[args.predict_category]].type(
                th.LongTensor).to(device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            all_preds.append(batch_pred)
            all_labels.append(batch_labels)

    all_preds = th.cat(all_preds, dim=0)
    all_labels = th.cat(all_labels, dim=0)
    model.train()
    return compute_acc(all_preds, all_labels)


def run(args, device, data):
    # Unpack data
    train_nid, n_classes, g = data
    shuffle = True

    if args.no_sampling:
        # NB: this is not the best practice to disable sampling
        fanouts = [-1, -1]
    else:
        fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    print("fanouts:", fanouts)
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    feat_name = 'feat'

    for epoch in range(args.num_epochs):

        sample_time = 0
        tot_num_seeds = 0
        tot_num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        total_num_layer0_nodes = 0
        total_num_remote_nodes = 0
        total_num_remote_feat_fetch_size = 0
        total_num_layer1_nodes = 0

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()
            sample_time += tic_step - start
            num_seeds = len(blocks[-1].dstdata[dgl.NID][args.predict_category])
            num_inputs = sum([len(v)
                             for _, v in blocks[0].srcdata[dgl.NID].items()])
            tot_num_seeds += num_seeds
            tot_num_inputs += num_inputs

            # move to target device
            # fetch features/labels
            # find the remote nodes
            pb = g.get_partition_book()
            num_remote_nodes = 0
            num_remote_feat_fetch_size = 0
            num_layer0_nodes = 0
            for ntype in g.ntypes:
                partid = pb.nid2partid(input_nodes[ntype], ntype)
                num_remote_nodes_type = (partid != pb.partid).sum().item()
                num_remote_nodes += num_remote_nodes_type
                if ntype == 'paper':
                    num_remote_feat_fetch_size += num_remote_nodes_type * 768 * 2
                else:
                    num_remote_feat_fetch_size += num_remote_nodes_type * 64 * 2

                num_layer0_nodes += len(input_nodes[ntype])

            total_num_layer0_nodes += num_layer0_nodes
            total_num_remote_nodes += num_remote_nodes
            total_num_remote_feat_fetch_size += num_remote_feat_fetch_size

            num_layer1_nodes = 0
            b = blocks[1]
            for ntype in g.ntypes:
                num_layer1_nodes += len(b.srcdata[dgl.NID][ntype])

            total_num_layer1_nodes += num_layer1_nodes

            print(f"Epoch {epoch} Step {step} Sample Time: {sample_time:.4f} #Layer0: {total_num_layer0_nodes} #Remote: {total_num_remote_nodes} #RemoteFeatFetchSize: {total_num_remote_feat_fetch_size} #Layer1: {total_num_layer1_nodes} #NumSeeds: {tot_num_seeds}")

            sys.exit(0)


def main(args):
    set_seed(args.seed)
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(f"get world size {dist.get_world_size()}")
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(
        args.graph_name, part_config=args.part_config)
    print(socket.gethostname(), "rank:", g.rank())
    args.rank = g.rank()

    pb = g.get_partition_book()
    if "trainer_id" in g.nodes[args.predict_category].data:
        trainer_id = g.nodes[args.predict_category].data["trainer_id"]
    else:
        trainer_id = None

    train_nid = {args.predict_category: dgl.distributed.node_split(
        g.nodes[args.predict_category].data["train_mask"], pb, ntype=args.predict_category, force_even=True, node_trainer_ids=trainer_id
    )}

    local_nid = pb.partid2nids(
        pb.partid, ntype=args.predict_category).detach().numpy()
    print(train_nid)
    print(local_nid)

    del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.nodes[args.predict_category].data["labels"][np.arange(
            g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    data = train_nid, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    print(args)
    main(args)
