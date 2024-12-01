import argparse
import os
import socket
import sys
import time
from contextlib import contextmanager
from typing import List
from datetime import timedelta

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl.distributed import load_partition_book

from dist_context import initialize
from model import DistEmbedLayer, get_model
from share_mem_utils import copy_graph_to_shared_mem, get_graph_from_shared_mem

parser = argparse.ArgumentParser(description="GCN")
parser.add_argument("--graph_name", type=str, help="graph name")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--id", type=int, help="the partition id")
parser.add_argument(
    "--ip_config", type=str, help="The file for IP configuration"
)
parser.add_argument(
    "--part_dir", type=str, help="The path to the partition"
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
    "--ntypes-w-feats", type=str, nargs="*", default=[],
    help="Node types with features"
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

def create_all_local_groups():
    """create all local groups."""
    ranks = list(range(dist.get_world_size()))
    local_world_size = args.local_world_size
    local_groups = []
    for i in range(len(ranks) // local_world_size):
        local_ranks = ranks[i * local_world_size : (i + 1) * local_world_size]
        local_groups.append(dist.new_group(local_ranks, timeout=timedelta(seconds=18000)))
    
    return local_groups

def create_all_mirror_groups():
    """create all mirrored groups (same local rank in all machines)."""
    ranks = list(range(dist.get_world_size()))
    local_world_size = args.local_world_size
    mirror_groups = []
    for i in range(local_world_size):
        mirror_ranks = ranks[i::local_world_size]
        mirror_groups.append(dist.new_group(mirror_ranks, timeout=timedelta(seconds=18000)))

    return mirror_groups

def load_partition(part_id: int, graph_name: str, local_group: dist.ProcessGroup) -> dgl.DGLGraph:
    local_root_rank = args.machine_rank * args.local_world_size
    if args.local_rank == 0:
        assert local_root_rank == args.rank
        g = dgl.load_graphs(os.path.join(args.part_dir, f"part{part_id}/graph.dgl"))[0][0]
        new_g = copy_graph_to_shared_mem(g, graph_name, local_root_rank, local_group)
    else:
        new_g = get_graph_from_shared_mem(graph_name, local_root_rank, local_group)
    
    print(f"Rank {args.rank}: loaded graph")
    dist.barrier()
    return new_g


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
            if not isinstance(blocks[-1].dstdata[dgl.NID], dict):
                ntype = g.ntypes[0]
                input_nodes = {ntype: input_nodes.cpu()}
            batch_inputs = embed_layer(input_nodes)
            blocks = [block.to(device) for block in blocks]
            batch_labels = g.nodes[args.predict_category].data["label"][seeds[args.predict_category]].type(th.LongTensor).to(device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            all_preds.append(batch_pred)
            all_labels.append(batch_labels)

    all_preds = th.cat(all_preds, dim=0)
    all_labels = th.cat(all_labels, dim=0)
    model.train()
    return compute_acc(all_preds, all_labels)


def run(args, device, data, local_group: dist.ProcessGroup, mirror_group: dist.ProcessGroup):
    # Unpack data
    train_nid, test_nid, n_classes, g = data
    shuffle = True

    if args.no_sampling:
        # NB: this is not the best practice to disable sampling
        fanouts = [-1, -1] 
    else:
        fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    print("fanouts:", fanouts)
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    test_dataloader = dgl.dataloading.DataLoader(
        g,
        test_nid,
        sampler,
        batch_size=args.batch_size_eval,
        shuffle=False,
        drop_last=True,
    )

    pb = load_partition_book(os.path.join(args.part_dir, args.graph_name + ".json"), args.machine_rank)[0]
    print(f"args.ntypes_w_feats: {args.ntypes_w_feats}")
    args.ntypes_w_feats = args.ntypes_w_feats[0].split(",") if len(args.ntypes_w_feats) > 0 else []
    embed_layer = DistEmbedLayer(
        device,
        g,
        args.num_hidden,
        args.ntypes_w_feats,
        args.graph_name,
        dgl_sparse_emb=args.dgl_sparse,
        feat_name="feat",
        partition_book=pb,
        predict_category=args.predict_category,
        cache_method=args.cache_method,
        args=args
    )
    # embed_layer.broadcast()

    # Define model and optimizer
    model = get_model(args.model, g, args.predict_category, args.num_hidden, args.num_hidden, n_classes, 
                      args.num_layers, dist=True, process_group=mirror_group)
    model = model.to(device)

    model.broadcast()

    if args.num_gpus == -1:
         model = th.nn.parallel.DistributedDataParallel(
            model,
            # NB: the part of the model is distributed in GPUs of the local machine 
            process_group=local_group, 
            find_unused_parameters=True,
        )
    else:
        model = th.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # NB: the part of the model is distributed in GPUs of the local machine 
            process_group=local_group, 
            find_unused_parameters=True,
        )
    print(f"Rank {args.rank}: created model")

    # If there are dense parameters in the embedding layer
    # or we use Pytorch saprse embeddings.
    if len(embed_layer.node_projs) > 0 or not args.dgl_sparse:
        print(f"Rank {args.rank}: before embed_layer DDP")
        embed_layer = embed_layer.to(device)
        if args.num_gpus == -1:
            embed_layer = th.nn.parallel.DistributedDataParallel(
                embed_layer, 
                find_unused_parameters=True, process_group=local_group if args.graph_name in ["igb-het", "donor"] else None
            )
        else:
            embed_layer = th.nn.parallel.DistributedDataParallel(
                embed_layer, device_ids=[device], output_device=device, 
                find_unused_parameters=True, process_group=local_group if args.graph_name in ["igb-het", "donor"] else None
            )
        print(f"Rank {args.rank}: after embed_layer DDP")

    if isinstance(embed_layer, nn.parallel.DistributedDataParallel):
        embed_layer_module = embed_layer.module
    else:
        embed_layer_module = embed_layer

    if args.dgl_sparse and len(embed_layer_module.node_embeds.keys()) > 0:
        emb_optimizer = dgl.distributed.optim.SparseAdam(
            list(embed_layer_module.node_embeds.values()), lr=args.sparse_lr
        )
        print(f"Rank {args.rank} optimize DGL sparse embedding: {embed_layer_module.node_embeds.keys()}")
        all_params = list(model.parameters()) + list(embed_layer_module.node_projs.parameters())
    else:
        emb_optimizer = None
        all_params = list(model.parameters()) + list(embed_layer_module.node_projs.parameters())

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    optimizer = optim.Adam(all_params, lr=args.lr)

    # Training loop
    iter_tput = []
    epoch = 0
    print(f"Rank {args.rank}: start training")
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        feat_copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        emb_update_time = 0

        emb_update_breakdown = th.zeros(5, dtype=th.float32)
        emb_update_tot_time = th.zeros(1, dtype=th.float32)

        tot_num_seeds = 0
        tot_num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()
            sample_time += tic_step - start

            if isinstance(blocks[-1].dstdata[dgl.NID], dict):
                num_seeds = len(blocks[-1].dstdata[dgl.NID][args.predict_category])
                num_inputs = sum([len(v) for _, v in blocks[0].srcdata[dgl.NID].items()])
            else:
                # to dict of node types
                ntype = g.ntypes[0]
                input_nodes = {ntype: input_nodes.cpu()}
                num_seeds = len(blocks[-1].dstdata[dgl.NID])
                num_inputs = len(blocks[0].srcdata[dgl.NID])
            tot_num_seeds += num_seeds
            tot_num_inputs += num_inputs

            # move to target device
            # fetch features/labels
            batch_inputs = embed_layer(input_nodes)
            start = time.time()
            blocks = [block.to(device) for block in blocks]
            batch_labels = g.nodes[args.predict_category].data["label"][seeds[args.predict_category]].type(th.LongTensor).to(device)
            feat_copy_time += time.time() - start + embed_layer_module._fetch_feat_time

            # Compute loss and prediction
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            forward_end = time.time()

            optimizer.zero_grad()
            if args.dgl_sparse and emb_optimizer is not None:
                emb_optimizer.zero_grad()

            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - start
            backward_time += compute_end - forward_end

            optimizer.step()
            update_time += time.time() - compute_end
            update_end = time.time()
            if args.dgl_sparse and emb_optimizer is not None:
                emb_optimizer.step(dist=True, group=local_group)
                emb_update_breakdown[0] += emb_optimizer._send_grad_time
                emb_update_breakdown[1] += emb_optimizer._pull_time
                emb_update_breakdown[2] += emb_optimizer._push_time
                emb_update_breakdown[3] += emb_optimizer._h2d_d2h_time
                emb_update_breakdown[4] += emb_optimizer._comp_time
                emb_update_tot_time[0] += emb_optimizer._tot_time
            elif emb_optimizer is not None:
                emb_optimizer.step()
            emb_update_time += time.time() - update_end

            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(num_seeds / step_t)
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                tput = np.mean(iter_tput[3:])
                step_t = np.mean(step_time[-args.log_every :])

                stats = th.tensor(
                    [loss.item(), acc.item(), tput, step_t]).to(device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats = stats / dist.get_world_size()
                loss, acc, tput, step_t = stats[0].item(), stats[1].item(), stats[2].item(), stats[3].item()

                if args.rank == 0:
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} MB | time {:.3f} s".format(
                            args.rank,
                            epoch,
                            step,
                            loss,
                            acc,
                            tput,
                            gpu_mem_alloc,
                            step_t
                        )
                    )
            start = time.time()

        toc = time.time()
        tot_time = toc - tic

        all_time_tensor = th.tensor(
            [tot_time, sample_time, feat_copy_time, forward_time, backward_time, update_time, emb_update_time],
            device=device,
        )
        dist.all_reduce(all_time_tensor, op=dist.ReduceOp.SUM)
        all_time_tensor /= dist.get_world_size()
        tot_time, sample_time, feat_copy_time, forward_time, backward_time, update_time, emb_update_time = all_time_tensor.tolist()

        if args.rank == 0: 
            print(
                "Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, feat_copy: {:.4f}, "
                "forward: {:.4f}, backward: {:.4f}, model update: {:.4f}, emb update: {:.4f} #seeds: {}, "
                "#inputs: {}".format(
                    args.rank,
                    tot_time,
                    sample_time,
                    feat_copy_time,
                    forward_time,
                    backward_time,
                    update_time,
                    emb_update_time,
                    tot_num_seeds,
                    tot_num_inputs,
                )
            )

        print(f"Part {args.rank}, gpu cache hit rate: {embed_layer_module.cache_hit_rate}")
        # synchronize emb breakdown
        if args.dgl_sparse and emb_optimizer is not None:
            # dist.all_reduce(emb_update_breakdown, op=dist.ReduceOp.SUM)
            # emb_update_breakdown = emb_update_breakdown / dist.get_world_size()
            send_grad_time, pull_time, push_time, h2d_d2h_time, comp_time = emb_update_breakdown.tolist()
            print("Part {}, send_grad: {:.4f}, pull: {:.4f}, push: {:.4f}, h2d_d2h: {:.4f}, comp: {:.4f}".format(
                args.rank, send_grad_time, pull_time, push_time, h2d_d2h_time, comp_time
            ))

            all_emb_update_tot_time = [th.zeros(1, dtype=th.float32) for _ in range(dist.get_world_size())]
            dist.all_gather(all_emb_update_tot_time, emb_update_tot_time)
            all_emb_update_tot_time = [f"{t.item():.4f}" for t in all_emb_update_tot_time]

            print(f"Part {args.rank}, emb_update_cnt: {emb_optimizer.update_cnt}")
            if args.rank == 0:
                print("emb_update_tot_time: {}".format(all_emb_update_tot_time))
        epoch += 1

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            test_acc = evaluate(model, embed_layer, g, test_dataloader, device)
            # allreduce
            accs = th.tensor([test_acc, ]).to(device)
            dist.all_reduce(accs, op=dist.ReduceOp.SUM)
            accs = accs / dist.get_world_size()
            test_acc = accs[0].item()
            
            if args.rank == 0:
                print(
                    "Test Acc {:.4f}, time: {:.4f}".format(
                        test_acc, time.time() - start
                    )
                )


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    initialize(args.ip_config)

    dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=18000))
    print(f"get world size {dist.get_world_size()}")

    args.rank = dist.get_rank()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    args.machine_rank = int(os.environ["GROUP_RANK"])
    print(f"rank {args.rank}, local rank {args.local_rank}, machine rank {args.machine_rank}")

    set_seed(args.seed + args.local_rank)

    local_groups = create_all_local_groups()
    local_group = local_groups[args.machine_rank]
    mirror_groups = create_all_mirror_groups()
    mirror_group = mirror_groups[args.local_rank]

    g = load_partition(args.machine_rank, args.graph_name, local_group=local_group)

    train_mask = g.nodes[args.predict_category].data['train_mask']
    nodes = g.nodes(args.predict_category)
    train_nid = {args.predict_category: nodes[train_mask][args.local_rank::args.local_world_size]}
    if args.graph_name == "mag240m":
        # NB: MAG240M's test set does not have labels
        test_mask = g.nodes[args.predict_category].data['val_mask']
    else:
        test_mask = g.nodes[args.predict_category].data['test_mask']
    test_nid = {args.predict_category: nodes[test_mask][args.local_rank::args.local_world_size]}

    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = args.rank % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        print("rank", args.rank, "device", device)
    n_classes = args.n_classes

    data = train_nid, test_nid, n_classes, g
    run(args, device, data, local_group, mirror_group)
    print("parent ends")

if __name__=="__main__":
    print(args)
    main(args)