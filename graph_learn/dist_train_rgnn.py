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

import argparse, datetime
import os.path as osp
import time
import logging 

import graphlearn_torch as glt
import numpy as np
import sklearn.metrics
import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

from rgnn import RGNN


torch.manual_seed(42)


def evaluate(model, dataloader, predict_category):
  predictions = []
  labels = []
  with torch.no_grad():
    for batch in dataloader:
      batch_size = batch[predict_category].batch_size
      out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
      labels.append(batch[predict_category].y[:batch_size].cpu().numpy())
      predictions.append(out.argmax(1).cpu().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = sklearn.metrics.accuracy_score(labels, predictions)
    return acc

def run_training_proc(local_proc_rank, num_nodes, node_rank, num_training_procs,
    hidden_channels, num_classes, num_layers, model_type, num_heads, fan_out,
    epochs, batch_size, learning_rate, log_every,
    dataset, dataset_name, predict_category,
    train_idx, val_idx, test_idx,
    master_addr,
    training_pg_master_port,
    train_loader_master_port,
    val_loader_master_port,
    test_loader_master_port,
    with_gpu, edge_dir,
    rpc_timeout):

  # logging to both file and stdout
  # Create a file handler
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  file_handler = logging.FileHandler(f'graphlearn_{model_type}_{dataset_name}_{fan_out}.log', mode='w+')
  file_handler.setLevel(logging.INFO)

  # Create a console handler
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)

  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)

  logger = logging.getLogger(__file__)
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  logger.setLevel(logging.INFO)

  logger.info("Start training...")
  for handler in logger.handlers:
    handler.flush()

  # Initialize graphlearn_torch distributed worker group context.
  glt.distributed.init_worker_group(
    world_size=num_nodes*num_training_procs,
    rank=node_rank*num_training_procs+local_proc_rank,
    group_name='distributed-igbh-trainer'
  )

  current_ctx = glt.distributed.get_context()
  if with_gpu:
    current_device = torch.device(local_proc_rank % torch.cuda.device_count())
  else:
    current_device = torch.device('cpu')

  # Initialize training process group of PyTorch.
  torch.distributed.init_process_group(
    backend='gloo',
    rank=current_ctx.rank,
    world_size=current_ctx.world_size,
    init_method='tcp://{}:{}'.format(master_addr, training_pg_master_port)
  )

  # Create distributed neighbor loader for training
  train_idx = train_idx.split(train_idx.size(0) // num_training_procs)[local_proc_rank]
  train_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=(predict_category, train_idx),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    edge_dir=edge_dir,
    collect_features=True,
    to_device=torch.device('cpu'),
    worker_options = glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=[torch.device('cpu')],
      worker_concurrency=1,
      master_addr=master_addr,
      master_port=train_loader_master_port,
      channel_size='10GB',
      # pin_memory=True,
      rpc_timeout=rpc_timeout,
    )
  )
  # Create distributed neighbor loader for validation.
  val_idx = val_idx.split(val_idx.size(0) // num_training_procs)[local_proc_rank]
  val_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=(predict_category, val_idx),
    batch_size=batch_size,
    shuffle=False,
    edge_dir=edge_dir,
    collect_features=True,
    to_device=current_device,
    worker_options = glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=[torch.device('cpu')],
      worker_concurrency=1,
      master_addr=master_addr,
      master_port=val_loader_master_port,
      channel_size='10GB',
      # pin_memory=True,
      rpc_timeout=rpc_timeout,
    )
  )

  # Create distributed neighbor loader for testing.
  test_idx = test_idx.split(test_idx.size(0) // num_training_procs)[local_proc_rank]
  test_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=(predict_category, test_idx),
    batch_size=batch_size,
    shuffle=False,
    edge_dir=edge_dir,
    collect_features=True,
    to_device=current_device,
    worker_options = glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=[torch.device('cpu')],
      worker_concurrency=1,
      master_addr=master_addr,
      master_port=test_loader_master_port,
      channel_size='10GB',
      # pin_memory=True,
      rpc_timeout=rpc_timeout,
    )
  )

  # Define model and optimizer.
  if with_gpu:
    torch.cuda.set_device(current_device)
  ntypes = sorted(dataset.get_node_types())
  model = RGNN(ntypes,
               dataset.get_edge_types(),
               {ntype: dataset.node_features[ntype].shape[1] for ntype in ntypes},
               hidden_channels,
               num_classes,
               num_layers=num_layers,
               dropout=0.5,
               model=model_type,
               heads=num_heads,
               node_type=predict_category).to(current_device)
  model = DistributedDataParallel(model,
                                  device_ids=[current_device.index] if with_gpu else None,
                                  find_unused_parameters=True)

  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

  loss_fcn = torch.nn.CrossEntropyLoss().to(current_device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  best_accuracy = 0
  training_start = time.time()
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_acc = 0
    idx = 0
    gpu_mem_alloc = 0

    sample_feat_copy_time = 0
    forward_time = 0 
    backward_time = 0
    update_time = 0
    tot_num_seeds = 0

    step_time = []
    start = time.time()
    epoch_start = time.time()
    for batch in train_loader:
      idx += 1
      batch = batch.to(current_device)
      tic_step = time.time()
      sample_feat_copy_time += tic_step - start
      batch_size = batch[predict_category].batch_size
      tot_num_seeds += batch_size 

      out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
      y = batch[predict_category].y[:batch_size]
      loss = loss_fcn(out, y)
      forward_end = time.time()
      forward_time += forward_end - tic_step
      optimizer.zero_grad()
      loss.backward()
      backward_end = time.time()
      backward_time += backward_end - forward_end
      optimizer.step()
      update_time += time.time() - backward_end

      total_loss += loss.item()
      train_acc += sklearn.metrics.accuracy_score(y.cpu().numpy(),
          out.argmax(1).detach().cpu().numpy())*100
      gpu_mem_alloc += (
          torch.cuda.max_memory_allocated() / 1000000
          if with_gpu
          else 0
      )
      if idx  % log_every == 0:
        all_tensor = torch.tensor(
          [total_loss / idx, train_acc / idx],
          device=current_device
        )
        torch.distributed.all_reduce(all_tensor)
        all_tensor /= torch.distributed.get_world_size()
        loss, acc = all_tensor.tolist()

        if torch.distributed.get_rank() == 0:
          logger.info(
                "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                "{:.1f} MB | time {:.3f} s".format(
                    torch.distributed.get_rank(),
                    epoch,
                    idx,
                    loss,
                    acc,
                    tot_num_seeds * torch.distributed.get_world_size() / (time.time() - epoch_start),
                    gpu_mem_alloc / idx,
                    (time.time() - epoch_start) / idx
                )
            )
          for handler in logger.handlers:
            handler.flush()

      start = time.time()

    tot_time = time.time() - epoch_start

    train_acc /= idx
    gpu_mem_alloc /= idx
    if with_gpu:
      torch.cuda.synchronize()
      torch.distributed.barrier()

    model.eval()
    test_acc = evaluate(model, test_loader, predict_category).item()*100
    if best_accuracy < test_acc:
      best_accuracy = test_acc
    
    all_tensor = torch.tensor(
      [tot_time, sample_feat_copy_time, forward_time, backward_time, update_time, train_acc, test_acc, total_loss],
      device=current_device
    )
    torch.distributed.all_reduce(all_tensor)
    all_tensor /= torch.distributed.get_world_size()
    tot_time, sample_feat_copy_time, forward_time, backward_time, update_time, train_acc, test_acc, total_loss = all_tensor.tolist()

    if with_gpu:
      torch.cuda.synchronize()
      torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
      logger.info(
          "Rank{:02d} | Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Test Acc {:.2f} | Time {} | GPU {:.1f} MB |"
          "sample+feat_copy: {:.4f} | forward: {:.4f} | backward: {:.4f} | model update: {:.4f}".format(
              current_ctx.rank,
              epoch,
              total_loss,
              train_acc,
              test_acc,
              tot_time, 
              gpu_mem_alloc, 
              sample_feat_copy_time,
              forward_time,
              backward_time,
              update_time
          )
      )
      for handler in logger.handlers:
        handler.flush()

  # model.eval()
  # test_acc = evaluate(model, test_loader).item()*100
  # print("Rank {:02d} Test Acc {:.2f}%".format(current_ctx.rank, test_acc))
  # print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dataset', type=str, default='medium',
      # choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument('--num_classes', type=int, default=2983,
     help='number of classes')
  parser.add_argument('--in_memory', type=int, default=1,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  # Model
  parser.add_argument('--model', type=str, default='rgat',
                      choices=['rgat', 'rsage', 'rgcn', 'hgt'])
  # Model parameters
  parser.add_argument('--fan_out', type=str, default='25,20') 
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--hidden_channels', type=int, default=64)
  parser.add_argument('--learning_rate', type=int, default=1e-2)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--num_layers', type=int, default=2)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--log_every', type=int, default=20)
  # Distributed settings.
  parser.add_argument("--num_nodes", type=int, default=2,
      help="Number of distributed nodes.")
  parser.add_argument("--node_rank", type=int, default=0,
      help="The current node rank.")
  parser.add_argument("--num_training_procs", type=int, default=2,
      help="The number of traning processes per node.")
  parser.add_argument("--master_addr", type=str, default='localhost',
      help="The master address for RPC initialization.")
  parser.add_argument("--training_pg_master_port", type=int, default=12111,
      help="The port used for PyTorch's process group initialization across training processes.")
  parser.add_argument("--train_loader_master_port", type=int, default=12112,
      help="The port used for RPC initialization across all sampling workers of train loader.")
  parser.add_argument("--val_loader_master_port", type=int, default=12113,
      help="The port used for RPC initialization across all sampling workers of val loader.")
  parser.add_argument("--test_loader_master_port", type=int, default=12114,
      help="The port used for RPC initialization across all sampling workers of test loader.")
  parser.add_argument("--cpu_mode", action="store_true",
      help="Only use CPU for sampling and training, default is False.")
  parser.add_argument("--edge_dir", type=str, default='out',
      help="sampling direction, can be 'in' for 'by_dst' or 'out' for 'by_src' for partitions.")
  parser.add_argument("--rpc_timeout", type=int, default=180,
                      help="rpc timeout in seconds")
  args = parser.parse_args()
  # when set --cpu_mode or GPU is not available, use cpu only mode.
  args.with_gpu = (not args.cpu_mode) and torch.cuda.is_available()
  if args.with_gpu:
    assert(not args.num_training_procs > torch.cuda.device_count())
  
  print(f'with_gpu: {args.with_gpu}')

  print('--- Loading data partition ...\n')
  predict_category = 'paper' if args.dataset == 'medium' else 'Project' # donor 

  data_pidx = args.node_rank % args.num_nodes
  dataset = glt.distributed.DistDataset(edge_dir=args.edge_dir)
  dataset.load(
    root_dir=osp.join(args.path, f'{args.dataset}-partitions'),
    partition_idx=data_pidx,
    graph_mode='CPU', 
    feature_with_gpu=True,
    whole_node_label_file={predict_category: osp.join(args.path, f'{args.dataset}-label', 'label.pt')}
  )
  train_idx = torch.load(
    osp.join(args.path, f'{args.dataset}-train-partitions', f'partition{data_pidx}.pt')
  )
  val_idx = torch.load(
    osp.join(args.path, f'{args.dataset}-val-partitions', f'partition{data_pidx}.pt')
  )
  test_idx = torch.load(
    osp.join(args.path, f'{args.dataset}-test-partitions', f'partition{data_pidx}.pt')
  )
  train_idx.share_memory_()
  val_idx.share_memory_()
  test_idx.share_memory_()

  dataset_name = 'igb-het' if args.dataset == 'medium' else args.dataset

  print('--- Launching training processes ...\n')
  torch.multiprocessing.spawn(
    run_training_proc,
    args=(args.num_nodes, args.node_rank, args.num_training_procs,
          args.hidden_channels, args.num_classes, args.num_layers, args.model, args.num_heads, args.fan_out,
          args.epochs, args.batch_size, args.learning_rate, args.log_every,
          dataset, dataset_name, predict_category,
          train_idx, val_idx, test_idx,
          args.master_addr,
          args.training_pg_master_port,
          args.train_loader_master_port,
          args.val_loader_master_port,
          args.test_loader_master_port,
          args.with_gpu,
          args.edge_dir,
          args.rpc_timeout),
    nprocs=args.num_training_procs,
    join=True
  )
