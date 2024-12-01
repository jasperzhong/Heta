"""share memory utils"""
import os
import pickle
import time
from typing import List

import dgl
import dgl.backend as F
import torch
import torch.distributed as dist
from dgl import heterograph_index
from dgl._ffi.ndarray import empty_shared_mem
from dgl.distributed.shared_mem_utils import (_get_edata_path, _get_ndata_path,
                                              _to_shared_mem)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]

def _get_shared_mem_ndata(g, graph_name, name, shape, dtype):
    """Get shared-memory node data 
    """
    data = empty_shared_mem(
        _get_ndata_path(graph_name, name), False, shape, _torch_dtype_to_str(dtype)
    )
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)


def _get_shared_mem_edata(g, graph_name, name, shape, dtype):
    """Get shared-memory edge data 
    """
    data = empty_shared_mem(
        _get_edata_path(graph_name, name), False, shape, _torch_dtype_to_str(dtype)
    )
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def _serialize_data(data):
    """Serialize the data into a ByteTensor"""
    pickled_data = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(pickled_data)
    tensor = torch.ByteTensor(storage)
    return tensor

def _deserialize_data(tensor):
    """Deserialize the ByteTensor back into the data"""
    pickled_data = tensor.numpy().tobytes()
    data = pickle.loads(pickled_data)
    return data

def copy_graph_to_shared_mem(g: dgl.DGLHeteroGraph, graph_name: str, rank: int,  local_group: dist.ProcessGroup):
    """Copy a graph to shared memory."""
    start = time.time()
    new_g = g.shared_memory(graph_name, formats="csc")
    print(f"copying graph to shared memory takes {time.time() - start:.3f} seconds")

    start = time.time()
    # Copy node features to shared memory
    for ntype in g.ntypes:
        for key in g.nodes[ntype].data.keys():
            feature = g.nodes[ntype].data[key]
            print(f"copying {ntype} {key} to shared memory")
            shared_mem_feature = _to_shared_mem(feature, _get_ndata_path(graph_name, ntype+"_"+key))
            print(f"copying {ntype} {key} to shared memory done")
            new_g.nodes[ntype].data[key] = shared_mem_feature
    print(f"copying node features to shared memory takes {time.time() - start:.3f} seconds")

    # Copy edge features to shared memory
    for etype in g.canonical_etypes:
        for key in g.edges[etype].data.keys():
            feature = g.edges[etype].data[key]
            shared_mem_feature = _to_shared_mem(feature, _get_edata_path(graph_name, '-'.join(etype)+"_"+key))
            new_g.edges[etype].data[key] = shared_mem_feature
        
    # broadcast ntypes/etypes and keys and tensor shape and dtype to all local workers
    broadcast_data = {
        "ndata_keys": {ntype: list(g.nodes[ntype].data.keys()) for ntype in g.ntypes},
        "edata_keys": {etype: list(g.edges[etype].data.keys()) for etype in g.canonical_etypes},
        "ndata_shapes": {ntype: {key: g.nodes[ntype].data[key].shape for key in g.nodes[ntype].data.keys()} for ntype in g.ntypes},
        "edata_shapes": {etype: {key: g.edges[etype].data[key].shape for key in g.edges[etype].data.keys()} for etype in g.canonical_etypes},
        "ndata_dtypes": {ntype: {key: g.nodes[ntype].data[key].dtype for key in g.nodes[ntype].data.keys()} for ntype in g.ntypes},
        "edata_dtypes": {etype: {key: g.edges[etype].data[key].dtype for key in g.edges[etype].data.keys()} for etype in g.canonical_etypes},
    }
    tensor = _serialize_data(broadcast_data)
    dist.broadcast_object_list([tensor], src=rank, group=local_group)
    return new_g


def get_graph_from_shared_mem(graph_name: str, rank: int, local_group: dist.ProcessGroup) -> dgl.DGLHeteroGraph:
    """Get a graph from shared memory."""
    # get ntypes/etypes and keys and tensor shape and dtype from local worker
    recv = [None]
    dist.broadcast_object_list(recv, src=rank, group=local_group)
    data = _deserialize_data(recv[0])

    start = time.time()
    while True:
        g, ntypes, etypes = heterograph_index.create_heterograph_from_shared_memory(
            graph_name
        )
    
        if g is None:
            time.sleep(0.05)
            continue    
        break
    print(f"loading graph from shared memory takes {time.time() - start:.3f} seconds")
    g = dgl.DGLGraph(g, ntypes, etypes)

    # Get node features from shared memory
    for ntype in g.ntypes:
        keys = data["ndata_keys"][ntype]
        for key in keys:
            shape = data["ndata_shapes"][ntype][key]
            dtype = data["ndata_dtypes"][ntype][key]
            feature = _get_shared_mem_ndata(g, graph_name, ntype+"_"+key, shape, dtype)
            g.nodes[ntype].data[key] = feature
    
    # Get edge features from shared memory
    for etype in g.canonical_etypes:
        keys = data["edata_keys"][etype]
        for key in keys:
            shape = data["edata_shapes"][etype][key]
            dtype = data["edata_dtypes"][etype][key]
            feature = _get_shared_mem_edata(g, graph_name, '-'.join(etype)+"_"+key, shape, dtype)
            g.edges[etype].data[key] = feature

    return g
        