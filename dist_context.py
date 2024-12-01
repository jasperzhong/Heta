import os
import sys

from dgl.distributed import rpc
from dgl.distributed.constants import MAX_QUEUE_SIZE
from dgl.distributed.dist_context import CustomPool
from dgl.distributed.graph_partition_book import (HeteroDataName,
                                                  PartitionPolicy)
from dgl.distributed.kvstore import KVServer, init_kvstore
from dgl.distributed.partition import load_partition_book
from dgl.distributed.role import init_role
from dgl.distributed.rpc_client import connect_to_server
from dgl.distributed.rpc_server import start_server
from dgl.distributed.server_state import ServerState


def initialize(
    ip_config,
    max_queue_size=MAX_QUEUE_SIZE,
    net_type="socket",
    num_worker_threads=1,
):
    """Initialize DGL's distributed module

    This function initializes DGL's distributed module. It acts differently in server
    or client modes. In the server mode, it runs the server code and never returns.
    In the client mode, it builds connections with servers for communication and
    creates worker processes for distributed sampling.

    Parameters
    ----------
    ip_config: str
        File path of ip_config file
    max_queue_size : int
        Maximal size (bytes) of client queue buffer (~20 GB on default).

        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str, optional
        Networking type. Valid options are: ``'socket'``, ``'tensorpipe'``.

        Default: ``'socket'``
    num_worker_threads: int
        The number of OMP threads in each sampler process.

    Note
    ----
    Users have to invoke this API before any DGL's distributed API and framework-specific
    distributed API. For example, when used with Pytorch, users have to invoke this function
    before Pytorch's `pytorch.distributed.init_process_group`.
    """
    if os.environ.get("DGL_ROLE", "client") == "server":

        assert (
            os.environ.get("DGL_SERVER_ID") is not None
        ), "Please define DGL_SERVER_ID to run DistGraph server"
        assert (
            os.environ.get("DGL_IP_CONFIG") is not None
        ), "Please define DGL_IP_CONFIG to run DistGraph server"
        assert (
            os.environ.get("DGL_NUM_SERVER") is not None
        ), "Please define DGL_NUM_SERVER to run DistGraph server"
        assert (
            os.environ.get("DGL_NUM_CLIENT") is not None
        ), "Please define DGL_NUM_CLIENT to run DistGraph server"
        assert (
            os.environ.get("DGL_CONF_PATH") is not None
        ), "Please define DGL_CONF_PATH to run DistGraph server"
        formats = os.environ.get("DGL_GRAPH_FORMAT", "csc").split(",")
        formats = [f.strip() for f in formats]
        rpc.reset()
        keep_alive = bool(int(os.environ.get("DGL_KEEP_ALIVE", 0)))
        server_id = int(os.environ.get("DGL_SERVER_ID"))
        ip_config = os.environ.get("DGL_IP_CONFIG")
        num_servers = int(os.environ.get("DGL_NUM_SERVER"))
        num_clients = int(os.environ.get("DGL_NUM_CLIENT"))
        kv_store = KVServer(
            server_id,
            ip_config,
            num_servers,
            num_clients
        )
        gpb, graph_name, ntypes, etypes  = load_partition_book(os.environ.get("DGL_CONF_PATH"), part_id=kv_store.part_id)
        print("initialize, ntypes: ", ntypes)
        for ntype in ntypes:
            node_name = HeteroDataName(True, ntype, "")
            kv_store.add_part_policy(
                PartitionPolicy(node_name.policy_str, gpb)
            )

        server_state = ServerState(
            kv_store=kv_store,
            local_g=None,
            partition_book=gpb,
            keep_alive=keep_alive
        )
        start_server(
            server_id,
            ip_config,
            num_servers,
            num_clients,
            server_state=server_state,
            net_type=net_type,
        )
        print(
            "start graph service on server {} for part {}".format(
                server_id, kv_store.part_id
            )
        )
        sys.exit()
    else:
        num_workers = int(os.environ.get("DGL_NUM_SAMPLER", 0))
        num_servers = int(os.environ.get("DGL_NUM_SERVER", 1))
        group_id = int(os.environ.get("DGL_GROUP_ID", 0))
        rpc.reset()
        global SAMPLER_POOL
        global NUM_SAMPLER_WORKERS
        is_standalone = (
            os.environ.get("DGL_DIST_MODE", "standalone") == "standalone"
        )
        if num_workers > 0 and not is_standalone:
            SAMPLER_POOL = CustomPool(
                num_workers,
                (
                    ip_config,
                    num_servers,
                    max_queue_size,
                    net_type,
                    "sampler",
                    num_worker_threads,
                    group_id,
                ),
            )
        else:
            SAMPLER_POOL = None
        NUM_SAMPLER_WORKERS = num_workers
        if not is_standalone:
            assert (
                num_servers is not None and num_servers > 0
            ), "The number of servers per machine must be specified with a positive number."
            connect_to_server(
                ip_config,
                num_servers,
                max_queue_size,
                net_type,
                group_id=group_id,
            )
        init_role("default")
        init_kvstore(ip_config, num_servers, "default")


