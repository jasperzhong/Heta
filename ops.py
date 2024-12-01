"""operators"""
from typing import List

import torch as th
from dgl.backend.pytorch.sparse import gsddmm, gspmm
from torch.distributed import ReduceOp, all_reduce


def multi_edge_softmax(sub_graph_lst: List, logits_lst: List, norm_by='dst', dist: bool = False):
    """global edge softmax 

    compute the global score_max 
    """
    gidx_lst = [
        sub_graph._graph if norm_by == 'dst' else sub_graph._graph.reverse() \
            for sub_graph in sub_graph_lst 
    ]

    score_max_lst = [
        gspmm(gidx, 'copy_rhs', 'max', None, logits)
        for gidx, logits in zip(gidx_lst, logits_lst)
    ]

    score_max = th.stack(score_max_lst, dim=1).max(dim=1)[0]

    if dist:
        all_reduce(score_max, op=ReduceOp.MAX)
    
    # sub and exp 
    score_lst = [
        th.exp(gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
        for gidx, score in zip(gidx_lst, logits_lst)
    ]

    # sum
    score_sum = th.stack(score_lst, dim=1).sum(dim=1)

    if dist:
        all_reduce(score_sum, op=ReduceOp.SUM)
    
    import pdb 
    pdb.set_trace()
    # div
    out_lst = [
        gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
        for gidx, score in zip(gidx_lst, score_lst)
    ]   

    return out_lst

    
