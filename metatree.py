"""metatree"""
from typing import Any, List, Tuple

import dgl
import networkx as nx


def get_num_nodes(rels: List[Tuple], graph: dgl.DGLGraph):
    """number of nodes in the tree"""
    tot = 0
    for u, etype, v in rels:
        tot += graph.number_of_nodes(u)
    return tot

def get_num_edges(rels: List[Tuple], graph: dgl.DGLGraph):
    """number of edges in the tree"""
    tot = 0
    for rel in rels:
        tot += graph.number_of_edges(rel)
    return tot

class Subtree:
    """subtree of metatree"""

    def __init__(self, subtree: nx.DiGraph, node2size):
        self._subtree  = subtree
        edges = [(u.split(':')[1], data['etype'], v.split(':')[1], data['weight']) for u, v, data in subtree.edges(data=True)]
        deduplicated_edges = list(set(edges))
        self._rels = [(u, etype, v) for u, etype, v, _ in deduplicated_edges]
        self._weight = sum([weight for _, _, _, weight in deduplicated_edges])
        leave_nodes = [
            node.split(':')[1] for node in subtree.nodes() 
        ]
        self._leave_nodes = list(set(leave_nodes))
        # add leave node weight
        self._weight += sum([node2size[node] for node in self._leave_nodes])
    
    @property
    def num_rels(self):
        return len(self._rels)
    
    @property
    def subtree(self):
        return self._subtree

    @property
    def weight(self):
        return self._weight
    
    @property
    def rels(self):
        return self._rels
    
    @property
    def leave_nodes(self):
        return self._leave_nodes
    
    def __repr__(self):
        """print weights and rels"""
        return "Subtree(weight={}, rels={})".format(self._weight, self._subtree.edges(data="etype"))


class MetaTree:
    """A metagraph is a tree of node/edge types. A metatree is a tree of metagraphs 
    rooted at a specific node type using breadth-first search.

    It is weighted and directed. The weight of an edge is the number of edges in the
    corresponding relation. The weight of a node is the sum of the weights of its
    outgoing edges.
    """

    def __init__(self, graph: dgl.DGLGraph, root: str, list_of_metapaths: List[List[Tuple[str, str, str]]] = None, reverse_edge_type_prefix="rev_", depth: int =2):
        """
        Args:
            graph (dgl.DGLGraph): the input graph
            root (str): the root node type
            list_of_metapaths (List[List[Tuple[str, str, str]]]): a list of metapaths (src ntype, etype, dst ntype)
            reverse_edge_type_prefix (str): the prefix of reverse edge type
            depth (int): the depth of the metatree
        """
        self._mg = graph.metagraph()
        self._root = root
        self._reverse_edge_type_prefix = reverse_edge_type_prefix

        rels = graph.canonical_etypes
        self._rel2size = {rel: graph.number_of_edges(etype=rel)  for rel in rels}
        self._node2size = {ntype: graph.number_of_nodes(ntype=ntype) for ntype in graph.ntypes}

        # build the metatree
        if list_of_metapaths is not None:
            self._tree = self._build_from_metapaths(root, list_of_metapaths)
        else:
            self._tree = self._build_metatree(root, depth)

    def _build_from_metapaths(self, root: str, list_of_metapaths: List[List[Tuple[str, str, str]]]):
        tree = nx.DiGraph()
        tree.add_node("root:"+root, weight=self._node2size[root])
        for metapath in list_of_metapaths:
            for i in range(len(metapath)):
                src, etype, dst = metapath[i]
                src_layer = len(metapath) - i
                dst_layer = src_layer - 1
                src_label = "hop" + str(src_layer)
                dst_label = "hop" + str(dst_layer) if dst_layer > 0 else "root"
                labeled_src = src_label + ':' + src
                labeled_dst = dst_label + ':' + dst
                if labeled_src not in tree:
                    tree.add_node(labeled_src, weight=self._node2size[src])
                if labeled_dst not in tree:
                    tree.add_node(labeled_dst, weight=self._node2size[dst])
                if tree.has_edge(labeled_src, labeled_dst):
                    pass
                else:
                    tree.add_edge(labeled_src, labeled_dst, weight=self._rel2size[(src, etype, dst)], etype=etype)
                    print(f"add edge {labeled_src} {etype} {labeled_dst}")

        return tree
    
    def _build_metatree(self, root: str, depth: int = 2):
        """bfs for depth steps"""
        tree = nx.DiGraph()
        tree.add_node("root:" + root, weight=self._node2size[root])
        queue = [root]
        for i in range(depth):
            new_queue = []
            for node in queue:
                for _, neighbor, etype in self._mg.out_edges(node, keys=True):
                    src, dst = neighbor, node
                    if etype.startswith(self._reverse_edge_type_prefix):
                        etype = etype[len(self._reverse_edge_type_prefix):]
                    elif src == dst:
                        pass
                    else:
                        etype = self._reverse_edge_type_prefix + etype
                    
                    labeled_src = "hop" + str(i+1) + ':' + src
                    labeled_dst = "hop" + str(i) + ':' + dst if i > 0 else "root" + ':' + dst
                    if labeled_src not in tree:
                        tree.add_node(labeled_src, weight=self._node2size[src])
                    if labeled_dst not in tree:
                        tree.add_node(labeled_dst, weight=self._node2size[dst])
                    
                    if tree.has_edge(labeled_src, labeled_dst):
                        pass
                    else:
                        tree.add_edge(labeled_src, labeled_dst, weight=self._rel2size[(src, etype, dst)], etype=etype)
                        print(f"add edge {labeled_src} {etype} {labeled_dst}")
                        new_queue.append(src)
            queue = new_queue
        
        return tree

    @property
    def tree(self):
        return Subtree(self._tree, self._node2size)
    
    def get_all_subtrees_from_root(self) -> List[Subtree]:
        """get all subtrees rooted at the root node"""
        # each child of root is a subtree
        subtrees = []
        tree_for_search = self._tree.reverse(copy=True)
        # remove root in this search tree
        root = "root:" + self._root
        tree_for_search.remove_node(root)

        for child, _, data in self._tree.in_edges(root, data=True):
            if child == root:
                subtree = self._tree.edge_subgraph([(child, root)])
            else:
                dfs_edges = list(nx.dfs_edges(tree_for_search, child))
                # reverse the edges
                dfs_edges = [(v, u) for u, v in dfs_edges]
                subtree = self._tree.edge_subgraph(dfs_edges).copy()
                subtree.add_edge(child, root, **data)
                if self._tree.has_edge(root, child):
                    subtree.add_edge(root, child, **self._tree[root][child])
            subtrees.append(Subtree(subtree, self._node2size))
        return subtrees
        