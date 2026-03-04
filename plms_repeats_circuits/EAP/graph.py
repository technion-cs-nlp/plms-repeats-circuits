# Based on code from https://github.com/hannamw/EAP-IG/blob/0edbdd72e3683db69c363a23deb1775f44ec8376/eap/graph.py 
from typing import List, Dict, Union, Tuple, Literal, Optional, Set, Type
from collections import defaultdict
from pathlib import Path 
import json
import heapq
from enum import Enum
import torch
from transformer_lens import HookedESM3, SupportedESM3Config, HookedTransformerConfig, HookedESMC, SupportedESMCConfig
from transformer_lens.components.mlps.esm3_hooked_mlp import swiglu_correction_fn
import numpy as np
import warnings

class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs, 
    while the out_hook gets its outputs.
    """
    name: str #name of the component 
    layer: int #layer of the component 
    in_hook: str #name of the input hook of the component 
    out_hook: str #name of the output hook of the component 
    index: Tuple #index that represent the index of the HEAD
    parents: Set['Node'] #set of parents representing the nodes that enter the node 
    parent_edges: Set['Edge'] #set representing the edges that enter the node 
    children: Set['Node'] #set of children representing the nodes that exit the node
    child_edges: Set['Edge']   #set representing the edges that exit the node 
    qkv_inputs: Optional[List[str]] #set of hooks for hooking qkv input :) 
    in_graph: bool #whether the node is included in the final circuit, if not it will not be used in the final circuit
    score: Optional[float] = None #score of the node via eap, if not set it will not be used in the final circuit
    
    def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple, qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook 
        self.index = index
        self.in_graph = True #whether the node is included in the final circuit 
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs
        self.score = None #score of the node via eap, if not set it will not be used in the final circuit

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Node({self.name}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)


class LogitNode(Node):
    def __init__(self, n_layers:int):
        #name = f'logits_{logits_type}'
        name="logits"
        index = slice(None) #It effectively means "slice the entire sequence." we dont need slicing since its not like different attention heads
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index) #there is not output hook only input

class MLPNode(Node):
    def __init__(self, layer: int):
        name = f'm{layer}' 
        index = slice(None) #It effectively means "slice the entire sequence." we dont need slicing since its not like different attention heads
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index)

class MLPWithNeuronNode(MLPNode):
    def __init__(self, layer: int, neurons_num: int): 
        super().__init__(layer=layer)
        self.neurons_num = neurons_num
        self.neurons_scores = None #scores of the neurons, will be set later
        self.neurons_indicies_in_graph = torch.arange(neurons_num) # creates a tensor of neurons indices, #it means all neurons are in the graph at first
        self.neurons_out_hook = f"blocks.{layer}.mlp.hook_post"

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int):
        name = f'a{layer}.h{head}' 
        self.head = head
        index = (slice(None), slice(None), head) #Full slice for first two dimensions, specific index (e.g., 1) in the third dimension
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, [f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    def __init__(self):
        name = 'input' 
        index = slice(None) #It effectively means "slice the entire sequence." we dont need slicing since its not like different attention heads
        super().__init__(name, 0, '', "hook_embed", index)  #only hooking embedding output , not input 


class Edge:
    name: str #describes the connection of two nodes.
    parent: Node #the in part of the edge
    child: Node #the out part of the edge
    hook: str #name of the hook that need to be patched in order to path the info flow
    index: Tuple # the index need to be chosen for patching activations
    score : Optional[float] #score of the edge via eap
    in_graph: bool #whether the edge is included in the final circuit 
    weight: Optional[float] #weight of the edge
    def __init__(self, parent: Node, child: Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None, weight: Optional[float]=1.0):
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent 
        self.child = child
        self.qkv = qkv
        self.score = None
        self.in_graph = True #at first all edges are in the graph
        self.weight = weight
        if isinstance(child, AttentionNode): #we require hooking entire qkv and not a specific attention node
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input' #maybe we should use the hook name from node?
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook
    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0: 
            return "#FF0000" #red
        else:
            return "#000000" #black

    def __eq__(self, other):
        return self.name == other.name

    
    def __repr__(self):
        return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class GraphType(Enum):
    Edges= 1
    Nodes= 2

class Graph:
    nodes: Dict[str, Node] #contains a set of nodes
    edges: Dict[str, Edge] #contatins a set of edges
    n_forward: int #TODO-FIGURE 
    n_backward: int #TODO-FIGURE 
    cfg: HookedTransformerConfig #is set when we give model !
    graph_type: GraphType = GraphType.Edges #if we want to use nodes graph or edges graph

    def __init__(self, graph_type=GraphType.Edges):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0
        self.graph_type = graph_type

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None, weight: Optional[float]=1.0):
        edge = Edge(parent, child, qkv, weight)
        self.edges[edge.name] = edge #name is defined in the initialization of the edge 
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)
    
    def prev_index(self, node: Node) -> Union[int, slice]:
        """Return the forward index before which all nodes contribute to the input of the given node
        Args:
            node (Node): The node to get the prev forward index of

        Returns:
            Union[int, slice]: an index representing the prev forward index of the node
        """
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            if self.cfg['parallel_attn_mlp']:
                return 1 + node.layer * (self.cfg['n_heads'] + 1)
            else:
                return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def forward_index(self, node:Node, attn_slice=True):
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            raise ValueError("No forward index for logits node")
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads'] #embedding layer + the layers that was until now *num components per layer + num nodes in current layer related to attention
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head #slice to get index of all heads... else we get the index of the specific head we asked
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
        

    def backward_index(self, node:Node, qkv=None, attn_slice=True):
        if isinstance(node, InputNode):
            raise ValueError(f"No backward for input node") #we currenly dont care about a gradient with respect to input ! might not be true when adding new component #TODO- Figure
        elif isinstance(node, LogitNode):
            return -1 #we will save it in last location 
        elif isinstance(node, MLPNode):
            return (node.layer) * (3 * self.cfg['n_heads'] + 1) + 3 * self.cfg['n_heads'] #now we save backword gradient also for qkv so it *3. we dont have 1 in the begin cause we dont save gradient for embedding
        elif isinstance(node, AttentionNode):
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            i = node.layer * (3 * self.cfg['n_heads'] + 1) + ('qkv'.index(qkv) * self.cfg['n_heads']) #we are saving q gradients for all heads, then k gradients, then v gradients. so if for example we take index of k which is 1 * n_heads we skip all the locations where we saved q:) 
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head #same idea as forward index
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def get_scores(self, nonzero=False, in_graph=False, sort=True):
        """Return the scores of the edges in the graph
        Args:
            nonzero (bool): whether to return only the nonzero scores
            in_graph (bool): whether to return only the scores of the edges that are in the graph
            sort (bool): whether to sort the scores before returning them
        Returns:
            torch.Tensor: a tensor of the scores of the edges in the graph"""
        s = torch.tensor([edge.score for edge in self.edges.values() if edge.score != 0 and (edge.in_graph or not in_graph)]) if nonzero else torch.tensor([edge.score for edge in self.edges.values()])
        return torch.sort(s).values if sort else s
    
    def set_all_edges_in_graph(self, in_graph: bool = True):
        for edge in self.edges.values():
            edge.in_graph = in_graph
        for node in self.nodes.values():
            node.in_graph = in_graph
    
    def set_all_nodes_in_graph(self, in_graph: bool = True):
        for node in self.nodes.values():
            node.in_graph = in_graph
        for edge in self.edges.values():
            edge.in_graph = in_graph
    
    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())
    
    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())
    
    def count_total_nodes(self):
        return len(self.nodes.values())
    
    def count_total_edges(self):
        return len(self.edges.values())
    
    def count_attention_nodes(self, filter_by_in_graph: bool = True):
        return len([node for node in self.nodes.values() if isinstance(node, AttentionNode) and (not filter_by_in_graph or node.in_graph == True)])
    
    def count_mlp_nodes(self, filter_by_in_graph: bool = True):
        return len([node for node in self.nodes.values() if isinstance(node, MLPNode) and (not filter_by_in_graph or node.in_graph == True)])
    
    def check_if_node_is_one_of_types(self, node: Node, node_types: List[Type[Node]]):
        return any(isinstance(node, node_type) for node_type in node_types)
    
    def check_if_node_name_is_one_of_names(self, node: Node, node_names: List[str]):
        return any(node.name == node_name for node_name in node_names)
    
    def aggregate_edges_by_types(self, edge_src_node_types, edge_dst_node_types, aggregation_type: Literal['sum', 'mean', 'max', 'min'], filter_by_in_graph: bool = True):
        if self.graph_type != GraphType.Edges:
            raise ValueError(f"Aggregate edges by types is only supported for Edges Graph. Use Edges Graph instead.")
        if aggregation_type not in ['sum', 'mean', 'max', 'min']:
            raise ValueError(f"Invalid aggregation type: {aggregation_type}. Must be one of ['sum', 'mean', 'max', 'min']")
        if aggregation_type == 'sum':
          return sum([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_is_one_of_types(edge.parent, edge_src_node_types) and self.check_if_node_is_one_of_types(edge.child, edge_dst_node_types) and (not filter_by_in_graph or edge.in_graph == True)]) 
        elif aggregation_type == 'mean':
          return torch.mean([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_is_one_of_types(edge.parent, edge_src_node_types) and self.check_if_node_is_one_of_types(edge.child, edge_dst_node_types) and (not filter_by_in_graph or edge.in_graph == True)]).item()
        elif aggregation_type == 'max':
          return max([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_is_one_of_types(edge.parent, edge_src_node_types) and self.check_if_node_is_one_of_types(edge.child, edge_dst_node_types) and (not filter_by_in_graph or edge.in_graph == True)])
        elif aggregation_type == 'min':
          return min([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_is_one_of_types(edge.parent, edge_src_node_types) and self.check_if_node_is_one_of_types(edge.child, edge_dst_node_types) and (not filter_by_in_graph or edge.in_graph == True)])
        else:
            raise ValueError(f"Invalid aggregation type: {aggregation_type}. Must be one of ['sum', 'mean', 'max', 'min']")
    
    def aggregate_edges_by_names(self, edge_src_node_names, edge_dst_node_names, aggregation_type: Literal['sum', 'mean', 'max', 'min'], filter_by_in_graph: bool = True):
      if self.graph_type != GraphType.Edges:
        raise ValueError(f"Aggregate edges by names is only supported for Edges Graph. Use Edges Graph instead.")
      if aggregation_type not in ['sum', 'mean', 'max', 'min']:
        raise ValueError(f"Invalid aggregation type: {aggregation_type}. Must be one of ['sum', 'mean', 'max', 'min']")
      if aggregation_type == 'sum':
        return sum([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_name_is_one_of_names(edge.parent, edge_src_node_names) and self.check_if_node_name_is_one_of_names(edge.child, edge_dst_node_names) and (not filter_by_in_graph or edge.in_graph == True)])
      elif aggregation_type == 'mean':
        return torch.mean([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_name_is_one_of_names(edge.parent, edge_src_node_names) and self.check_if_node_name_is_one_of_names(edge.child, edge_dst_node_names) and (not filter_by_in_graph or edge.in_graph == True)]).item()
      elif aggregation_type == 'max':
        return max([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_name_is_one_of_names(edge.parent, edge_src_node_names) and self.check_if_node_name_is_one_of_names(edge.child, edge_dst_node_names) and (not filter_by_in_graph or edge.in_graph == True)])
      elif aggregation_type == 'min':
        return min([edge.score for edge in self.edges.values() if edge.score is not None and self.check_if_node_name_is_one_of_names(edge.parent, edge_src_node_names) and self.check_if_node_name_is_one_of_names(edge.child, edge_dst_node_names) and (not filter_by_in_graph or edge.in_graph == True)])
      else:
        raise ValueError(f"Invalid aggregation type: {aggregation_type}. Must be one of ['sum', 'mean', 'max', 'min']")

    def apply_threshold(self, threshold: float, absolute: bool):
        """Apply a threshold to the graph, setting the in_graph attribute of edges Graph / Nodes Graph to True if the score is above the threshold
        Args:
            threshold (float): the threshold to apply
            absolute (bool): whether to take the absolute value of the scores before applying the threshold"""   
        
        #always reset
        for node in self.nodes.values():
            node.in_graph = False 
        for edge in self.edges.values():
            edge.in_graph = False

        threshold = float(threshold)

        if self.graph_type == GraphType.Nodes:
            unscored_nodes = [node for node in self.nodes.values() if node.score is None]
            if len(unscored_nodes) > 1 or (len(unscored_nodes) == 1 and unscored_nodes[0].name != 'logits'):
                print(f"Warning: {len(unscored_nodes)} nodes have no score is greater then expected or logits is not the node not scored.")
            else:
                for node in unscored_nodes:
                    node.score = torch.inf #we set the score to inf so it will be included in the top n nodes

            for node in self.nodes.values():
                node.in_graph = abs(node.score) >= threshold if absolute else node.score >= threshold
                
            for edge in self.edges.values():
                edge.in_graph = edge.parent.in_graph and edge.child.in_graph

        elif self.graph_type == GraphType.Edges:
            for node in self.nodes.values():
                node.in_graph = True    # will be pruned later if needed

            for edge in self.edges.values():
                edge.in_graph = abs(edge.score) >= threshold if absolute else edge.score >= threshold

            
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}. Must be either GraphType.Edges or GraphType.Nodes.")
    

            
    def apply_topn(self, n:int, absolute: bool):
        """Apply a top n filter to the graph, setting the in_graph attribute of the n edges with the highest scores to True
        Args:
            n (int): the number of edges to include
            absolute (bool): whether to take the absolute value of the scores before applying the threshold"""
        
        #always reset
        for node in self.nodes.values():
            node.in_graph = False 
        for edge in self.edges.values():
            edge.in_graph = False

        if n < 0:
            raise ValueError(f"n must be greater than 0, but got {n}")
        if n == 0:
            return
        if self.graph_type == GraphType.Nodes:
            unscored_nodes = [node for node in self.nodes.values() if node.score is None]
            if len(unscored_nodes) > 1 or (len(unscored_nodes) == 1 and unscored_nodes[0].name != 'logits'):
                print(f"Warning: {len(unscored_nodes)} nodes have no score is greater then expected or logits is not the node not scored.")
            else:
                for node in unscored_nodes:
                    node.score = torch.inf #we set the score to inf so it will be included in the top n nodes

            if n >= len(self.nodes):
                n = len(self.nodes)
            a = abs if absolute else lambda x: x
            sorted_nodes = sorted(list(self.nodes.values()), key = lambda node: a(node.score), reverse=True)
            for node in sorted_nodes[:n]:
                node.in_graph = True
                for edge in node.child_edges:
                    edge.in_graph = True

            # for edge in self.edges.values():
            #     edge.in_graph = edge.parent.in_graph and edge.child.in_graph  
        
        elif self.graph_type == GraphType.Edges:
            if n >= len(self.edges):
                n = len(self.edges) #if n is larger than the number of edges, we just include all edges

            a = abs if absolute else lambda x: x
            for node in self.nodes.values():
                node.in_graph = False

            sorted_edges = sorted(list(self.edges.values()), key = lambda edge: a(edge.score), reverse=True)
            for edge in sorted_edges[:n]:
                edge.in_graph = True 
                edge.parent.in_graph = True 
                edge.child.in_graph = True 

            for edge in sorted_edges[n:]:
                edge.in_graph = False
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}. Must be either GraphType.Edges or GraphType.Nodes.")
        

    def apply_greedy(self, n_edges, reset=True, absolute: bool = True):
        """Perform a greedy search on the graph, starting from the logits node and selecting the reachable edge with the highest score at each step (of n_edges). An edge is reachable if its child is in the graph; if an edge is selected but its parent is not in the graph, the parent is added.
        Args:
            n_edges (int): the number of edges to include
            reset (bool): whether to reset the in_graph attribute of all nodes and edges before applying the greedy search (defaults to True, you probably want to keep it that way)
            absolute (bool): whether to take the absolute value of the scores before applying the threshold"""
        if self.graph_type != GraphType.Edges:
            raise ValueError(f"Greedy search is only supported for Edges Graph. Use Edges Graph instead.")
        
        if reset:
            for node in self.nodes.values():
                node.in_graph = False 
            for edge in self.edges.values():
                edge.in_graph = False
            self.nodes['logits'].in_graph = True

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph], key = lambda edge: abs_id(edge.score), reverse=True)

        edges = heapq.merge(candidate_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key = lambda edge: abs_id(edge.score), reverse=True)
                edges = heapq.merge(edges, parent_parent_edges, key = lambda edge: abs_id(edge.score), reverse=True)

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        self.nodes['logits'].in_graph = any(parent_edge.in_graph for parent_edge in self.nodes['logits'].parent_edges)

        for node in reversed(self.nodes.values()): #reversed because we inserted from input... and we only defined result to logits
            if isinstance(node, LogitNode):
                continue 
            
            if any(child_edge.in_graph for child_edge in node.child_edges) : #if the node has at least one child
                node.in_graph = True #it is in the graph!
            else: #no child is in the graph
                if prune_childless: #no childs
                    node.in_graph = False 
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else: 
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True 
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if not isinstance(node, InputNode) and node.in_graph and not any(parent_edge.in_graph for parent_edge in node.parent_edges):
                    node.in_graph = False 
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False


    @classmethod
    def from_model(cls, model_or_config: Union[HookedESM3, HookedTransformerConfig, Dict, HookedESMC], graph_type=GraphType.Edges):
        graph = Graph(graph_type=graph_type)
        if isinstance(model_or_config, HookedESM3) or isinstance(model_or_config, HookedESMC):
            cfg = model_or_config.cfg
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'esm3_scaling_factor': cfg.esm3_scaling_factor}
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'esm3_scaling_factor': cfg.esm3_scaling_factor}
        else:
            graph.cfg = model_or_config
        
        input_node = InputNode()
        graph.nodes[input_node.name] = input_node #we store the first node, not edges yet
        residual_stream = [input_node] #at first we put the input node in the residual stream
        scaling_factor = 1 / graph.cfg["esm3_scaling_factor"]
        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg['n_heads'])] 
            mlp_node = MLPNode(layer)
            
            for attn_node in attn_nodes: 
                graph.nodes[attn_node.name] = attn_node #we store attn nodes, no edges yet
            graph.nodes[mlp_node.name] = mlp_node     #create mlp node and store
                                    
            if graph.cfg['parallel_attn_mlp']:
                for node in residual_stream:
                    for attn_node in attn_nodes:                                    
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter, weight=1.0 if isinstance(node, InputNode) else scaling_factor)
                    graph.add_edge(node, mlp_node,  weight=1.0 if isinstance(node, InputNode) else scaling_factor)
                
                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream: #we add edge from node in the curr redisual stream to each attn head and each qkv
                    for attn_node in attn_nodes:     
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter,  weight=1.0 if isinstance(node, InputNode) else scaling_factor)
                residual_stream += attn_nodes # add current attention to residual stream

                for node in residual_stream:
                    graph.add_edge(node, mlp_node, weight=1.0 if isinstance(node, InputNode) else scaling_factor) #we add out edges only from attn nodes to mlp
                residual_stream.append(mlp_node)
                        
        logit_node = LogitNode(graph.cfg['n_layers'])
        for node in residual_stream:
            graph.add_edge(node, logit_node,  weight=1.0 if isinstance(node, InputNode) else scaling_factor)
            
        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1) #number of nodes we care a about in forward pass : (attn heads nodes+mlp)*n_layers+embedding #TODO- add geometric attention, add embedding layer, add unembedding layer hooks. 
        #in forward we don't need seprate activation for each qkv node since the output of attention head is used to approximate the qkv gradients - the attention layer is a function of qkv so they have same output
        graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1 #number of nodes we care a about in forward pass : (attn heads nodes+mlp)*n_layers+unembedding #TODO- add geometric attention, add embedding layer, add unembedding layer hooks

        return graph


    
    
    def to_json(self, filename: str):
        """Save the graph to a JSON file.
        Args:
            filename (str): the filename to save the graph to.
        """
        d = {
            'cfg': self.cfg,
            'graph_type': self.graph_type.name,  # Save enum name
            'nodes': {
                str(name): {
                    'in_graph': bool(node.in_graph),
                    'score': None if not hasattr(node, 'score') or node.score is None else float(node.score)
                }
                for name, node in self.nodes.items()
            },
            'edges': {
                str(name): {
                    'score': None if edge.score is None else float(edge.score),
                    'in_graph': bool(edge.in_graph)
                }
                for name, edge in self.edges.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, filename):
        """
        Load a Graph object from a JSON file.
        Supports both old and new formats.
        """
        with open(filename, 'r') as f:
            d = json.load(f)

        g = Graph.from_model(d['cfg'])

        # Parse graph_type if present, fallback to Edges
        graph_type_str = d.get('graph_type', 'Edges')
        try:
            g.graph_type = GraphType[graph_type_str]
        except KeyError:
            g.graph_type = GraphType.Edges  # fallback if unknown

        for name, val in d['nodes'].items():
            if isinstance(val, dict):
                g.nodes[name].in_graph = val.get('in_graph', False)
                if 'score' in val:
                    g.nodes[name].score = val['score']
            else:
                # backward-compatible: old format where val is bool
                g.nodes[name].in_graph = val

        for name, info in d['edges'].items():
            g.edges[name].score = info['score']
            g.edges[name].in_graph = info['in_graph']

        return g
    

    def __eq__(self, other):
        #asserting the graphs has same node names
        if not isinstance(other, Graph):
            return False
        if self.graph_type != other.graph_type:
            return False
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (set(self.edges.keys()) == set(other.edges.keys()))
        if not keys_equal:
            return False
        #asserting the graphs has same in_graph nodes
        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False
            if node.score != None and other.nodes[name].score != None and not np.allclose(node.score, other.nodes[name].score):
                return False
         #asserting the edges has same in_graph edges   
        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(edge.score, other.edges[name].score):
                return False
        return True

    def get_attention_nodes_names_in_graph(self):
        return [node.name for node in self.nodes.values() if isinstance(node, AttentionNode) and node.in_graph]
    
    def get_mlp_nodes_names_in_graph(self):
        return [node.name for node in self.nodes.values() if isinstance(node, MLPNode) and node.in_graph]

    def get_nodes_names_in_graph(self):
        return [node.name for node in self.nodes.values() if node.in_graph]
    
    def get_edges_in_graph(self):
        """Return set of edge names that are in the graph."""
        return {edge_key for edge_key, edge in self.edges.items() if edge.in_graph}
    
    def get_nodes_in_graph(self):
        """Return set of node names that are in the graph."""
        return {node_key for node_key, node in self.nodes.items() if node.in_graph}
    
    def reset_graph_state(self):
        """Reset all nodes and edges to in_graph=False."""
        self.set_all_edges_in_graph(in_graph=False)


class NeuronGraph(Graph):
    # this only supports nodes graph currently, not edges!
    def __init__(self):
        super().__init__(GraphType.Nodes)
        self.n_layers = None #will be set later
        self.n_heads_per_layer = None #will be set later
        self.n_neurons_per_mlp = None #will be set later
        self.edges = None # not supported for neurons graph
        self.n_backward = None # not supported for neurons graph
        self.n_forward_neurons = None # will be set later

        #show some warning for awarness it only supports nodes graph currently, not edges! (print it like the deprecated warnings when i run something in jupyter and its deprecated)
        warnings.warn("NeuronGraph only supports nodes graph currently, not edges!")

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None, weight: Optional[float]=1.0):
        raise NotImplementedError("NeuronGraph does not support add_edge")
    
    def prev_index(self, node: Node) -> Union[int, slice]:
        raise NotImplementedError("NeuronGraph does not support prev_index")
    
    def forward_index(self, node:Node, attn_slice=True, return_index_in_neurons_array=False):
        if return_index_in_neurons_array:
            if isinstance(node, MLPWithNeuronNode):
               i = node.layer * self.n_neurons_per_mlp 
               return slice(i, i + self.n_neurons_per_mlp) # slice to get index of all neurons... we dont support slicing of neurons yet
               #so for mlp layer 1 and 4096 neurons, we start in [4096, i+4096)]
            else:
                raise ValueError(f"Node {node} is not a MLPWithNeuronNode")
        else:
            return super().forward_index(node, attn_slice)

    def backward_index(self, node:Node, qkv=None, attn_slice=True, return_index_in_neurons_array=False):
        raise NotImplementedError("NeuronGraph does not support backward_index")

    def get_scores(self, nonzero=False, in_graph=False, sort=True):
        raise NotImplementedError("NeuronGraph does not support get_scores of edges")
    
    def count_included_edges(self):
        raise NotImplementedError("NeuronGraph does not support count_included_edges")

    def count_included_nodes(self):
        count = 0
        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                if node.in_graph:
                    count += len(node.neurons_indicies_in_graph)
            else:
                if node.in_graph:
                    count += 1
        return count

    def count_total_nodes(self):
        count_nodes_not_mlps = len([node for node in self.nodes.values() if not isinstance(node, MLPWithNeuronNode)])
        count_neurons = sum([node.neurons_num for node in self.nodes.values() if isinstance(node, MLPWithNeuronNode)])
        return count_nodes_not_mlps + count_neurons
    
    def count_attention_nodes(self, filter_by_in_graph: bool = True):
        return len([node for node in self.nodes.values() if isinstance(node, AttentionNode) and (not filter_by_in_graph or node.in_graph == True)])
    
    def count_mlp_nodes(self, filter_by_in_graph: bool = True):
        return len([node for node in self.nodes.values() if isinstance(node, MLPWithNeuronNode) and (not filter_by_in_graph or node.in_graph == True)])

    def count_neurons(self, filter_by_in_graph: bool = True):
        count = 0
        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                if filter_by_in_graph:
                    if node.in_graph:
                        count += len(node.neurons_indicies_in_graph)
                else:
                    count += node.neurons_num
        return count

    def count_nodes_not_as_neurons(self, filter_by_in_graph: bool = True):
        return len([node for node in self.nodes.values() if (not filter_by_in_graph or node.in_graph == True)])

    def set_all_nodes_in_graph(self, in_graph: bool = True):
        for node in self.nodes.values():
            node.in_graph = in_graph
            if isinstance(node, MLPWithNeuronNode):
                if in_graph:
                    node.neurons_indicies_in_graph = torch.arange(node.neurons_num)
                else:
                    node.neurons_indicies_in_graph = torch.empty(0, dtype=torch.long)
    
    def set_all_mlp_with_neurons_in_graph(self, in_graph: bool = True):
        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                node.in_graph = in_graph
                if in_graph:
                    node.neurons_indicies_in_graph = torch.arange(node.neurons_num)
                else:
                    node.neurons_indicies_in_graph = torch.empty(0, dtype=torch.long)
    
    def set_all_edges_in_graph(self, in_graph: bool = True):
        raise NotImplementedError("NeuronGraph does not support set_all_edges_in_graph")
    
    def apply_threshold(self, threshold: float, absolute: bool):
        raise NotImplementedError("NeuronGraph does not support apply_threshold")
    

    def apply_topn(self, n:int, absolute: bool):

        #always reset
        self.set_all_nodes_in_graph(in_graph=False)

        if n < 0:
            raise ValueError(f"n must be greater than 0, but got {n}")
        if n == 0:
            return
        if n >= self.count_total_nodes():
            self.set_all_nodes_in_graph(in_graph=True)
            return
        
        unscored_nodes = [node for node in self.nodes.values() if node.score is None]
        if len(unscored_nodes) > 1 or (len(unscored_nodes) == 1 and unscored_nodes[0].name != 'logits'):
            print(f"Warning: {len(unscored_nodes)} nodes have no score is greater then expected or logits is not the node not scored.")
            raise RuntimeError(f"NeuronGraph has more then one unscored node or logits is not the node not scored.")
        else:
            for node in unscored_nodes:
                node.score = torch.inf #we set the score to inf so it will be included in the top n nodes 
        
        a = abs if absolute else lambda x: x

        # Collect all scores with their identifiers
        all_scores = []
        
        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                # For MLP nodes, we need to handle individual neuron scores
                if node.neurons_scores is not None:
                    for neuron_idx, neuron_score in enumerate(node.neurons_scores):
                        all_scores.append((a(neuron_score), 'neuron', node, neuron_idx))
                else:
                  raise RuntimeError(f"NeuronGraph has no neurons scores for node {node.name}")
            else:
                # For non-MLP nodes (input, attention, logits), use node score
                if node.score is not None:
                    all_scores.append((a(node.score), 'node', node, None))
        
        # Sort by score (descending) and take top n
        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_n_items = all_scores[:n]
        
        # Apply selections
        for score, item_type, node, neuron_idx in top_n_items:
            if item_type == 'node':
                node.in_graph = True
            elif item_type == 'neuron':
                node.in_graph = True
                # Add this neuron to the graph
                if len(node.neurons_indicies_in_graph) == 0:
                    node.neurons_indicies_in_graph = torch.tensor([neuron_idx])
                else:
                    node.neurons_indicies_in_graph = torch.cat([
                        node.neurons_indicies_in_graph, 
                        torch.tensor([neuron_idx])
                    ])
                    node.neurons_indicies_in_graph = torch.unique(node.neurons_indicies_in_graph)
        
        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                node.in_graph = len(node.neurons_indicies_in_graph) > 0


    def get_topn(self, n: int, absolute: bool = True):
        """
        Return the top-n scored elements without applying them to the graph.
        Each element is a tuple (score, type, node_name, neuron_idx or None).
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        a = abs if absolute else (lambda x: x)
        all_scores = []

        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                if node.neurons_scores is not None:
                    for neuron_idx, neuron_score in enumerate(node.neurons_scores):
                        all_scores.append((a(neuron_score), 'neuron', node.name, neuron_idx))
                else:
                    raise RuntimeError(f"NeuronGraph has no neuron scores for node {node.name}")
            else:
                if node.score is not None:
                    all_scores.append((a(node.score), 'node', node.name, None))

        # Sort and take top n
        all_scores.sort(key=lambda x: x[0], reverse=True)
        return all_scores[:n]

    def apply_topn_only_neurons(self, n_neurons:int, n_nodes:int, absolute: bool):
        self.set_all_nodes_in_graph(in_graph=False)
        self.apply_topn_on_nodes(n_nodes, absolute=absolute)

        if n_neurons < 0:
            raise ValueError(f"n_neurons must be greater than 0, but got {n_neurons}")
        if n_neurons == 0:
            return

        n_neurons = min(n_neurons, self.count_neurons(filter_by_in_graph=True))

        a = abs if absolute else lambda x: x

        # Collect all scores with their identifiers
        all_scores = []
        
        for node in self.nodes.values():
            if not node.in_graph:
                continue #we only care about nodes that are in the graph
            if isinstance(node, MLPWithNeuronNode):
                # For MLP nodes, we need to handle individual neuron scores
                if node.neurons_scores is not None:
                    for neuron_idx, neuron_score in enumerate(node.neurons_scores):
                        all_scores.append((a(neuron_score), 'neuron', node, neuron_idx))
                else:
                  raise RuntimeError(f"NeuronGraph has no neurons scores for node {node.name}")
            else:
                continue #we only care about neurons scores
        
        # Sort by score (descending) and take top n
        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_n_items = all_scores[:n_neurons]
        
        self.set_all_mlp_with_neurons_in_graph(in_graph=False) #reset all mlp with neurons in graph
        # Apply selections
        for score, item_type, node, neuron_idx in top_n_items:
                node.in_graph = True
                # Add this neuron to the graph
                if len(node.neurons_indicies_in_graph) == 0:
                    node.neurons_indicies_in_graph = torch.tensor([neuron_idx])
                else:
                    node.neurons_indicies_in_graph = torch.cat([
                        node.neurons_indicies_in_graph, 
                        torch.tensor([neuron_idx])
                    ])
                    node.neurons_indicies_in_graph = torch.unique(node.neurons_indicies_in_graph)
        
        for node in self.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                node.in_graph = len(node.neurons_indicies_in_graph) > 0

    def apply_topn_neurons_per_layer(self, n_neurons:int, n_nodes:int, absolute: bool):
        """Apply top N neurons selection per MLP layer.
        
        First selects top N nodes, then for each MLP layer that is in the graph
        after the top N nodes selection, selects the top N highest-scored neurons
        from that specific layer.
        
        Args:
            n_neurons (int): Number of top neurons to select per MLP layer
            n_nodes (int): Number of top nodes to select first
            absolute (bool): Whether to use absolute values for neuron scores
        """
        self.set_all_nodes_in_graph(in_graph=False)
        self.apply_topn_on_nodes(n_nodes, absolute=absolute)

        if n_neurons < 0:
            raise ValueError(f"n_neurons must be greater than 0, but got {n_neurons}")
        if n_neurons == 0:
            return

        
        # Process each MLP layer independently
        for node in self.nodes.values():
            if not node.in_graph:
                continue #we only care about nodes that are in the graph
            if isinstance(node, MLPWithNeuronNode):
                if node.neurons_scores is None:
                    raise RuntimeError(f"NeuronGraph has no neurons scores for node {node.name}")
                
                scores = torch.as_tensor(node.neurons_scores, dtype=torch.float32)
                scores = scores.abs() if absolute else scores
                topk = torch.topk(scores, min(n_neurons, node.neurons_num))
                node.neurons_indicies_in_graph = torch.sort(topk.indices).values


    def apply_topn_on_nodes(self, n:int, absolute: bool):
        #always reset
        self.set_all_nodes_in_graph(in_graph=False)

        if n < 0:
            raise ValueError(f"n must be greater than 0, but got {n}")
        if n == 0:
            return
        unscored_nodes = [node for node in self.nodes.values() if node.score is None]
        if len(unscored_nodes) > 1 or (len(unscored_nodes) == 1 and unscored_nodes[0].name != 'logits'):
            print(f"Warning: {len(unscored_nodes)} nodes have no score is greater then expected or logits is not the node not scored.")
        else:
            for node in unscored_nodes:
                node.score = torch.inf #we set the score to inf so it will be included in the top n nodes

        if n >= len(self.nodes):
            n = len(self.nodes)
        a = abs if absolute else lambda x: x
        sorted_nodes = sorted(list(self.nodes.values()), key = lambda node: a(node.score), reverse=True)
        for node in sorted_nodes[:n]:
            self.set_node_state(node.name, in_graph=True)


    def apply_greedy(self, n_neurons, reset=True, absolute: bool = True):
        raise NotImplementedError("NeuronGraph does not support apply_greedy")
    
    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        raise NotImplementedError("NeuronGraph does not support prune_dead_nodes")
    
    @classmethod
    def from_model(cls, model_or_config: Union[HookedESM3, HookedTransformerConfig, Dict, HookedESMC], graph_type=GraphType.Edges):
        if graph_type != GraphType.Nodes:
            raise ValueError(f"NeuronGraph can only be created from model with graph_type=GraphType.Nodes")
            
        graph = NeuronGraph()
        
        if isinstance(model_or_config, HookedESM3) or isinstance(model_or_config, HookedESMC):
            cfg = model_or_config.cfg
            d_mlp = model_or_config.blocks[0].mlp.l2.in_features
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'esm3_scaling_factor': cfg.esm3_scaling_factor, 'd_mlp': d_mlp, "is_neurons_graph": True}
            
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            if (cfg.model_name == 'esm3') or ("esmc" in cfg.model_name):
                d_mlp = swiglu_correction_fn(cfg.esm3_mlp_expansion_ratio, cfg.d_model)
            else:
                d_mlp = cfg.d_mlp
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'esm3_scaling_factor': cfg.esm3_scaling_factor, 'd_mlp': d_mlp, "is_neurons_graph": True}
        else:
            graph.cfg = model_or_config
        
        input_node = InputNode()
        graph.nodes[input_node.name] = input_node #we store the first node, not edges yet

        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg['n_heads'])] 
            mlp_node = MLPWithNeuronNode(layer=layer, neurons_num=graph.cfg['d_mlp'])
            
            for attn_node in attn_nodes: 
                graph.nodes[attn_node.name] = attn_node #we store attn nodes, no edges yet
            graph.nodes[mlp_node.name] = mlp_node     #create mlp node and store
                                    
                        
        logit_node = LogitNode(graph.cfg['n_layers'])
            
        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1) #number of nodes we care a about in forward pass : (attn heads nodes+mlp)*n_layers+embedding #TODO- add geometric attention, add embedding layer, add unembedding layer hooks. 

        # set only n_forward_neurons for mlp node
        graph.n_forward_neurons = graph.cfg['n_layers'] * graph.cfg['d_mlp'] # number of neurons in mlp nodes for all layers
        graph.n_layers = graph.cfg['n_layers']
        graph.n_heads_per_layer = graph.cfg['n_heads']
        graph.n_neurons_per_mlp = graph.cfg['d_mlp']

        return graph

    
    def to_json(self, filename: str):
        """Save the graph to a JSON file.
        Args:
            filename (str): the filename to save the graph to.
        """
        d = {
            'cfg': self.cfg,
            'graph_type': self.graph_type.name,  # Save enum name
            'nodes': {
                str(name): {
                    'in_graph': bool(node.in_graph),
                    'score': None if not hasattr(node, 'score') or node.score is None else float(node.score),
                    'neurons_scores': None if not hasattr(node, 'neurons_scores') or node.neurons_scores is None else list(node.neurons_scores.tolist()),
                    'neurons_indicies_in_graph': None if not hasattr(node, 'neurons_indicies_in_graph') or node.neurons_indicies_in_graph is None else list(node.neurons_indicies_in_graph.tolist()),
                }
                for name, node in self.nodes.items()
            },
            'edges': {
            }
        }
        with open(filename, 'w') as f:
            json.dump(d, f)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            d = json.load(f)

        cfg = d.get("cfg", {})
        is_neuron_graph = cfg.get("is_neurons_graph", False)
        has_neuron_scores = any(
            isinstance(v, dict) and v.get("neurons_scores") is not None
            for v in d.get("nodes", {}).values()
        )

        if not (is_neuron_graph and has_neuron_scores):
            return Graph.from_json(filename)

        graph_type = d.get('graph_type', 'Nodes')
        try:
            graph_type = GraphType[graph_type]
        except KeyError:
            graph_type = GraphType.Nodes

        g = NeuronGraph.from_model(cfg, graph_type=graph_type)

        for name, val in d['nodes'].items():
            node = g.nodes.get(name)
            if node is None or not isinstance(val, dict):
                continue

            node.in_graph = val.get('in_graph', False)
            node.score = val.get('score', None)

            if 'neurons_scores' in val and val['neurons_scores'] is not None:
                node.neurons_scores = torch.tensor(val['neurons_scores'])
            if 'neurons_indicies_in_graph' in val and val['neurons_indicies_in_graph'] is not None:
                node.neurons_indicies_in_graph = torch.tensor(
                    val['neurons_indicies_in_graph'], dtype=torch.long
                )

        return g


    def __eq__(self, other):
        #assert its same class
        if not isinstance(other, NeuronGraph):
            return False

        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys()))
        if not keys_equal:
            return False

        #asserting the graphs has same in_graph nodes
        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False
            
            if node.score != None and other.nodes[name].score != None and node.score != other.nodes[name].score:
                return False
            
            if isinstance(node, MLPWithNeuronNode):
                if not torch.equal(
                    node.neurons_indicies_in_graph, other.nodes[name].neurons_indicies_in_graph
                ):
                    return False
                if (
                    node.neurons_scores is not None
                    and other.nodes[name].neurons_scores is not None
                    and not torch.allclose(node.neurons_scores, other.nodes[name].neurons_scores)
                ):
                    return False

        
        return True

    def set_node_state(self, node_name, in_graph: bool):
        self.nodes[node_name].in_graph = in_graph
        if isinstance(self.nodes[node_name], MLPWithNeuronNode):
            if in_graph:
                self.nodes[node_name].neurons_indicies_in_graph = torch.arange(self.nodes[node_name].neurons_num)
            else:
                self.nodes[node_name].neurons_indicies_in_graph = torch.empty(0, dtype=torch.long)
    
    def get_edges_in_graph(self):
        """Not supported for NeuronGraph."""
        raise NotImplementedError("NeuronGraph does not support get_edges_in_graph")
    
    def get_nodes_in_graph(self, neurons_only=False):
        """
        Return set of node identifiers that are in the graph.
        For neurons, returns tuples like ('m0', neuron_idx) for each neuron.
        For other nodes, returns node names.
        
        Args:
            neurons_only: If True, only return neurons from MLP nodes (exclude attention, input, logits)
        """
        nodes_in_graph = set()
        for node_key, node in self.nodes.items():
            if not node.in_graph:
                continue
            
            if isinstance(node, MLPWithNeuronNode):
                # Add each neuron as a tuple (node_name, neuron_idx)
                for neuron_idx in node.neurons_indicies_in_graph.tolist():
                    nodes_in_graph.add((node_key, neuron_idx))
            else:
                # Skip non-neuron nodes if neurons_only is True
                if not neurons_only:
                    nodes_in_graph.add(node_key)
        
        return nodes_in_graph
    
    def reset_graph_state(self):
        """Reset all nodes to in_graph=False, and clear neuron indices for MLP nodes."""
        self.set_all_nodes_in_graph(in_graph=False)

