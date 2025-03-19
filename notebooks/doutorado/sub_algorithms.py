# -*- coding: utf-8 -*-

"""



            SUBDETERMINATED ALGORITHMS

                    FOR THE VERSION 0.2 MAG MODULE



  Some of these algorithms were adapted from networkx.algorithms.  
  https://networkx.github.io/documentation/stable/reference/algorithms/index.html


  Date: 08/28/2020

  Based upon NetworkX 2.2
  
"""

from networkx.exception import NetworkXNoPath
from networkx.utils import not_implemented_for
from itertools import permutations, count
from collections import deque
import networkx as nx
import MAG as mag
import random
from heapq import heappush, heappop, merge
from networkx.algorithms.centrality.betweenness import _single_source_dijkstra_path_basic, _accumulate_edges, _rescale_e
from networkx.algorithms.centrality.betweenness import _accumulate_endpoints, _accumulate_basic, _rescale
#from networkx.algorithms.shortest_paths.weighted import _weight_function
from MAG import MultiAspectGraph, MultiAspectDiGraph, MultiAspectMultiGraph, MultiAspectMultiDiGraph
from collections import defaultdict
import collections
import functools


FORWARD = 'forward'
REVERSE = 'reverse'

__author__ = """Juliana Z. G. Mascarenhas (julianam@lncc.br), Klaus Wehmuth (wehmuthklaus@gmail.com) and Artur Ziviani (ziviani@lncc.br)"""


all=['sub_edge_dfs',
     'sub_dfs_edges',
     'sub_dfs_tree',
     'sub_dfs_predecessors',
     'sub_dfs_successors',
     'sub_dfs_preorder_nodes',
     'sub_dfs_postorder_nodes',
     'sub_dfs_labeled_edges',
     'sub_generic_bfs_edges',
     'sub_bfs_edges',
     'sub_bfs_tree',
     'sub_bfs_tree',
     'sub_bfs_successors',
     'sub_bfs_beam_edges',
     'sub_shortest_path',
     'sub_all_shortest_paths',
     'sub_shortest_path_length',
     'sub_has_path',
     'sub_single_source_shortest_path',
     'sub_all_pairs_shortest_path',
     'sub_single_source_shortest_path_length',
     'sub_all_pairs_shortest_path_length',
     'sub_predecessor',
     'sub_astar_path',
     'sub_astar_path_length',
     'sub_algorithms.py_shortest_simple_paths',
     'sub_efficiency',
     'sub_global_efficiency',
     'sub_local_reaching_centrality',
     'sub_global_reaching_centrality',
     'sub_betweenness_centrality',
     'sub_edge_betweenness_centrality',
     'sub_closeness_centrality', 
     'sub_dijkstra_path',
     'dijkstra_path_length',
     'sub_single_source_dijkstra_path',
     'sub_single_source_dijkstra_path_length',
     'sub_single_source_dijkstra',
     'sub_multi_source_dijkstra_path',
     'sub_multi_source_dijkstra_path_length',
     'sub_multi_source_dijkstra',
     'sub_dijkstra_predecessor_and_distance',
     'sub_all_pairs_dijkstra',
     'sub_all_pairs_dijkstra_path_length',
     'sub_all_pairs_dijkstra_path',
     'subDeterminedEdgePartition',
     'DFS_Sub']


"""




TRAVERSAL




"""

#Depth First Search on Edges


    
"""
def sub_edge_dfs(G, zeta, source=None, orientation=None, loop=False):
    
      This is the subdetermineted version of dfs algorithm.
      Perform a depth-first-search over the nodes of G and yield the edges in order.
      This may not generate all edges in G (see edge_dfs).

      Parameters
      ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source and target: nodes
            This nodes are the subdeterminated vertices.

        orientation : None | 'original' | 'reverse' | 'ignore' (default: None)
            For directed graphs and directed multigraphs, edge traversals need not
            respect the original orientation of the edges.
            When set to 'reverse' every edge is traversed in the reverse direction.
            When set to 'ignore', every edge is treated as undirected.
            When set to 'original', every edge is treated as directed.
            In all three cases, the yielded edge tuples add a last entry to
            indicate the direction in which that edge was traversed.
            If orientation is None, the yielded edge has no direction indicated.
            The direction is respected, but not reported.

        loop: False by default.
            If is True, the result include the loops.

      Returns
      -------
          generator


    if source is None:
      # all nodes
      nodes = G
      # a list of nodes
    elif type(source) == list:
        nodes = [nd for sub_G in source for nd in G if _sub_node(nd,zeta) == sub_G]
    else:
      # a single node
        nodes = [n for n in G if _sub_node(n,zeta) == source]
    if not nodes:
        return

    directed = G.is_directed()
    kwds = {'data': False}
    if G.is_multigraph() is True:
        kwds['keys'] = True

    # set up edge lookup
    if orientation is None:
        def edges_from(node):
            return iter(G.edges(node, **kwds))
    elif not directed or orientation == 'original':
        def edges_from(node):
            for e in G.edges(node, **kwds):
                yield e + (FORWARD,)
    elif orientation == 'reverse':
        def edges_from(node):
            for e in G.in_edges(node, **kwds):
                yield e + (REVERSE,)
    elif orientation == 'ignore':
        def edges_from(node):
            for e in G.edges(node, **kwds):
                yield e + (FORWARD,)
            for e in G.in_edges(node, **kwds):
                yield e + (REVERSE,)
    else:
        raise nx.NetworkXError("invalid orientation argument.")

    # set up formation of edge_id to easily look up if edge already returned
    if directed:
        def edge_id(edge):
            # remove direction indicator
            return edge[:-1] if orientation is not None else edge
    else:
        def edge_id(edge):
            # single id for undirected requires frozenset on nodes
            return (frozenset(edge[:2]),) + edge[2:]

    # Basic setup
    check_reverse = directed and orientation in ('reverse', 'ignore')

    visited_edges = set()
    visited_sub_edges = set()
    visited_nodes = set()
    edges = {}

    # start DFS
    for start_node in nodes:
        stack = [start_node]
        while stack:
            current_node = stack[-1]
            if current_node not in visited_nodes:
                edges[current_node] = edges_from(current_node)
                visited_nodes.add(current_node)

            try:
                edge = next(edges[current_node])
                edge_sub = _sub_edge(edge, zeta) # added for subdetermination
            except StopIteration:
                # No more edges from the current node.
                stack.pop()
            else:
                edgeid = edge_id(edge)
                edgesubid = edge_id(edge_sub) # added for subdetermination
                if edgeid not in visited_edges:
                    visited_edges.add(edgeid)
                    if edgesubid not in visited_sub_edges: # added for subdetermination
                      visited_sub_edges.add(edgesubid)
                      # Mark the traversed "to" node as to-be-explored.
                      if check_reverse and edge[-1] == REVERSE:
                        stack.append(edge[0])
                      else:
                        stack.append(edge[1])
                      if edge_sub[0] != edge_sub[1] or loop: # added for subdetermination
                        yield edge_sub
"""

#
#
#Depth First Search
#
#

def sub_dfs_edges(G, zeta, source=None,  depth_limit=None ):
  """
    This is the subdetermineted version of dfs edges algorithm.
    Perform a depth-first-search over the nodes of G and yield the edges in order.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

        loop: False by default.
            If is True, the result include the loops.

    Returns
    -------
        edges: generator
               A generator of edges in the depth-first-search.
        
  """
  Gsub = G.subdetermination(zeta)       # added for subdetermination
  if source is None:
      # edges for all components
      nodes = Gsub
  else:
      # edges for components with source
      nodes = [source]
  visited = set()
  if depth_limit is None:
      depth_limit = len(Gsub)
  for start in nodes:
      if start in visited:
          continue
      visited.add(start)
      stack = [(start, depth_limit, iter(Gsub[start]))]
      while stack:
          parent, depth_now, children = stack[-1]
          reachable = set()             # added for subdetermination
          edges = list(sub_generic_bfs_edges(G, zeta, parent))  # added for subdetermination
          for e in edges:               # added for subdetermination
             reachable.add(e[0])
             reachable.add(e[1])          
          try:
              child = next(children)
              if child not in visited and child in reachable:   # child in reachable added for subdetermination
                  yield parent, child
                  visited.add(child)
                  if depth_now > 1:
                      stack.append((child, depth_now - 1, iter(Gsub[child])))
          except StopIteration:
              stack.pop()                

    
def sub_dfs_tree(G, zeta, source=None, depth_limit=None ):
  """

    This is the subdetermineted version of dfs tree algorithm.
    This function will return a generator tree in a graph after the DFS.
    The edges of this graph are subdeterminated using the zeta.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

        loop: False by default.
            If is True, the result include the loops.

    Returns
    -------
    T:  MultiAspectDiGraph
        An oriented tree
        
  """
  T = MultiAspectDiGraph()
  T.add_edges_from(sub_dfs_edges(G, zeta, source,  depth_limit))
  return T
  
    
def sub_dfs_predecessors(G, zeta, source=None, depth_limit=None):
  """
    This is the subdetermineted version of dfs predecessor algorithm.
    This function will return a dict {node:predecessor}

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
        dict

  """
  return {t: s for s, t in sub_dfs_edges(G, zeta, source, depth_limit)}


def sub_dfs_successors(G, zeta, source=None, depth_limit=None):
  """
    This is the subdetermineted version of dfs successors algorithm.
    This function will return a dict {node:sucessors}

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

         depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
        dict

  """
  d = defaultdict(list)
  for s, t in sub_dfs_edges(G, zeta, source=source, depth_limit=depth_limit):
      d[s].append(t)
  return dict(d)


def sub_dfs_preorder_nodes(G, zeta, source=None, depth_limit=None):
  """
  This is the subdetermineted version of dfs preorder nodes algorithm.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

         depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
     nodes: generator
       A generator of nodes in a depth-first-search pre-ordering.   
  """
  edges = sub_dfs_labeled_edges(G, zeta, source=source, depth_limit=depth_limit, loop=False)
  return (v for u, v, d in edges if d == 'forward')


def sub_dfs_postorder_nodes(G, zeta, source=None, depth_limit=None):
  """
    Generate nodes in a depth-first-search post-ordering starting at source.
    Subdetermination version.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

         depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
     nodes: generator
       A generator of nodes in a depth-first-search post-ordering.

  """
  edges = sub_dfs_labeled_edges(G, zeta, source=source, depth_limit=depth_limit, loop=False)
  return (v for u, v, d in edges if d == 'reverse')


def sub_dfs_labeled_edges(G, zeta, source=None, depth_limit=None):
  """
    This is the subdetermineted version of dfs labeled edges algorithm.
    
    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

         depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
    edges: generator
       A generator of triples of the form (*u*, *v*, *d*), where (*u*,
       *v*) is the edge being explored in the depth-first search and *d*
       is one of the strings 'forward', 'nontree', or 'reverse'. A
       'forward' edge is one in which *u* has been visited but *v* has
       not. A 'nontree' edge is one in which both *u* and *v* have been
       visited but the edge is not in the DFS tree. A 'reverse' edge is
       on in which both *u* and *v* have been visited and the edge is in
       the DFS tree.  
  """
  Gsub = G.subdetermination(zeta)       # added for subdetermination
  if source is None:
      # edges for all components
      nodes = Gsub
  else:
      # edges for components with source
      nodes = [source]
  visited = set()
  if depth_limit is None:
      depth_limit = len(Gsub)
  for start in nodes:
      if start in visited:
          continue
      yield start, start, 'forward'
      visited.add(start)
      stack = [(start, depth_limit, iter(Gsub[start]))]
      while stack:
          parent, depth_now, children = stack[-1]
          reachable = set()             # added for subdetermination
          edges = list(sub_generic_bfs_edges(G, zeta, parent))  # added for subdetermination
          for e in edges:               # added for subdetermination
             reachable.add(e[0])
             reachable.add(e[1]) 
          try:
              child = next(children)
              if child in visited and child in reachable:   # child in reachable added for subdetermination
                  yield parent, child, 'nontree'
              elif child in reachable:                      # child in reachable added for subdetermination
                  yield parent, child, 'forward'
                  visited.add(child)
                  if depth_now > 1:
                      stack.append((child, depth_now - 1, iter(Gsub[child])))
          except StopIteration:
              stack.pop()
              if stack:
                  yield stack[-1][0], parent, 'reverse'
      yield start, start, 'reverse'

#
#
#Breadth First Search
#
#

def sub_generic_bfs_edges(G, zeta, source, neighbors=None, depth_limit=None):
    """
    Iterate over edges in a breadth-first search.

    The breadth-first search begins at `source` and enqueues the
    neighbors of newly visited nodes specified by the `neighbors`
    function.

    Parameters
    ----------
    G : NetworkX graph

    zeta: tuple
       Tuple containing the subdetermination to be applied. This tuple is
       formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
       (i.e. removed), while a 1 indicates an aspect to be mantained.
       (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
       where the first and third aspects are manteined and the second
       aspect is subdetermined).

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    neighbors : function
        A function that takes a newly visited node of the graph as input
        and returns an *iterator* (not just a list) of nodes that are
        neighbors of that node. If not specified, this is just the
        ``G.neighbors`` method, but in general it can be any function
        that returns an iterator over some or all of the neighbors of a
        given node, in any order.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    Yields
    ------
    edge
        Edges in the breadth-first search starting from `source`.

    """
    if neighbors is None:
       neighbors = G.neighbors
    neighbors = G.neighbors
    vtx = subDeterminedVertexPartition(G,zeta)
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)

    queue = list()
    queue.append((source, depth_limit))

    while queue:
        parent, depth_now = queue[0]
        queue.pop(0)
        search_queue = list()       
        for nd in vtx[parent]:
            if list(neighbors(nd)) == []:               # TODO
               return
            search_queue.append((nd, depth_limit, neighbors(nd)))
        
        full_visited = set()
        while search_queue:
            full_parent, depth_now, children = search_queue[0]
            try:
                full_child = next(children)
                if full_child not in full_visited:
                    full_visited.add(full_child)
                    child = _sub_node(full_child, zeta)   # added for subdetermination
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, depth_limit))
                        yield parent, child
                    elif child not in visited:
                        search_queue.append((full_child, depth_now - 1, neighbors(full_child)))
            except StopIteration:
                search_queue.pop(0)

def sub_bfs_edges(G, zeta, source, reverse=False, depth_limit=None):
  """
    This function is the subdetermineted version of the original algorithm and will return the edges list as a result of bfs search. 
    The eventuals loops will be disconsidered.

    original: networkx.algorithms.traversal.breadth_first_search

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

        reverse : bool, optional
           If True traverse a directed graph in the reverse direction

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
    edges: generator
       A generator of edges in the breadth-first-search.

     To get the nodes in a breadth-first search order::

        >>> edges = nx.bfs_edges(G, zeta, root)
        >>> nodes = [root] + [v for u, v in edges]  
  """
  if reverse and G.is_directed():
      successors = G.predecessors
  else:
      successors = G.neighbors
  for e in sub_generic_bfs_edges(G, zeta, source, successors, depth_limit):
    yield e
  
    
def sub_bfs_tree(G, zeta, source, reverse=False, depth_limit=None):
  """
    This is the subdeterminetd version of the original algorithm.
    It is a combinational algorithm and will return the subdeterminated result.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

    Returns
    -------
    T: MultiAspectDiGraph
       An oriented tree
    
  """
  T = MultiAspectDiGraph()
  T.add_node(source)
  edges_gen = sub_bfs_edges(G, zeta, source, reverse=reverse, depth_limit=depth_limit)
  T.add_edges_from(edges_gen)
  return T
  

def sub_bfs_predecessors(G, zeta, source, depth_limit=None):
  """
    This is the subdeterminetd version of the original algorithm.
    It is a combinational algorithm and will return the subdeterminated result.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

    Returns
    -------
    pred: iterator
        (node, predecessors) iterator where predecessors is the list of
        predecessors of the node.
    
  """
  for s, t in sub_bfs_edges(G, zeta, source, depth_limit=depth_limit):
      yield (t, s)


def sub_bfs_successors(G, zeta, source, depth_limit=None):
  """
    This is the subdeterminetd version of the original algorithm.
    It is a combinational algorithm and will return the subdeterminated result.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

    Returns
    -------
    succ: iterator
       (node, successors) iterator where successors is the list of
       successors of the node.
    
  """
  parent = source
  children = []
  for p, c in sub_bfs_edges(G, zeta, source, depth_limit=depth_limit):
      if p == parent:
          children.append(c)
          continue
      yield (parent, children)
      children = [c]
      parent = p
  yield (parent, children)

#
#Beam search
#

def sub_bfs_beam_edges(G, zeta, source, value, width=None):
  """
    Iterates over edges in a beam search.

    The beam search is a generalized breadth-first search in which only
    the "best" *w* neighbors of the current node are enqueued, where *w*
    is the beam width and "best" is an application-specific
    heuristic. In general, a beam search with a small beam width might
    not visit each node in the graph.

    Subdetermination version.

    
    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertices.

         value: function
            A function that takes a node of the graph as input and returns a
            real number indicating how "good" it is. A higher value means it
            is more likely to be visited sooner during the search. When
            visiting a new node, only the `width` neighbors with the highest
            `value` are enqueued (in decreasing order of `value`).

         width: int (default = None)
            The beam width for the search. This is the number of neighbors
            (ordered by `value`) to enqueue when visiting each new node.

    Returns
    -------
        list
    
  """
  if width is None:
      width = len(G)

  def successors(v):
    return iter(sorted(G.neighbors(v), key=value, reverse=True)[:width])

  for e in sub_generic_bfs_edges(G, zeta, source, successors):
      yield e



"""


Shortest Paths



"""

def sub_has_path(G, zeta, source, target):
    """
    Return *True* if *G* has a path from *source* to *target*.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source and target: nodes
            These nodes are the subdeterminated vertices.

         Par: dict of edges
            dict to store an Edges Partition that holds repeated edges generated by sub-determination.
            This is necessary because even though the edges are the same, their weight may vary.
            Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
 
    Returns
    -------
        bool - True or False
    
    """
    try:
        sub_shortest_path(G, zeta, source, target)
    except nx.NetworkXNoPath:
        return False
    return True

 
def sub_shortest_path(G, zeta, source=None, target=None, weight=None, method='dijkstra'):
    """Compute shortest paths in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
        Starting node for path. If not specified, compute shortest
        paths for each possible starting node.

    target : node, optional
        Ending node for path. If not specified, compute shortest
        paths to all possible nodes.

    weight : None or string, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.

    method : string, optional (default = 'dijkstra')
        The algorithm to use to compute the path.
        Supported options: 'dijkstra', 'bellman-ford'.
        Other inputs produce a ValueError.
        If `weight` is None, unweighted graph methods are used, and this
        suggestion is ignored.

    Returns
    -------
    path: list or dictionary
        All returned paths include both the source and target in the path.

        If the source and target are both specified, return a single list
        of nodes in a shortest path from the source to the target.

        If only the source is specified, return a dictionary keyed by
        targets with a list of nodes in a shortest path from the source
        to one of the targets.

        If only the target is specified, return a dictionary keyed by
        sources with a list of nodes in a shortest path from one of the
        sources to the target.

        If neither the source nor target are specified return a dictionary
        of dictionaries with path[source][target]=[list of nodes in path].

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.shortest_path(G, source=0, target=4))
    [0, 1, 2, 3, 4]
    >>> p = nx.shortest_path(G, source=0) # target not specified
    >>> p[4]
    [0, 1, 2, 3, 4]
    >>> p = nx.shortest_path(G, target=4) # source not specified
    >>> p[0]
    [0, 1, 2, 3, 4]
    >>> p = nx.shortest_path(G) # source, target not specified
    >>> p[0][4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    There may be more than one shortest path between a source and target.
    This returns only one of them.

    See Also
    --------
    all_pairs_shortest_path()
    all_pairs_dijkstra_path()
    all_pairs_bellman_ford_path()
    single_source_shortest_path()
    single_source_dijkstra_path()
    single_source_bellman_ford_path()
    """
#    if method not in ('dijkstra', 'bellman-ford'):  # bellman-ford unavailable 
#        # so we don't need to check in each branch later
#        raise ValueError('method not supported: {}'.format(method))

    if method not in ('dijkstra'):
        # so we don't need to check in each branch later
        raise ValueError('method not supported: {}'.format(method))
        
    method = 'unweighted' if weight is None else method
    if source is None:
        if target is None:
            # Find paths between all pairs.
            if method == 'unweighted':
                paths = dict(sub_all_pairs_shortest_path(G, zeta))
            elif method == 'dijkstra':
                paths = dict(sub_all_pairs_dijkstra_path(G, zeta, weight=weight))
#            else:  # method == 'bellman-ford':
#                paths = dict(sub_all_pairs_bellman_ford_path(G, zeta, weight=weight))
        else:
            # Find paths from all nodes co-accessible to the target.
#            with nx.utils.reversed(G):
                if method == 'unweighted':
                    paths = sub_single_source_shortest_path(G, zeta, target)
                elif method == 'dijkstra':
                    paths = sub_single_source_dijkstra_path(G, zeta, target, weight=weight)
#                else:  # method == 'bellman-ford':
#                    paths = nx.single_source_bellman_ford_path(G, zeta, target, weight=weight)
                # Now flip the paths so they go from a source to the target.
                for target in paths:
                   paths[target] = list(reversed(paths[target]))
    else:
        if target is None:
            # Find paths to all nodes accessible from the source.
            if method == 'unweighted':
                paths = sub_single_source_shortest_path(G, zeta, source)
            elif method == 'dijkstra':
                paths = sub_single_source_dijkstra_path(G, zeta, source, weight=weight)
#            else:  # method == 'bellman-ford':
#                paths = nx.single_source_bellman_ford_path(G, zeta, source, weight=weight)
        else:
            # Find shortest source-target path.
            if method == 'unweighted':
                paths = sub_dijkstra_path(G, zeta, source, target, weight=1)
            elif method == 'dijkstra':
                paths = sub_dijkstra_path(G, zeta, source, target, weight)
#            else:  # method == 'bellman-ford':
#                paths = nx.bellman_ford_path(G, zeta, source, target, weight)
    return paths


def sub_all_shortest_paths(G, zeta, source, target, weight=None, method='dijkstra'):
  """
    This is a subdetermineted version of the original algorithm.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source and target: nodes
            These nodes are the subdeterminated vertices.

         Par: dict of edges
           dict to store an Edges Partition that holds repeated edges generated by sub-determination.
           This is necessary because even though the edges are the same, their weight may vary.
           Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
 
    Returns
    -------
        list or dict
"""
  method = 'unweighted' if weight is None else method
  if method == 'unweighted':
      pred = sub_predecessor(G, zeta, source)
  elif method == 'dijkstra':
      pred, dist = sub_dijkstra_predecessor_and_distance(G, zeta, source, cutoff=None, weight='weight')
#  elif method == 'bellman-ford':
#      pred, dist = nx.bellman_ford_predecessor_and_distance(G, source, weight=weight)
  else:
      raise ValueError('method not supported: {}'.format(method))

  if target not in pred:
      raise nx.NetworkXNoPath('Target {} cannot be reached'
                                'from Source {}'.format(target, source))

  stack = [[target, 0]]
  top = 0
  while top >= 0:
      node, i = stack[top]
      if node == source:
          yield [p for p, n in reversed(stack[:top + 1])]
      if len(pred[node]) > i:
          top += 1
          if top == len(stack):
              stack.append([pred[node][i], 0])
          else:
              stack[top] = [pred[node][i], 0]
      else:
          stack[top - 1][1] += 1
          top -= 1


def sub_shortest_path_length(G, zeta, source=None, target=None, method='dijkstra'):
  """
    This is a subdetermineted version of the original algorithm.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source and target: nodes
            These nodes are the subdeterminated vertices.

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

        Par: dict of edges
           dict to store an Edges Partition that holds repeated edges generated by sub-determination.
           This is necessary because even though the edges are the same, their weight may vary.
           Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
 
        method : string, optional (default = 'dijkstra')
            The algorithm to use to compute the path.
            Supported options: 'dijkstra', 'bellman-ford'.
            Other inputs produce a ValueError.
            If `weight` is None, unweighted graph methods are used, and this suggestion is ignored.

    Returns
    -------
        number or dict

  """
  _check_parameters(G,zeta,source,target)
  path = sub_shortest_path(G, zeta, source, target, method)
  
  if type(path) is list:
    return len(path)-1 #number of nodes less the source
  else:    #dict
    if source is None and target is None:
      for s in path:
        targets = list(path[s].keys())
        for t in targets:
          length = len(path[s][t])-1
          path[s][t] = length
    else:
      for node in path:
        length = len(path[node])-1
        path[node] = length
    return path

 

"""



  Advanced Interface




"""

def sub_single_source_shortest_path(G, zeta, source, cutoff=None):
  """
    This is a subdeterminated version of the original algorithm.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source: node
            This node is the subdeterminated vertice.

        cutoff : integer, optional
            Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
        dict
    
  """  
  _check_parameters(G, zeta, source)

  if source:
      list_s = [x for x in G if _sub_node(x, zeta) == _sub_node(source, zeta)]
    
  source = list_s[0]
  if cutoff is None:
      cutoff = float('inf')
    
  #declare nextlevel and paths 
  nextlevel = list_s     # list of nodes to check at next level
  paths = {source: [source]}  # paths dictionary  (paths to key from source)
  sub_source = _sub_node(source, zeta)
  sub_paths = {sub_source: [sub_source]}
  return dict(_sub_single_shortest_path(G.adj, zeta, nextlevel, paths, sub_paths, cutoff))
      

def _sub_single_shortest_path(adj, zeta, firstlevel, paths, sub_paths, cutoff):
    """Returns shortest paths

    Shortest Path helper function
    Parameters
    ----------
        adj : dict
            Adjacency dict or view

         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).

        firstlevel : dict
            starting nodes, e.g. {source: 1} or {target: 1}

        paths : dict
            paths for starting nodes, e.g. {source: [source]}

        cutoff : int or float
            level at which we stop the process

        join : function
            function to construct a path from two partial paths. Requires two
            list inputs `p1` and `p2`, and returns a list. Usually returns
            `p1 + p2` (forward from source) or `p2 + p1` (backward from target)
    """    
    level = 0                  # the current level
    nextlevel = firstlevel
    while nextlevel and cutoff > level:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            if v not in paths:
                paths[v] = []
            for w in adj[v]:
                if w not in paths:
#                    if v == ('RAO', 'PTB', 2350) or w == ('RAO', 'PTB', 2350):
#                        p = 0
                    paths[w] = paths[v] + [w]
                    sub_w = _sub_node(w, zeta) #subdeterminated vertice 
                    if sub_w not in sub_paths: 
                        sub_v = _sub_node(v, zeta)
                        if sub_v != sub_w:
                           sub_paths[sub_w] = sub_paths[sub_v] + [sub_w]
                    nextlevel[w] = 1 # inserting the composite vertice on the nextlevel
        level += 1
    return sub_paths


def sub_all_pairs_shortest_path(G, zeta, cutoff=None):
  """
    This is a subdeterminated version of the original algorithm.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).

        cutoff : integer, optional
            Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
        generator
    
  """
  _check_parameters(G, zeta)
  seen = set()
  
  for n in G:
    sub_source = _sub_node(n, zeta)
    if sub_source in seen:
        continue
    else:
        seen.add(sub_source)
        yield (sub_source, sub_single_source_shortest_path(G, zeta, sub_source, cutoff=cutoff))


def sub_single_source_shortest_path_length(G, zeta, source, cutoff=None):
  """
    This is the subdeterminated version of the original algorithm.
    This function only will return the hop count from the source to target.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).
         
         source and target: nodes
            These nodes are the subdeterminated vertices.

        cutoff : integer, optional
            Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
        dict
    
  """
  dictt = sub_single_source_shortest_path(G, zeta, source, cutoff)
  for node in dictt:
    length = len(dictt[node])
    if length>0:
      dictt[node] = length-1
  
  return dictt


def sub_all_pairs_shortest_path_length(G, zeta, cutoff=None):
  """
    This is the subdeterminated version of the original algorithm.

    Parameters
    ----------
         G: the MAG object
           The searching for shortest path occurs for G.
           
         zeta: tuple
            Tuple containing the subdetermination to be applied. This tuple is
            formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
            (i.e. removed), while a 1 indicates an aspect to be mantained.
            (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
            where the first and third aspects are manteined and the second
            aspect is subdetermined).

        cutoff : integer, optional
            Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
        dict
    
  """
  dictt = dict(sub_all_pairs_shortest_path(G, zeta, cutoff))
  
  for node in dictt:
    targets = dictt[node]
    for t in targets:
      dictt[node][t] = len(dictt[node][t])-1
  return dictt
  

def sub_predecessor(G, zeta, source, vtx=None, target=None, cutoff=None, return_seen=None):
    """Returns dict of predecessors for the path from source to all nodes in G


    Parameters
    ----------
    G : NetworkX graph

    zeta: tuple
       Tuple containing the subdetermination to be applied. This tuple is
       formed by 0 and 1, so that 0 indicates an aspect to by subdetermined
       (i.e. removed), while a 1 indicates an aspect to be mantained.
       (e.g. (1,0,1) indicates a subdetermination on an oreder 3 MAG,
       where the first and third aspects are manteined and the second
       aspect is subdetermined).
            
    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path. If provided only predecessors between
       source and target are returned

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.


    Returns
    -------
    pred : dictionary
        Dictionary, keyed by node, of predecessors in the shortest path.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> list(G)
    [0, 1, 2, 3]
    >>> nx.predecessor(G, 0)
    {0: [], 1: [0], 2: [1], 3: [2]}

"""
   
    if vtx is None:
        vtx = subDeterminedVertexPartition(G,zeta)
        
    level = 0                  # the current level
    nextlevel = vtx[source]    # list of nodes to check at next level
    seen = set()
    sub_seen = {source: level}     # level (number of hops) when seen in BFS
    sub_pred = {source: []}        # predecessor dictionary
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        while thislevel:
            v = thislevel[0]
            thislevel.pop(0)
            sub_v = _sub_node(v, zeta)
            for w in G[v]:
                sub_w = _sub_node(w, zeta)
                if w not in seen:
                    seen.add(w)
                    if sub_w not in sub_seen.keys():
                        sub_seen[sub_w] = level
                        sub_pred[sub_w] = [sub_v]
                        nextlevel = nextlevel + vtx[sub_w]
                    elif (sub_seen[sub_w] == level and sub_v not in sub_pred[sub_w]):  # add v to predecessor list if it
                        sub_pred[sub_w].append(sub_v)                                  # is at the correct level                  

        if (cutoff and cutoff <= level):
            break

    if target is not None:
        if return_seen:
            if target not in sub_pred:
                return ([], -1)  # No predecessor
            return (sub_pred[target], sub_seen[target])
        else:
            if target not in sub_pred:
                return []  # No predecessor
            return sub_pred[target]
    else:
        if return_seen:
            return (sub_pred, sub_seen)
        else:
            return sub_pred

"""


Centrality


"""

def _sub_single_source_dijkstra_path_basic(G, zeta, source, weight="weight", vtx=None):
    Gsub = G.subdetermination(zeta)
    if vtx is None:
       vtx = subDeterminedVertexPartition(G,zeta)  

    level = 0  # the current level
    nextlevel = vtx[source]  # list of nodes to check at next level
    seen = {}  # level (number of hops) when seen in BFS
    sub_S = []                  # Nodes seen
    sub_pred = {source: []}  # predecessor dictionary
    sub_sigma = dict.fromkeys(Gsub, 0.0)    # sigma[v]=0 for v in G
    sub_sigma[source] = 1.0
    sub_D = {}                          # Distance
    sub_D[source] = 0.0
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            seen[v] = level
            sub_v = _sub_node(v, zeta)
            sub_S.append(sub_v)
            sub_Dv = sub_D[sub_v]
            sub_sigmav = sub_sigma[sub_v]
            for w in G[v]:
                sub_w = _sub_node(w, zeta)
                if sub_w not in sub_D:
                    sub_D[sub_w] = sub_Dv + 1
                    sub_pred[sub_w] = [sub_v]
                    nextlevel = nextlevel + vtx[sub_w]
                elif sub_D[sub_w] == sub_Dv + 1:  # add v to predecessor list if it
                    sub_pred[sub_w].append(sub_v)  # is at the correct level
                if sub_D[sub_w] == sub_Dv + 1:
                    sub_sigma[sub_w] += sub_sigmav
    return sub_S, sub_pred, sub_sigma


def _sub_single_source_shortest_path_basic(G, zeta, source, vtx=None):
    Gsub = G.subdetermination(zeta)
    if vtx is None:
       vtx = subDeterminedVertexPartition(G,zeta)  

    cont = dict.fromkeys(Gsub, 0.0)
    level = 0  # the current level
    nextlevel = vtx[source]  # list of nodes to check at next level
    seen = {}  # level (number of hops) when seen in BFS
    sub_S = []                  # Nodes seen
    sub_pred = {source: []}  # predecessor dictionary
    sub_sigma = dict.fromkeys(Gsub, 0.0)    # sigma[v]=0 for v in G
    sub_sigma[source] = 1.0
    sub_D = {}                          # Distance
    sub_D[source] = 0.0
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            seen[v] = level
            sub_v = _sub_node(v, zeta)
            sub_S.append(sub_v)
            sub_Dv = sub_D[sub_v]
            sub_sigmav = sub_sigma[sub_v]
            for w in G[v]:
                sub_w = _sub_node(w, zeta)
                if sub_w not in sub_D:
                    sub_D[sub_w] = sub_Dv + 1
                    sub_pred[sub_w] = [sub_v]
                    nextlevel = nextlevel + vtx[sub_w]
                elif sub_D[sub_w] == sub_Dv + 1:  # add v to predecessor list if it
                    sub_pred[sub_w].append(sub_v)  # is at the correct level
                if sub_D[sub_w] == sub_Dv + 1:
                    sub_sigma[sub_w] += sub_sigmav
                    cont[sub_w] += 1
    return sub_S, sub_pred, sub_sigma


  
"""

  Betweenness Centrality

"""

def sub_betweenness_centrality(G, zeta, vtx=None, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    """
    The subdeterminated version of betweenness centrality algorithm.
    This was adapted from the original algorithm.

    Parameters
    ----------
        G: the MAG object
            The searching for shortest path occurs for G.
           
        zeta: is a binary list with the positions related to the aspects of MultiAspectGraph.
            The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.
            The zeta determining the aspect that references the search.
         
        k : int, optional (default=None)
            If k is not None use k node samples to estimate betweenness.
            The value of k <= n where n is the number of nodes in the graph.
            Higher values give better approximation.

        normalized: (bool, optional (default=True))
            Whether to normalize the edge weights by the total sum of edge weights.

        weight: None or string, optional (default=None)
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        endpoints: bool, optional
            If True include the endpoints in the shortest path counts.

        seed integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
            Note that this is only used if k is not None.

    Returns
    -------
        dict
      
    """
    G_sub = G.subdetermination(zeta)
    betweenness = dict.fromkeys(G_sub, 0.0) # b[v]=0 for sub_v in G
    
    if k is None:
        nodes = G_sub
    else:
        random.seed(seed)
        nodes = random.sample(G_sub.nodes(), k)

    if vtx is None:                                             #TUDO ver vtx passado
        vtx = subDeterminedVertexPartition(G,zeta)
        
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma = _sub_single_source_shortest_path_basic(G, zeta, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma = _sub_single_source_dijkstra_path_basic(G, zeta, s, weight)
        # accumulation
        if endpoints:
            betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness = _accumulate_basic(betweenness, S, P, sigma, s)

    if normalized == True:
        betweenness = _rescale(betweenness, len(G), normalized=normalized,
                              directed=G.is_directed(), k=k, endpoints=True)
        eds = G.number_of_edges() ** 2
        betweenness.update({n: betweenness[n]/eds for n in betweenness.keys()})
    return betweenness
  

def sub_edge_betweenness_centrality(G, zeta, vtx=None, k=None, normalized=True, weight=None, seed=None):
    """
      The subdeterminated version of edge betweenness centrality algorithm.
      This was adapted from the original algorithm.

    Parameters
    ----------
        G: the MAG object
            The searching for shortest path occurs for G.
           
        zeta: is a binary list with the positions related to the aspects of MultiAspectGraph.
            The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.
            The zeta determining the aspect that references the search.
         
        k : int, optional (default=None)
            If k is not None use k node samples to estimate betweenness.
            The value of k <= n where n is the number of nodes in the graph.
            Higher values give better approximation.
    
        normalized: (bool, optional (default=True))
            Whether to normalize the edge weights by the total sum of edge weights.

        weight: None or string, optional (default=None)
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        endpoints: bool, optional
            If True include the endpoints in the shortest path counts.

        seed: integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
            Note that this is only used if k is not None.

    Returns
    -------
        dict
      
    """
    G_sub = G.subdetermination(zeta)
    betweenness = dict.fromkeys(G_sub, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G_sub.edges()
    betweenness.update(dict.fromkeys(G_sub.edges(), 0.0))
    
    if k is None:
        nodes = G_sub
    else:
        random.seed(seed)
        nodes = random.sample(G_sub.nodes(), k)

    if vtx is None:                                             #TUDO ver vtx passado
        vtx = subDeterminedVertexPartition(G,zeta)
        
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma = _sub_single_source_shortest_path_basic(G, zeta, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma = _sub_single_source_dijkstra_path_basic(G, zeta, s, weight)
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    # rescaling
        for b in betweenness:
           bb = betweenness[b] / (2 * len(G_sub))
           betweenness[b] = bb
        
    for n in G_sub:  # remove nodes to only return edges
        try:
          del betweenness[n]
        except:
          pass

    if normalized == True:
        betweenness = _rescale(betweenness, len(G), normalized=normalized,
                              directed=G.is_directed(), k=k, endpoints=True)
#        eds = G.number_of_edges() ** 2
#        betweenness.update({n: betweenness[n]/eds for n in betweenness.keys()})
    
#    betweenness = _rescale_e(betweenness, len(G_sub), normalized=normalized,
#                             directed=G.is_directed())
    return betweenness



def sub_closeness_centrality(G, zeta, distance=None):
    r"""Compute closeness centrality for nodes.

    Closeness centrality [1]_ of a node `u` is the reciprocal of the
    average shortest path distance to `u` over all `n-1` reachable nodes.

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`,
    and `n` is the number of nodes that can reach `u`.

    Notice that higher values of closeness indicate higher centrality.

 
    Parameters
    ----------
    G : graph
      A NetworkX graph

    zeta: is a binary list with the positions related to the aspects of MultiAspectGraph.
          The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.
          The zeta determining the aspect that references the search.
            
    u : node, optional
      Return only the value for node u

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations

    wf_improved : bool, optional (default=True)
      If True, scale by the fraction of nodes reachable. This gives the
      Wasserman and Faust improved formula. For single component graphs
      it is the same as the original formula. 

    reverse : bool, optional (default=False)
      If True and G is a digraph, reverse the edges of G, using successors
      instead of predecessors.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality

    Notes
    -----
    The closeness centrality is normalized to `(n-1)/(|G|-1)` where
    `n` is the number of nodes in the connected part of graph
    containing the node.  If the graph is not completely connected,
    this algorithm computes the closeness centrality for each
    connected part separately scaled by that parts size.

    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    References
    ----------
    .. [1] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
    .. [2] pg. 201 of Wasserman, S. and Faust, K.,
       Social Network Analysis: Methods and Applications, 1994,
       Cambridge University Press.
    """
    G_sub = G.subdetermination(zeta)
    closeness = dict.fromkeys(G_sub, 0.0)
    
    if distance is not None:
       path_Lenght = sub_single_source_dijkstra_path_length
    else:
       path_Lenght = sub_single_source_shortest_path_length

    
    for sub_nd in G_sub.nodes:
        pl = path_Lenght(G, zeta, sub_nd, cutoff=None)
        closeness[sub_nd] = sum(pl.values())

    q = 0
        
    """
    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(nx.single_source_dijkstra_path_length, weight=distance)
    else:  # handle either directed or undirected
        if G.is_directed() and not reverse:
            path_length = nx.single_target_shortest_path_length
        else:
            path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes()
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if wf_improved:
                s = (len(sp) - 1.0) / (len(G) - 1)
                closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality
    """
    
"""




  Shortest path algorithms for weighed graphs




"""



#DIJKSTRA 


def sub_dijkstra_path(G, zeta, source, target, weight='weight'):
    """Returns the shortest weighted path from source to target in G.

    Uses Dijkstra's Method to compute the shortest weighted path
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node

    target : node
       Ending node

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    path : list
       List of nodes in a shortest path.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
       If no path exists between source and target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The weight function can be used to include node weights.

    >>> def func(u, v, d):
    ...     node_u_wt = G.nodes[u].get('node_weight', 1)
    ...     node_v_wt = G.nodes[v].get('node_weight', 1)
    ...     edge_wt = d.get('weight', 1)
    ...     return node_u_wt/2 + node_v_wt/2 + edge_wt

    In this example we take the average of start and end node
    weights of an edge and add it to the weight of the edge.

    See Also
    --------
    sub_bidirectional_dijkstra(), sub_bellman_ford_path()
    """
    (_, path) = sub_single_source_dijkstra(G, zeta, source, target=target, weight=weight)
    return path
  

def sub_dijkstra_path_length(G, zeta, source, target, weight='weight'):
    """Returns the shortest weighted path length in G from source to target.

    Uses Dijkstra's Method to compute the shortest weighted path length
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       starting node for path

    target : node label
       ending node for path

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    length : number
        Shortest path length.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    sub_bidirectional_dijkstra(), sub_bellman_ford_path_length()

    """
    
    if source == target:
        return 0
    weight = _weight_function(G, weight)
    length = _sub_dijkstra(G, zeta, source, weight, target=target)
    try:
        return length[target]
    except KeyError:
        raise nx.NetworkXNoPath(
            "Node %s not reachable from %s" % (target, source))
        
        
def sub_single_source_dijkstra_path(G, zeta, source, cutoff=None, weight='weight'):
    """Find shortest weighted paths in G from a source node.

    Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    paths : dictionary
       Dictionary of shortest path lengths keyed by target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> path=nx.single_source_dijkstra_path(G,0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    sub_single_source_dijkstra(), sub_single_source_bellman_ford()

    """
    return sub_multi_source_dijkstra_path(G, zeta, {source}, cutoff=cutoff, weight=weight)


def sub_single_source_dijkstra_path_length(G, zeta, source, cutoff=None, weight='weight'):
    """Find shortest weighted path lengths in G from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length from source.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    sub_single_source_dijkstra(), sub_single_source_bellman_ford_path_length()

    """
    return sub_multi_source_dijkstra_path_length(G, zeta, {source}, cutoff=cutoff, weight=weight)


def sub_single_source_dijkstra(G, zeta, source, target=None, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list.
       If target is None, paths and lengths to all nodes are computed.
       The return value is a tuple of two dictionaries keyed by target nodes.
       The first dictionary stores distance to each target node.
       The second stores the path to each target node.
       If target is not None, returns a tuple (distance, path), where
       distance is the distance from source to target and path is a list
       representing the path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    sub_single_source_dijkstra_path()
    sub_single_source_dijkstra_path_length()
    sub_single_source_bellman_ford()
    """
    return sub_multi_source_dijkstra(G, zeta, {source}, cutoff=cutoff, target=target, weight=weight)
  
  
def sub_multi_source_dijkstra_path(G, zeta, sources, cutoff=None, weight='weight'):
    """Find shortest weighted paths in G from a given set of source
    nodes.

    Compute shortest path between any of the source nodes and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
   cutoff : integer or float, optional
      Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    paths : dictionary
       Dictionary of shortest paths keyed by target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    sub_multi_source_dijkstra(), sub_multi_source_bellman_ford()

    """  
    (_, path) = sub_multi_source_dijkstra(G, zeta, sources, cutoff=cutoff, weight=weight)
    return path

  
def sub_multi_source_dijkstra_path_length(G, zeta, sources, cutoff=None, weight='weight'):
    """Find shortest weighted path lengths in G from a given set of
    source nodes.

    Compute the shortest path length between any of the source nodes and
    all other reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition. 
      
    cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length to nearest source.


    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    sub_multi_source_dijkstra()

    """  
    if not sources:
        raise ValueError('sources must not be empty')
    weight = _weight_function(G, weight)
    return _sub_dijkstra_multisource(G, zeta, sources, weight, cutoff=cutoff)
    

def sub_multi_source_dijkstra(G, zeta, sources, target=None, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Uses Dijkstra's algorithm to compute the shortest paths and lengths
    between one of the source nodes and the given `target`, or all other
    reachable nodes if not specified, for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    Par: dict of edges
      dict to store an Edges Partition that holds repeated edges generated by sub-determination.
      This is necessary because even though the edges are the same, their weight may vary.
      Therefore, in order to calculate the shortest path we use the smallest weight on each partitition.       

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list
       If target is None, returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from one of the source nodes.
       The second stores the path from one of the sources to that node.
       If target is not None, returns a tuple of (distance, path) where
       distance is the distance from source to target and path is a list
       representing the path from source to target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    sub_multi_source_dijkstra_path()
    sub_multi_source_dijkstra_path_length()

    """

    if not sources:
        raise ValueError('sources must not be empty')
    if target in sources:
        return (0, [target])
    
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _sub_dijkstra_multisource(G, zeta, sources, weight, paths=paths, cutoff=cutoff, target=target)
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))
  
  
def _sub_dijkstra(G, zeta, source, weight, pred=None, paths=None, cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths from a
    single source.

    This is a convenience function for :func:`_dijkstra_multisource`
    with all the arguments the same, except the keyword argument
    `sources` set to ``[source]``.

    """
    return _sub_dijkstra_multisource(G, zeta, [source], weight, pred=pred, paths=paths, cutoff=cutoff, target=target)
  
  


def _sub_dijkstra_multisource(G, zeta, sources, weight, pred=None, paths=None, cutoff=None, target=None):
  """Uses Dijkstra's algorithm to find shortest weighted paths

  Parameters
  ----------
  G : NetworkX graph

  sources : non-empty iterable of nodes
    Starting nodes for paths. If this is just an iterable containing
    a single node, then all paths computed by this function will
    start from that node. If there are two or more nodes in this
    iterable, the computed paths may begin from any one of the start
    nodes.

  weight: function
    Function with (u, v, data) input that returns that edges weight
    
  Par: dict of edges
    dict to store an Edges Partition that holds repeated edges generated by sub-determination.
    This is necessary because even though the edges are the same, their weight may vary.
    Therefore, in order to calculate the shortest path we use the smallest weight on each partitition.

  pred: dict of lists, optional(default=None)
    dict to store a list of predecessors keyed by that node
    If None, predecessors are not stored.

  paths: dict, optional (default=None)
    dict to store the path list from source to each node, keyed by node.
    If None, paths are not stored.

  target : node label, optional
    Ending node for path. Search is halted when target is found.

  cutoff : integer or float, optional
    Depth to stop the search. Only return paths with length <= cutoff.

  Returns
  -------
  distance : dictionary
    A mapping from node to shortest distance to that node from one
    of the source nodes.

  Raises
  ------
  NodeNotFound
    If any of `sources` is not in `G`.

  Notes
  -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

  """
  
  G_succ = G._succ if G.is_directed() else G._adj
  sub_sources = [nd for sub_G in sources for nd in G if _sub_node(nd,zeta) == sub_G] # added for subdetermination
  Gsub = G.subdetermination(zeta)
  
  push = heappush
  pop = heappop
  dist = {}  # dictionary of final distances
  sub_dist = {}   # added for subdetermination
  seen = {}
  sub_seen = {}   # added for subdetermination
  # fringe is heapq with 3-tuples (distance,c,node)
  # use the count c to avoid comparing nodes (may not be able to)
  c = count()
  fringe = []
  for source in sub_sources:
    if source not in G:
      raise nx.NodeNotFound("Source {} not in G".format(source))
    seen[source] = 0
    sub_seen[_sub_node(source,zeta)] = 0   # added for subdetermination
    push(fringe, (0, next(c), source))
  while fringe:
    (d, _, v) = pop(fringe)
    sub_v = _sub_node(v,zeta)   # added for subdetermination  
    if v in dist:
      continue  # already searched this node.
    dist[v] = d
    sub_dist[sub_v] = d 
    if sub_v == target:
      break
    for u, e in G_succ[v].items():
      sub_u = _sub_node(u,zeta)   # added for subdetermination
      if sub_v == sub_u:          # added for subdetermination
          cost = 0
      else:   
          cost = Gsub.get_edge_data(sub_v, sub_u)['weight']      #weight(sub_v, sub_u, e)
      if cost is None:
        continue
      vu_dist = dist[v] + cost
      if cutoff is not None:
        if vu_dist > cutoff:
          continue
      if u in dist:
        if vu_dist < dist[u]:
          raise ValueError('Contradictory paths found:',
                            'negative weights?')
      elif u not in seen or vu_dist < seen[u]:
        seen[u] = vu_dist
        push(fringe, (vu_dist, next(c), u))
        if sub_u not in sub_seen or vu_dist < seen[u]:   # added for subdetermination
          sub_seen[sub_u] = vu_dist
          sub_dist[sub_u] = vu_dist  # added for subdetermination
          if paths is not None:
            paths[sub_u] = paths[sub_v] + [sub_u]
          if pred is not None:
            pred[sub_u] = [sub_v]
      elif vu_dist == seen[u]:
        if pred is not None:
          if sub_u not in sub_seen:   # added for subdetermination
            pred[sub_u].append(sub_v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
  return sub_seen    # modified for subdetermination

  
def sub_dijkstra_predecessor_and_distance(G, zeta, source, cutoff=None, weight='weight'):
    """Compute weighted shortest path length and predecessors.

    Uses Dijkstra's Method to obtain the shortest weighted paths
    and return dictionaries of predecessors for each node and
    distance for each node from the `source`.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    pred, distance : dictionaries
       Returns two dictionaries representing a list of predecessors
       of a node and the distance to each node.
       Warning: If target is specified, the dicts are incomplete as they
       only contain information for the nodes along a path to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The list of predecessors contains more than one element only when
    there are more than one shortest paths to the key node.

    """
    weight = _weight_function(G, weight)
    pred = {source: []}  # dictionary of predecessors
    dist = _sub_dijkstra(G, zeta, source, weight, pred=pred, cutoff=cutoff)
    return (pred, dist)
  
 
  
def sub_all_pairs_dijkstra(G, zeta, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    Par: dict of edges
       dict to store an Edges Partition that holds repeated edges generated by sub-determination.
       This is necessary because even though the edges are the same, their weight may vary.
       Therefore, in order to calculate the shortest path we use the smallest weight on each partitition.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edge[u][v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Yields
    ------
    (node, (distance, path)) : (node obj, (dict, dict))
        Each source node has two associated dicts. The first holds distance
        keyed by target and the second holds paths keyed by target.
        (See single_source_dijkstra for the source/target node terminology.)
        If desired you can apply `dict()` to this function to create a dict
        keyed by source node to the two dicts.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The yielded dicts only have keys for reachable nodes.
    """
    nodes = {_sub_node(n, zeta) for n in G}
    for n in nodes:
      dist, path = sub_single_source_dijkstra(G, zeta, n, cutoff=cutoff, weight=weight)
      yield (n, (dist, path))
      
  
def sub_all_pairs_dijkstra_path_length(G, zeta, cutoff=None, weight='weight'):
    """Compute shortest path lengths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    distance : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path length as the key value.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionary returned only has keys for reachable node pairs.
    """
    nodes = {_sub_node(n, zeta) for n in G}
    length = sub_single_source_dijkstra_path_length
    for n in nodes:
        yield (n, length(G, zeta, n, cutoff=None, weight=weight))            

            
def sub_all_pairs_dijkstra_path(G, zeta, cutoff=None, weight='weight'):
    """Compute shortest paths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    distance : dictionary
       Dictionary, keyed by source and target, of shortest paths.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    floyd_warshall(), all_pairs_bellman_ford_path()

    """
    nodes = {_sub_node(n, zeta) for n in G}
    path = sub_single_source_dijkstra_path
    for n in nodes:
      yield (n, path(G, zeta, n, cutoff=cutoff, weight=weight))



"""



  SUPORT ALGORITHMS




"""

def _weight_function(G, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
#    if G.is_multigraph():
#        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)

"""
def _weight_function(Edge_Partition, weight):
    Returns a function that returns the weight of an edge.

    The returned function is specifically suitable for input to
    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

    Parameters
    ----------
    Par : Edges Partition.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.

    If `weight` is not callable, the minimum edge weight over all parallel 
    edges is returned. 
    
    Usage: 
         1 - Asign weight funtion ->  weight = _weight_function(Edge_Partition, "weight")
         2 - Use funtion -> w = weight(u,v, Edge_Partition)

    """
#    P = Edge_Partition
#    if callable(weight):
#        return weight
#    return lambda u, v, _: min(P[u,v])


def subDeterminedEdgePartition(G,zeta):
   """Returns a partition of subdetermined edges, containing the wieghts
   of every edge in all equivalence class.
   The tuple zeta has the same number of
   elements as the number of aspects on the original MAG H, 
   and for each aspect carries a value 1 or 0, so that only 
   the aspects marked with 1 will remain on the resulted MAG. 
   The resulting sub-determined MAG has the edges of the original 
   MAG projected over the reduced aspect structure given by the 
   sub-determination.
   input: MAG H and sub-determination tuple zeta
   output: Sub-determined MAG Hz
   """
   Par = collections.defaultdict(list)
   asps = list(zeta) # + list(zeta)
   n = len(asps)
   E = G.edges
   for e in E:
      ez = [[],[]]
      for i in range(n):
         if asps[i] != 0:
            ez[0].append(e[0][i])
            ez[1].append(e[1][i])            
      if ez[0] != ez[1]: 
         tez = (tuple(ez[0]),tuple(ez[1]))
         w = G[e[0]][e[1]]["weight"]
         Par[tez].append(w)
   return Par

def subDeterminedVertexPartition(G,zeta):
   Par = collections.defaultdict(list)
   V = G.nodes
   for v in V:
       sub_v = _sub_node(v, zeta)
       Par[sub_v].append(v)
   return Par
   

def _check_parameters(G, zeta, source=None, target=None):
  if G.is_mag == False:
      msg = 'Implemented only for MAG'
      raise TypeError(msg)
  
  if len(G._aspect) != len(zeta):
    raise ValueError("The list of aspects didn't mach. \
                      Number of aspects in zeta list is diferent on the MAG aspects list")
  nonzeros = zeta.count(1)
  
  #aspects that wasn`t be subdeterminated
  aspect = [G._aspect[i] for i in range(len(zeta)) if zeta[i]== 1]
  
  if source != None:
      if nonzeros != len(source):
          raise ValueError("The zeta and source don't match")
      
      for i in range(len(source)):
        answer = True if source[i] in aspect[i] else False
        if answer == False:
            msg = 'There is some error in one or more these parameters: {} or {}.'
            raise ValueError(msg.format(zeta,source))
            
  if target != None:
      if nonzeros != len(target):
          raise ValueError("The zeta and source don't match")

      for i in range(len(target)):
        answer = True if target[i] in aspect[i] else False
        if answer == False:
            msg = 'There is an error in one or more these parameters: {} or {}.'
            raise ValueError(msg.format(zeta,target))
  

def _list_existing_nodes(G, zeta, source, target):
  
  list_target = []  # list()
  list_source = []  # list()

  nodes = list(G)
 # make it a list
  if source is None:
      src = []
  else:
      src = [source]
  if target is None:
      tgt = []
  else:
      tgt = [target]

  #get all possible nodes (source, target)
  for i in range(len(nodes)):
      n = _sub_node(nodes[i], zeta)
      if n in src:
         list_source.append(nodes[i])
      if n in tgt:
          list_target.append(nodes[i])
  
  if not list_target:
    list_target = None
  
  if not list_source:
    list_source = None
    
  return list_source, list_target
  

def _new_source(G,zeta,source):
  lenz = len(zeta)
  nodes = list(G)
  for i in range(len(nodes)):
    atual = tuple([nodes[i][j] for j in range(lenz) if zeta[j] == 1])
    if atual == source:
      return nodes[i]    



# The subdetermination of a list of edges with loops
"""
def _sub_list(multi, list_edges, zeta, loop):
  result = [] # list()
  sub_list = [_sub_edge(e, zeta) for e in list_edges]  
  
  if multi == False:
    Color = dict.fromkeys(sub_list, 0)
    for e in sub_list:
      if Color[e] == 0 and (e[0] !=e [1] or loop==True):
        result.append(e)
        Color[e] = 1
  else:
    for e in sub_list:
      if e[0] != e[1] or loop==True:
        result.append(e)
  
  return result
"""

def _sub_edge(edge,zeta):
  return (_sub_node(edge[0], zeta),_sub_node(edge[1], zeta))


def _sub_list_nodes(listn, zeta):
  sub_listn = [] # list()
  Color = dict.fromkeys([_sub_node(node, zeta) for node in listn], 0)
  for n in listn:
    n = _sub_node(n, zeta)
    if Color[n] == 0:
      sub_listn.append(n)
      Color[n] = 1
  
  return sub_listn


def _sub_node(node,zeta):
  return tuple( [node[j] for j in range (len(zeta)) if zeta[j] == 1])

"""
def _id_function(function):
  Type = str(type(function))
  if Type == "<class 'function'>":
    return _update_weight_function
  elif Type == "<class 'builtin_function_or_method'>":
    return _update_weight_builtin
  else:
    return _update_weight_type

def _update_weight_builtin(function, old_values, values):
  #function: sum, max ... th built-in functions that recivied an interable as parameter
  return function([old_values,values])
  
def _update_weight_function(function, old_values, values):
    #function written by the user
    return function(old_values,values)
  
def _update_weight_type(function, old_values, values):
  #function as a list, set 
  basic = [float,int]
  if type(old_values) in basic:
    return function([old_values, values])
  else:
    return function(list(old_values)+[values])
"""
  
def _existing_nodes(nodes, zeta, source):
  
  list_source = []  # list()
  #get all possible nodes (source, target)
  for i in range(len(nodes)):
      n = _sub_node(nodes[i], zeta)
      if source == n:
          list_source.append(nodes[i])
  if not list_source:
    list_source = None
    
  return list_source
