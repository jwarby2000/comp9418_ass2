# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 00:33:18 2022

@author: Jake
"""

import graphviz
# Priority queue for Prim algorithm
import heapq as pq
import copy

class Graph:       
    def __init__(self, adj_list=None):
        self.adj_list = dict()
        if adj_list is not None:
            self.adj_list = adj_list.copy() # dict with graph's adjacency list
        self.colour = dict()
        self.edge_weights = dict() # maps a tuple (node1, node2) to a number

    def __len__(self):
        '''
        return the number of nodes in the graph
        '''
        return len(self.adj_list.keys())

    def __iter__(self):
        '''
        Let a user iterate over the nodes of the graph, like:
        for node in graph:
            print(node)
        '''
        return iter(self.adj_list.keys())
    
    def children(self, node):
        '''
        Return a list of children of a node
        '''
        return self.adj_list[node]

    def add_node(self, name):
        '''
        This method adds a node to the graph.
        '''
        if name not in self.adj_list:
            self.adj_list[name] = []

    def remove_node(self, name):
        '''
        This method removes a node, and any edges to or from the node
        '''
        for node in self.adj_list.keys():
            if name in self.adj_list[node]:
                self.adj_list[node].remove(name)
        del self.adj_list[name]

    def add_edge(self, node1, node2, weight=1, directed=True):
        '''
        This function adds an edge. If directed is false, it adds an edge in both directions
        '''
        # in case they don't already exist, add these nodes to the graph
        self.add_node(node1)
        self.add_node(node2)
        
        self.adj_list[node1].append(node2)
        self.edge_weights[(node1,node2)] = weight
        
        if not directed:
            self.adj_list[node2].append(node1)
            self.edge_weights[(node2,node1)] = weight
        
    def show(self, directed=True, positions=None):
        """
        Prints a graphical visualisation of the graph usign GraphViz
        arguments:
            `directed`, True if the graph is directed, False if the graph is undirected
            `pos: dictionary`, with nodes as keys and positions as values
        return:
            GraphViz object
        """
        if directed:
            dot = graphviz.Digraph(engine="neato", comment='Directed graph')
        else:
            dot = graphviz.Graph(engine="neato", comment='Undirected graph', strict=True)        
        dot.attr(overlap="false", splines="true")
        for v in self.adj_list:
            if positions is not None:
                dot.node(str(v), pos=positions[v])
            else:
                dot.node(str(v))
        for v in self.adj_list:
            for w in self.adj_list[v]:
                dot.edge(str(v), str(w))

        return dot
    
    def _dfs_r(self, v): # This is the main DFS recursive function
        """
        argument 
        `v`, next vertex to be visited
        `colour`, dictionary with the colour of each node
        """
        print('Visiting: ', v)
        self.colour[v] = 'grey' # Visited vertices are coloured 'grey'
        for w in self.adj_list[v]: # Let's visit all outgoing edges from v
            if self.colour[w] == 'white': # To avoid loops, we check if the next vertex hasn't been visited yet
                self._dfs_r(w)
        self.colour[v] = 'black' # When we finish the for loop, we know we have visited all nodes from v. It is time to turn it 'black'

    def dfs(self, start): # This is an auxiliary DFS function to create and initialize the colour dictionary
        """
        argument 
        `start`, starting vertex
        """    
        self.colour = {node: 'white' for node in self.adj_list.keys()} # Create a dictionary with keys as node numbers and values equal to 'white'
        self._dfs_r(start)
        return self.colour # We can return colour dictionary. It is useful for some operations, such as detecting connected components
    
    def dfs_all(self): # This is an auxiliary DFS function to create and initialize the colour dictionary
        """
        argument 
        `start`, starting vertex
        """    
        self.colour = {node: 'white' for node in self.adj_list.keys()} # Create a dictionary with keys as node numbers and values equal to 'white'
        for start in self.colour.keys():
            if self.colour[start] == 'white':
                self._dfs_r(start)
                
    def _find_cycle_r(self, v):
        """
        argument 
        `v`, next vertex to be visited
        """      
        print('Visiting: ', v)
        
        self.colour[v] = 'grey' # Visited vertices are coloured 'grey'
        for w in self.adj_list[v]: # Let's visit all outgoing edges from v
            if self.colour[w] == 'white': # To avoid loops, we check if the next vertex hasn't been visited yet
                if self._find_cycle_r(w):
                    return True
            elif self.colour[w] == 'grey':
                print(v, w, 'Cycle Detected')
                return True
        self.colour[v] = 'black' # When we finish the for loop, we know we have visited all nodes from v. It is time to turn it 'black'
    

        return False

    # This is an auxiliary function to create and initialize the colour dictionary    
    def find_cycle(self):
        """
        argument 
        `v`, starting vertex
        """        
        self.colour = dict([(node, 'white') for node in self.adj_list.keys()])
        for start in self.colour.keys():
            if self.colour[start] == 'white':
                if self._find_cycle_r(start):
                    return True
        return False
    
    def _topological_sort_r(self, v):
        """
        argument 
        `v`, current vertex
        """
        
        print('Visiting: ', v)
        self.colour[v] = 'grey' # Visited vertices are coloured 'grey'
        for w in self.adj_list[v]: # Let's visit all outgoing edges from v
            if self.colour[w] == 'white': # To avoid loops, we check if the next vertex hasn't been visited yet
                self._topological_sort_r(w)
        self.colour[v] = 'black' # When we finish the for loop, we know we have visited all nodes from v. It is time to turn it 'black'
        print('Pushing ', v, ' to stack')
        self.stack.append(v)
        
        
        
    # This is main function that prepares for the recursive function. It first colours all nodes as 'white' and call the
    # recursive function for an arbitrary node. When the recursive function returns, if we have any remaining 'white'
    # nodes, we call the recursive function again for these nodes.
    def topological_sort(self, plot = True):
        """
        argument 
        `G`, an adjacency list representation of a graph
        return a list with the topological order of the graph G
        """
        # We start with an empty stack
        self.stack = []
        # Colour is dictionary that associates node keys to colours. The colours are 'white', 'grey' and 'black'.
        self.colour = {node: 'white' for node in self.adj_list.keys()}
        # We call the recursive function to visit a first node. When the function returns, if there are any white 
        # nodes remaining, we call the function again for these white nodes
        for start in self.adj_list.keys():
            # If the node is 'white' we call the recursive function to vists the nodes connected to it in DFS order
            if self.colour[start] == 'white':
                # This is a call to topologicalSort_r
                self._topological_sort_r(start)
        # We need to reverse the list, we use a little trick with list slice
        self.stack = self.stack[::-1]
        
        if plot:
            # We use the neato engine since it allow to position the nodes
            dot = graphviz.Digraph(engine="neato", comment='Topological sort')
            # This line will avoid the edges to cross over the nodes
            dot.attr(overlap="false", splines="true")
            
            for i, v in enumerate(self.stack, 1):
                dot.node(str(v), pos=str(i)+',0!')

            # Create edges
            for v in self.stack:
                for w in self.adj_list[v]:
                    dot.edge(str(v), str(w))  
            return dot # How can I return self.stack as well as show this graph?
        
        else:
            return self.stack
        
    def transpose(self):
        """
        argument 
        `G`, an adjacency list representation of a graph
        """      
        gt = dict((v, []) for v in self.adj_list)
        for v in self.adj_list:
            for w in self.adj_list[v]:
                gt[w].append(v)
                
        return Graph(gt)
    
    def prim(self, start, minimum = True,  plot = True, directed = True):
        """
        argument 
        `start`, start vertex
        """      
        # Intialise set 'visited' with vertex s
        visited = {start} # Set is used here to ensure no duplicate verticies (this would create a cycle)
        # Initialise priority queue Q with an empty list
        Q = []
        # Initilise list tree with empty Graph object. This object will have the MST at the end of the execution
        tree = Graph()
        # Initialise the priority queue Q with outgoing edges from s
        for e in self.adj_list[start]:
            # There is a trick here. Python prioriy queues accept tuples but the first entry of the tuple must be the priority value
            if minimum:
                pq.heappush(Q, (self.edge_weights[(start,e)], start, e))
            else:
                pq.heappush(Q, (-self.edge_weights[(start,e)], start, e))
        while len(Q) > 0:
            # Remove element from Q with the smallest weight
            weight, v, u = pq.heappop(Q)
            # If the node is already in 'visited' we cannot include it in the MST since it would create a cycle
            if u not in visited:
                # Let's grow the MST by inserting the vertex in visited
                visited.add(u)
                # Also we insert the edge in tree
                tree.add_edge(v,u,weight = weight)
                # We iterate over all outgoing edges of u
                for e in self.adj_list[u]:
                    # We are interested in edges that connect to vertices not in 'visited' and with smaller weight than known values stored in a
                    if e not in visited:
                        # Edge e is of interest, let's store in the priority queue for future analysis
                        if minimum:
                            pq.heappush(Q, (self.edge_weights[(u,e)], u, e))    
                        else:
                            pq.heappush(Q, (-self.edge_weights[(u,e)], u, e)) 
        
        if plot:
            return tree.show(directed = directed)
                            
        return tree
    
    def strongly_connected(self, start):
        # First search
        first_search = self.dfs(start)
        if not any(v == 'white' for v in iter(first_search.values())): #If false, then all nodes are searched in the first search
            print("First search successful")    
            # Second Search
            G_T = self.transpose()
            second_search = G_T.dfs(start)
            if not any(v == 'white' for v in iter(second_search.values())):
                print("Second Search Successful!")
                return True
            else:
                print("Second search unsuccessful")
                return False
            
        else:
            print("First search unsuccessful")
            return False
        
    def copy(self):
        return copy.deepcopy(self)