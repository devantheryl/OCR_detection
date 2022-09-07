# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:41:41 2022

@author: LDE
"""


# Python3 program to find all the reachable nodes
# for every node present in arr[0..n-1]
from collections import deque
 

class Graph:
    
    def __init__(self, V):
        self.adj = [[] for i in range(V)]
        self.visited = [0 for i in range(V)]
        
    def addEdge(self,v, w):
         
        
        self.adj[v].append(w)
        self.adj[w].append(v)
     
    def BFS(self,componentNum, src):
         
        
         
        # Mark all the vertices as not visited
        # Create a queue for BFS
        #a =  visited
        queue = deque()
     
        queue.append(src)
     
        # Assign Component Number
        self.visited[src] = 1
     
        # Vector to store all the reachable
        # nodes from 'src'
        reachableNodes = []
        #print("0:",visited)
     
        while (len(queue) > 0):
             
            # Dequeue a vertex from queue
            u = queue.popleft()
     
            reachableNodes.append(u)
     
            # Get all adjacent vertices of the dequeued
            # vertex u. If a adjacent has not been visited,
            # then mark it visited and enqueue it
            for itr in self.adj[u]:
                if (self.visited[itr] == 0):
                     
                    # Assign Component Number to all the
                    # reachable nodes
                    self.visited[itr] = 1
                    queue.append(itr)
     
        return reachableNodes
     
    # Display all the Reachable Nodes
    # from a node 'n'
    def displayReachableNodes(self,m):
         
        for i in m:
            print(i, end = " ")
     
        print()
     
    def findReachableNodes(self,arr, n):
         
        
         
        # Get the number of nodes in the graph
     
        # Map to store list of reachable Nodes for a
        # given node.
        a = []
     
        # Initialize component Number with 0
        componentNum = 0
     
        # For each node in arr[] find reachable
        # Nodes
        reachableNodes = {}
        for i in range(n):
            u = arr[i]
     
            # Visit all the nodes of the component
            if (self.visited[u] == 0):
                componentNum += 1
     
                # Store the reachable Nodes corresponding
                # to the node 'i'
                a = self.BFS(componentNum, u)
     
            # At this point, we have all reachable nodes
            # from u, print them by doing a look up in map m.
            reachableNodes[u] = a
            
        return reachableNodes
 
