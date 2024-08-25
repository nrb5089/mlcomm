#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:46:51 2019

@author: nate
"""

import numpy as np
from copy import deepcopy as dc


class Binary_Tree:
    def __init__(self,depth,peak = 0):
        self.L = depth 
        self.branches = list() 
        self.num_nodes = 0
        self.peak = peak
        midx = 0
        for l in range(0,self.L+1):
            for k in range(0,int(2**l)):
                self.branches.append(Node(midx,
                                          self.L,
                                          self.peak,
                                          layer = l,
                                          layer_index = k,
                                          parent = 2**(l-1) -1 + np.floor(k/2).astype('int'),
                                          children = (2**(l+1) -1 + 2*k, 2**(l+1) -1 + 2*k + 1)))
                midx += 1
                self.num_nodes += 1
        self.total_nodes = dc(midx)
        
        #Calculate ancestors
        for midx in range(self.total_nodes):
            self.branches[midx].ancestors = []
            parent = np.array([self.branches[midx].parent])
            while parent.size != 0:
                self.branches[midx].ancestors.append(parent[0])
                parent = np.array([self.branches[parent[0]].parent])
                
                
    def get_idx(self,l,k):
        return int(2**l - 1 + k)
    
    def get_node(self,l,k):
        return self.branches[self.get_idx(l,k)]
    
class Node: 
    def __init__(self,
                 master_index,
                 num_layers,
                 peak,
                 layer,
                 layer_index,
                 parent,
                 children):
        self.midx = master_index             
        self.indices = (layer,layer_index)  ##index 0, .., 2^L-1 # index 0,...,2^layer-1
        
        if layer == peak:                      #Top layer has no parents and no siblings =(
#            self.parent = None
            self.parent = []
#            self.sibling = None
            self.sibling = []
        else:
            self.parent = parent
            if np.mod(layer_index,2) == 0:
                self.sibling = master_index + 1
                if layer_index !=0:
                    self.cousin = [master_index  - 1]
                else:
                    self.cousin = []
            else:
                self.sibling = master_index - 1
                if layer_index == 2**layer-1:
                    self.cousin = []
                else:
                    self.cousin = [master_index + 1]
                
            #ancestors[0] is the parent node
            self.ancestors = list([parent])
            
            
                
        
        if layer == num_layers:             #Last layer, not by choice, has no children
#            self.children = None
            self.children = []
        else:
            self.children = children        #Tuple or array of 2 elements
            
        
        #Calculate the indices of node's final descendents
        self.num_final_descendents = int( 2**num_layers / 2**layer )
        self.final_descendents = np.arange(layer_index * 2**num_layers / 2**layer,(layer_index + 1) * 2**num_layers / 2**layer).astype('int')
   

if __name__ == '__main__':
    mytree = Binary_Tree(7)      
    mybranches = mytree.branches