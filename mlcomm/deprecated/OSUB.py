#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:47:00 2019

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
import sys
sys.path.insert(0,'/media/nate/New Volume/DDrive/Education/PhD/Dissertation/Modeling/Multi-User/BA/mlcomm/mlcomm')
import array_play_codebooks
import binary_tree
import rician
from copy import deepcopy as dc
#import cdl
plt.close('all')


class OSUB:
    '''Graphical Verion'''
    def __init__(self,
                 GetReward,
                 Channel,
                 max_resolution,
                 start_layer = 1,
                 delta = 1,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.L = int(np.log2(self.N))           #Number of layers
        self.delta = delta
        self.seed = seed
        np.random.seed(seed = seed)
        
        self.l0 = start_layer #fixed
        max_node_midx = np.array([2,6,14,30,62])[self.l0-1]

        
        #Set custom tree attributes
        setattr(binary_tree.Node,'update', update)
        setattr(binary_tree.Node,'info_tuple',info_tuple)
        setattr(binary_tree.Node,'UCB',np.inf)
        setattr(binary_tree.Node,'LCB',-np.inf)
        setattr(binary_tree.Node,'Nt',0)
        setattr(binary_tree.Node,'mu_hat',0)
        setattr(binary_tree.Node,'sigma_hat',0)
        setattr(binary_tree.Node,'delta',delta)
        setattr(binary_tree.Node, 'lead_count',0)
        
        #Initialize Tree
        self.tree = binary_tree.Binary_Tree(self.L)
        
        #Calculate ancestors
        for midx in range(self.tree.total_nodes):
            self.tree.branches[midx].ancestors = []
            self.tree.branches[midx].auncles = []
            parent = np.array([self.tree.branches[midx].parent])
            while parent[0] > max_node_midx and parent.size != 0:
                self.tree.branches[midx].ancestors.append(parent[0])
                self.tree.branches[midx].auncles.append(self.tree.branches[parent[0]].sibling)
                parent = np.array([self.tree.branches[parent[0]].parent])

        #Initialize Codebook
        if __name__ =='__main__':
            self.W = array_play_codebooks.create_alkhateeb_chiu_codebook(M = Channel.M,N = max_resolution)
#        self.W = array_play_codebooks.create_codebook(M = Channel.M,N = max_resolution, Q = 128)
        
        
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''   
        x_eval = np.exp(-1j* 2* np.pi * self.channel.d * np.cos(self.channel.AoA)*np.arange(0,self.channel.M)/self.channel.lam)
        maxidx = np.argmax(np.abs(np.matmul(np.conj(self.W[-1]),x_eval)))
        self.k_star = maxidx
        if self.k_star + tol >= self.N:
            n_extra = self.k_star + tol - self.N
            self.N_star = np.concatenate([np.arange(self.k_star-tol,self.N),np.arange(0,n_extra+1)])
            return
        if self.k_star - tol < 0:
            n_extra = self.k_star - tol + self.N
            self.N_star = np.concatenate([np.arange(n_extra,self.N),np.arange(0,self.k_star + tol+1)])    
            return
        self.N_star = np.arange(self.k_star - tol,self.k_star + tol + 1)       
 
    
    def check_next(self,n0,n1,tc):
        return
        
    def determine_starting_nodes(self):
        '''Determine nodes under test for starting layer.  Subsequent layers
        will focus on just the children nodes of the previous thereafter'''
        self.nodes_ut = []
        for k in range(0,int(2**(self.l0+1))):
            idx = self.tree.get_idx(self.l0+1,k)
            self.nodes_ut.append(idx)
        self.Nl = len(self.nodes_ut)
        self.gamma = len(self.nodes_ut)-1
        return self.nodes_ut
    
    def find_maxes(self):
        '''Finds index of nodes_ut that corresponds to the highest average rewards,
        and removes it from nodes_ut'''
        node_UCBs = []     
        node_means = []
        nodes = self.tree.branches
        for idx in self.nodes_ut:
            node_UCBs.append(nodes[idx].UCB)
            node_means.append(nodes[idx].mu_hat)
        node_UCBs = np.array(node_UCBs)        
        node_means = np.array(node_means)
        max_UCB_idx = self.nodes_ut[np.argmax(node_UCBs)]                
        max_mu_idx = self.nodes_ut[np.argmax(node_means)]
        return max_UCB_idx, max_mu_idx

    def update_nuts(self,max_idx):
        nodes = self.tree.branches
#        ls = []
#        for idx in self.nodes_ut:
#            ls.append(nodes[idx].indices[0])
#        ls = np.array(ls)
        ell_v = 0

        self.nodes_ut = np.unique(np.concatenate([self.starting_nodes,
                                                  #[max_idx],
                                                  [nodes[max_idx].sibling],
                                                  nodes[max_idx].ancestors[ell_v::],
                                                  nodes[max_idx].auncles,
                                                  nodes[max_idx].children])).astype('int')
    
    def report(self):
        print('AoA: ' + str(self.channel.aoa))
        print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
        nodes_UT = []
        for idx in self.nodes_ut:
            nodes_UT.append(self.tree.branches[idx].info_tuple())
        print('nodes UT: ' + str(nodes_UT))
        print('k_star: ' + str(self.k_star))
        print('N_star: ' + str(self.N_star) + '\n')         
            
    def run_sim(self, T = 1000,tol =1,seed = 0, verbose = False):
        '''
        This simulation implements the stopping criteria that's associated with
        MAB algorithms 
        '''
        self.STOP = False
        self.neighborhood(tol)
        self.Nc = np.zeros(T)
        nodes = self.tree.branches              #Shorthand nodes list
        P = []                                  #Initialize path   
        self.starting_nodes = dc(self.determine_starting_nodes())
        
        t = 1

        while t <= T+1:
            if t > T:
                print('TIME HORIZON REACHED')
                self.report()
                return P
            
            #Conditions for choosing next node to sample
            if t <= 2**(self.l0 + 1):
                #Initially, sample top level starting vertices
                l_test,k_test = nodes[self.nodes_ut[t-1]].indices

            else:
                
                #Decisions are based on UCB and mean maxes
                max_UCB_idx,max_mu_idx = self.find_maxes()
                
                #leader is the max mean 
                nodes[max_mu_idx].lead_count += 1
                
                #This is straight out of the OSUB algorithm
#                if np.mod((nodes[max_mu_idx].lead_count - 1)/(self.gamma + 1),1) == 0:
                if np.mod((nodes[max_mu_idx].lead_count - 1)/(self.gamma + 1),1) == 0:
                    l_test,k_test = nodes[max_mu_idx].indices
                else:
                    l_test,k_test = nodes[max_UCB_idx].indices 
                    
            #Sample and beamforming application on ULA
            P.append((l_test,k_test,t))
            x,r= self.GetReward(t = t,mode = 'complex')
            y = np.abs(np.matmul(np.conj(self.W[l_test-1][k_test]),x))
            
            #Update rewards
            nodes[self.tree.get_idx(l_test,k_test)].update(y)
            
            #Find leader after top level nodes have been tested to make first tree expansion
            if t == 2**(self.l0 + 1):
                max_UCB_idx,max_mu_idx = self.find_maxes()

           
            #Based on new leader, update self.nodes_ut
            if t >= 2**(self.l0 + 1):
                self.update_nuts(max_mu_idx) 
#                self.gamma = 2
                self.gamma = len(self.nodes_ut)-1
                
                
            t += 1
            
            #TODO: This could be improved somehow.  Both nodes are required to be played now, but maybe make a requirement on the UCB
            try:
                if nodes[max_mu_idx].indices[0] == self.L and np.isinf(nodes[nodes[max_mu_idx].sibling].UCB) == False:
#                if nodes[max_mu_idx].indices[0] == self.L and max_mu_idx == max_UCB_idx and np.inf(nodes[max_UCB_idx]) == False:
                    self.STOP = True 
            except:
                pass

            #Continue to next node or conclude sim
            if self.STOP:
                l_hat,k_hat = nodes[max_mu_idx].indices
                P.append((l_hat,k_hat,'M',nodes[max_mu_idx].mu_hat))
                if l_hat == self.L:
                    if k_hat in self.N_star: 
                        self.Nc[t::] = 1
                    else: self.report()
                    break
        return P

#%%Method attributes added to Node object    
def update(self,y):
    c = 1
#    print('indices: ' + str(self.indices) + ' y: ' + str(y))
    self.Nt += 1
    self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
    self.UCB = self.mu_hat + np.sqrt(c*np.log(2/self.delta)/(2*self.Nt)) 
    self.sigma_hat = np.sqrt(((self.Nt-1)*self.sigma_hat**2 + (y-self.mu_hat)**2)/self.Nt) #std dev         
    
def info_tuple(self):
    return (self.indices[0],self.indices[1],self.midx,self.Nt,np.round(self.mu_hat,3))

#%%
if __name__ == '__main__':
    M = 64
    max_resolution = 128
    AoA = 52.44918552111278
    SNR = 0
    T = 150
    start_layer = 3
    epsilon = 7
    delta = .0001

    s = 974
    
    channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)

    agent = OSUB(channel.sample_signal,channel,max_resolution,start_layer = start_layer,delta = delta,seed =s) 
    P = agent.run_sim(T = T,verbose = True)       
#
    plt.figure(0)
    plt.plot(agent.Nc)
    plt.show()
