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


class LSE:
    '''Graphical Verion'''
    def __init__(self,
                 GetReward,
                 Channel,
                 max_resolution,
                 start_layer = 1,
                 epsilon = 1,
                 delta = 1,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.L = int(np.log2(self.N))           #Number of layers
        self.delta = delta
        self.epsilon = epsilon
        self.seed = seed
        np.random.seed(seed = seed)
        
        self.l0 = start_layer #fixed
        max_node_midx = np.array([2,6,14,30,62])[self.l0-1]
        self.tau0 = np.ceil(4/self.epsilon**2 * np.log(8/self.delta)) #default tau value, but may be incremented

        
        #Set custom tree attributes
        setattr(binary_tree.Node,'update', update)
        setattr(binary_tree.Node,'info_tuple',info_tuple)
        setattr(binary_tree.Node,'UCB',np.inf)
        setattr(binary_tree.Node,'LCB',-np.inf)
        setattr(binary_tree.Node,'Nt',0)
        setattr(binary_tree.Node,'mu_hat',0)
        setattr(binary_tree.Node,'sigma_hat',0)
        setattr(binary_tree.Node,'delta',delta)
        
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
        return self.nodes_ut
    
    def perform_movement(self):
        '''Finds index of nodes_ut that corresponds to the highest average rewards,
        and removes it from nodes_ut'''
        node_means = []        
        nodes = self.tree.branches
        for idx in self.nodes_ut:
            node_means.append(nodes[idx].mu_hat)
        node_means = np.array(node_means)
        max_idx = self.nodes_ut[np.argmax(node_means)]
#        print('max_mu: ' + str(nodes[max_idx].mu_hat))
        self.update_nuts(max_idx,node_means)

        self.update_tau()
        
        if nodes[max_idx].indices[0] == self.L:
            self.STOP = True
                
#        print('max_idx: ' + str(max_idx) + ' nodesut: ' + str(self.nodes_ut))
        return max_idx

    def update_nuts(self,max_idx,node_means):
        nodes = self.tree.branches
        ls = []
        for idx in self.nodes_ut:
            ls.append(nodes[idx].indices[0])
        ls = np.array(ls)
        ell_v = 0
#        if np.all(ls <= nodes[max_idx].indices[0]):
#            self.nodes_ut = np.delete(self.nodes_ut,np.argmax(node_means))
#            self.nodes_ut = np.unique(np.concatenate([self.nodes_ut,
#                                                      np.array(nodes[max_idx].children),
#                                                      ]).astype('int'))
    
        self.nodes_ut = np.unique(np.concatenate([self.starting_nodes,
#                                                  [max_idx],
                                                  nodes[max_idx].ancestors[ell_v::],
                                                  nodes[max_idx].auncles,
                                                  nodes[max_idx].children])).astype('int')
#        print(self.nodes_ut)
#        else:
#            self.determine_starting_nodes()
        
    def update_tau(self):
        Nts = []
        nodes = self.tree.branches
        for idx in self.nodes_ut:
            Nts.append(nodes[idx].Nt)
        Nts = np.array(Nts)
        max_Nt = np.max(Nts)
        #If max_Nt > self.tau then I need to sample everything by at least max_Nt
        
        if max_Nt > self.tau0:
            self.tau = dc(max_Nt)
        
        if max_Nt < self.tau0:
            self.tau = dc(self.tau0)
            
        if np.all(Nts == max_Nt) or np.all(Nts == self.tau0):
            self.tau = max_Nt + 1
            

    
    
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
        esampled = np.zeros(len(self.nodes_ut))
        self.prev_max_idx = []
        self.tau = dc(self.tau0)

        while t <= T+1:
#            print(t)
            #Node testing epoch in each loop
            for ii,idx in enumerate(self.nodes_ut):
                l_test,k_test = nodes[idx].indices
                if t > T:
                    print('TIME HORIZON REACHED')
                    self.report()
                    return P
                if nodes[idx].Nt == self.tau:
                    esampled[ii] = 1
                else:
                    l_test,k_test = nodes[idx].indices
                    P.append((l_test,k_test,t,self.tau))
                    x,r= self.GetReward(t = t,mode = 'complex')
                    y = np.abs(np.matmul(np.conj(self.W[l_test-1][k_test]),x))
                    nodes[idx].update(y)
                    t += 1
                
                      
            
            if np.all(esampled == 1): #2*Nl = 8 for all cases here
                max_idx = self.perform_movement() 
                esampled = np.zeros(len(self.nodes_ut))
            
            #Continue to next node or conclude sim
            if self.STOP:
                l_hat,k_hat = nodes[max_idx].indices
                P.append((l_hat,k_hat,'M',nodes[max_idx].mu_hat))
                if l_hat == self.L:
                    if k_hat in self.N_star: 
                        self.Nc[t::] = 1
                    else: self.report()
                    break
        return P
    
def update(self,y):
#    print('indices: ' + str(self.indices) + ' y: ' + str(y))
    self.Nt += 1
    self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
    self.sigma_hat = np.sqrt(((self.Nt-1)*self.sigma_hat**2 + (y-self.mu_hat)**2)/self.Nt) #std dev         
    
def info_tuple(self):
    return (self.indices[0],self.indices[1],self.midx,self.Nt,np.round(self.mu_hat,3))

if __name__ == '__main__':
    M = 64
    max_resolution = 128
    AoA = 68.30275395614899
    SNR = 0
    T = 128
    epsilon = 6
    delta = .001

    s = 277
    
    channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)

    agent = LSE(channel.sample_signal,channel,max_resolution,epsilon = epsilon,delta = delta,seed =s) 
    P = agent.run_sim(T = T,verbose = True)       
#
    plt.figure(0)
    plt.plot(agent.Nc)
    plt.show()
