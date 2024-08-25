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


class Hier_UCB:
    def __init__(self,
                 GetReward,
                 Channel,
                 max_resolution,
                 c = 1,
                 delta = 1,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.L = int(np.log2(self.N))           #Number of layers
        self.delta = delta
        self.c = c
        self.seed = seed
        np.random.seed(seed = seed)

        #Set custom tree attributes
        setattr(binary_tree.Node,'update', update)
        setattr(binary_tree.Node,'UCB',np.inf)
        setattr(binary_tree.Node,'LCB',-np.inf)
        setattr(binary_tree.Node,'Nt',0)
        setattr(binary_tree.Node,'mu_hat',0)
        setattr(binary_tree.Node,'sigma_hat',0)
        setattr(binary_tree.Node,'delta',delta)
        
        #Initialize Tree
        self.tree = binary_tree.Binary_Tree(self.L)

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
            self.nodes_ut.append(self.tree.get_idx(self.l0+1,k))
    
    def perform_elimination(self):
        '''Gather up means of all nodes and compare to max to eliminate'''
        node_means = []
        nodes = self.tree.branches
        for idx in self.nodes_ut:
            node_means.append(nodes[idx].mu_hat)
        node_means = np.array(node_means)
        elim_idxs = np.where(np.max(node_means) - node_means >= 2 * self.alpha)[0]
        self.nodes_ut = np.delete(self.nodes_ut,elim_idxs)
        
    def report(self,current_node,l_hat,k_hat):
        print('AoA: ' + str(self.channel.aoa))
        print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
        print('l_hat: ' + str(l_hat) + '\t' + 'k_hat: ' + str(k_hat))
        print('k_star: ' + str(self.k_star))
        print('N_star: ' + str(self.N_star))
        print('Desc Ind: ' + str(self.k_star in current_node.final_descendents) + '\n')             
            
    def run_sim(self, T = 1000, start_layer = 0,tol =1,seed = 0):
        '''
        This simulation implements the stopping criteria that's associated with
        MAB algorithms 
        '''
        self.l0 = start_layer
        self.neighborhood(tol)
        self.Nc = np.zeros(T)
        nodes = self.tree.branches              #Shorthand nodes list
        P = []                                  #Initialize path   
        self.determine_starting_nodes()
        t = 1
        while t <= T:
            
            #Node testing epoch in each loop
            for idx in self.nodes_ut:
                l_test,k_test = nodes[idx].indices
                x,r= self.GetReward(t = t,mode = 'complex')
                y = np.abs(np.matmul(np.conj(self.W[l_test-1][k_test]),x))
                nodes[idx].update(y)
                t += 1
                if t >= T:
                    print('TIME HORIZON REACHED \n')
                    self.report(nodes[self.nodes_ut[0]],l_test,self.nodes_ut)
                    return P
                
            self.alpha = np.sqrt(np.log(self.c*len(self.nodes_ut)*nodes[idx].Nt**2/self.delta)/nodes[idx].Nt**(4))
#            print('alpha: ' + str(self.alpha))
            self.perform_elimination() 
            
            #Continue to next node or conclude sim
            if len(self.nodes_ut) == 1:
#                print(t)
                l_hat,k_hat = nodes[self.nodes_ut[0]].indices
                P.append((l_hat,k_hat,t))
                if l_hat == self.L:
                    if k_hat in self.N_star: 
                        self.Nc[t::] = 1
                    else: self.report(nodes[self.nodes_ut[0]],l_hat,k_hat)
                    break
                else:
                    self.nodes_ut = nodes[self.nodes_ut[0]].children 

        return P
    
def update(self,y):
    self.Nt += 1
    self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
    self.sigma_hat = np.sqrt(((self.Nt-1)*self.sigma_hat**2 + (y-self.mu_hat)**2)/self.Nt) #std dev         
    

if __name__ == '__main__':
    M = 64
    max_resolution = 128
    AoA = 140
    SNR = -7.5
    T = 128
    c = 5
    delta = .001
    start_layer = 2
    s = 0
    
    channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)

    agent = Hier_UCB(channel.sample_signal,channel,max_resolution,c = c,delta = delta,seed =s) 
    P = agent.run_sim(T = T, start_layer = start_layer)       

    plt.figure(0)
    plt.plot(agent.Nc)
    plt.show()
