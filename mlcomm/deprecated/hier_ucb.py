#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:47:00 2019

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
sys.path.insert(0,'/media/nate/New Volume/DDrive/Education/PhD/Dissertation/Modeling/Multi-User/BA/mlcomm/mlcomm')
import array_play_codebooks
import beams
import binary_tree
from copy import deepcopy as dc
plt.close('all')


class Hier_UCB:
    def __init__(self,
                 GetReward,
                 Channel,
                 max_resolution,
                 delta = .001,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.L = int(np.log2(self.N))           #Number of layers
        self.sigrange = 10**(np.array(Channel.sigrange)/10)
        self.delta = delta
        self.seed = seed
        np.random.seed(seed = seed)

        #Set custom tree attributes
        setattr(binary_tree.Node,'update', update)
        setattr(binary_tree.Node,'UCB',np.inf)
        setattr(binary_tree.Node,'Nt',0)
        setattr(binary_tree.Node,'mu_hat',0)
        setattr(binary_tree.Node,'sigma_hat',0)
        
        #Initialize Tree
        self.tree = binary_tree.Binary_Tree(self.L)

        #Initialize Codebook
        self.W = array_play_codebooks.create_codebook(M = Channel.M,N = max_resolution, Q = 128)
        

    def run_sim(self, T = 1000,tol =2,track_flg = False):
        '''
        This simulation implements the stopping criteria that's associated with
        MAB algorithms 
        '''
        self.neighborhood(tol)
        self.Nc = []
        self.node_history = []
        self.r = []                             #Time array of rewards 
        self.R = []                             #Time array of regrets
        nodes = self.tree.branches              #Shorthand nodes list
        current_node = nodes[0]                 #Always start from the top in search mode
        P = []                                  #Initialize path   
        c0,c1 = current_node.children
        tc = 1                                  #local time index for individual binary bandits
        c_test = np.random.choice([c0,c1]) 
        c_prev = None
        for t in range(1,T+1):
#            c_test = np.random.choice([c0,c1]) 
            if c_prev == c0:
                c_test = c1
            if c_prev == c1:
                c_test = c0
            
            l_hat, k_hat = nodes[c_test].indices
            w_hat = self.W[l_hat-1][k_hat]
            x,r = self.GetReward(mode = 'complex')
            y = 10*np.log10(np.abs(np.matmul(np.conj(w_hat),x))) + 30
            y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
            
            nodes[c_test].update(y,t)
            
            P.append(nodes[c_test].indices)
            c_prev = dc(c_test)
            
            #Update Trackers
            if track_flg:
                self.R.append(self.channel.r_star-y)
                self.r.append(y)
            if current_node.indices[1] in self.N_star: self.Nc.append(1) 
            else: self.Nc.append(0)

            if self.check_next(nodes[c0],nodes[c1],tc):
                if nodes[c0].UCB > nodes[c1].UCB: 
                    current_node = nodes[c0]
                if nodes[c0].UCB < nodes[c1].UCB: 
                    current_node = nodes[c1] 
                if nodes[c0] == nodes[c1]:
                    current_node = nodes[np.random.choice([c0,c1])]
                    
                #Check if we are on the last layer
                if current_node.indices[0] == self.L: 
                    break
                else:
                    c0,c1 = current_node.children
                    c_test = np.random.choice([c0,c1]) 
                tc = 1
            else:
                tc += 1

        (l_hat,k_hat) = current_node.indices
        w_hat = self.W[l_hat-1][k_hat]
            
        for tc in range(t+1,T+1):
            if k_hat in self.N_star: 
                self.Nc.append(1)
            else: 
                self.Nc.append(0)             
            x,r= self.GetReward(mode = 'complex')
            y = 10*np.log10(np.abs(np.matmul(np.conj(w_hat),x))) + 30
            y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
            P.append(current_node.indices)    
            if track_flg:                
                self.R.append(self.channel.r_star-y)
                self.r.append(y)
                
        if np.abs(self.k_star-k_hat) > tol:
            print('AoA: ' + str(self.channel.aoa))
            print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
            print('k_hat: ' + str(k_hat))
            print('k_star: ' + str(self.k_star) + '\n')
        self.Nc = np.array(self.Nc)
        if track_flg:
            self.R = np.array(self.R)
            self.r = np.array(self.r) 
        return P
    
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''
        self.k_star = int(self.N*0.5*np.cos(self.channel.aoa*np.pi/180))
        if self.k_star < 0: self.k_star += self.N    

        if self.k_star + tol > self.N:
            n_extra = self.k_star + tol - self.N
            self.N_star = np.concatenate([np.arange(self.k_star-tol,self.N),np.arange(0,n_extra)])
            return
        if self.k_star - tol < 0:
            n_extra = self.k_star - tol + self.N
            self.N_star = np.concatenate([np.arange(n_extra,self.N),np.arange(0,self.k_star + tol)])    
            return
        self.N_star = np.arange(self.k_star - tol,self.k_star + tol + 1)
        

    def check_next(self,n0,n1,tc):
        Delta2 = (n0.mu_hat - n1.mu_hat)**2
        #play at least twice on each arm to obtain some type of variance measurement (otherwise it's 0).  

        #Traditional bandit sampling bound:
        cond = n0.Nt > np.max([8*n0.sigma_hat**2 * np.log(1/self.delta)/Delta2,2])  and n1.Nt > np.max([8*n1.sigma_hat**2 * np.log(1/self.delta)/Delta2,2])
        
        return cond
    
def update(self,y,t):
    self.Nt += 1
    self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
    self.sigma_hat = ((self.Nt-1)*self.sigma_hat + (y-self.mu_hat)**2)/self.Nt
    self.UCB = self.mu_hat + np.sqrt(2*self.sigma_hat**2 * np.log(t)/self.Nt)
    
if __name__ == '__main__':
    M = 128
    max_resolution = 128
    delta = 1e-5
    b = 1
    c = 0
    zeta = 1
    AoA = 145.45355100675263
    
    channel = beams.Beams(M = M,AoA = AoA,L=4,seed = 59)
    agent = Hier_UCB(channel.sample_signal,channel,max_resolution,delta = delta,seed =288)        
    P = agent.run_sim(T =300)
    history = agent.node_history
    plt.figure(0)
    plt.plot(agent.Nc)
    
    #Plot UCBs
    UCBs = []
    mus = []
    for ii in range(0,int(2*max_resolution -1)):
        UCBs.append(agent.tree.branches[ii].UCB)
        mus.append(agent.tree.branches[ii].mu_hat)
    UCBs = np.array(UCBs)
    mus = np.array(mus)    
    plt.figure(1)
#    for ii in range(0,agent.L):
#        plt.subplot(agent.L,1,ii+1)
    plt.stem(mus[-128::])
    plt.plot(UCBs[-128::],'ro')