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
import rician
from copy import deepcopy as dc
#import cdl
plt.close('all')

class LinearGraph:
    '''Linear connected graph object'''
    def __init__(self,N):
        self.N = N
        self.nodes = []
        for nn in range(N):
            self.nodes.append(Node(nn,N))
            
            
class Node:
    def __init__(self,idx,N):
        if idx == 0:
            self.neighbors = [idx + 1]
        elif idx == N-1:
            self.neighbors = [idx - 1]
        else:
            self.neighbors = [idx-1,idx + 1]
        
        self.Nt = 0
        self.mu_hat = 0
        self.UCB = np.inf
        self.lead_count = 0
        
    def update(self,y):
        c = 1
    #    print('indices: ' + str(self.indices) + ' y: ' + str(y))
        self.Nt += 1
        self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
        self.UCB = self.mu_hat + np.sqrt(c*np.log(2/.001)/(2*self.Nt)) 

            
class UBA:
    '''Graphical Version'''
    def __init__(self,
                 GetReward,
                 Channel,
                 max_resolution,
                 Psi = 4,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.Psi = Psi
        self.seed = seed
        np.random.seed(seed = seed)
             
        #Initialize Graph
        self.graph = LinearGraph(max_resolution)
        self.gamma = 2
        
        if __name__=='__main__':
#            self.W = array_play_codebooks.create_codebook(M = self.channel.M,N = self.N,Q = 128)[-1]
            self.W = array_play_codebooks.create_alkhateeb_chiu_codebook(M = self.channel.M,N = self.N)[-1]     
            
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''   
        x_eval = np.exp(-1j* 2* np.pi * self.channel.d * np.cos(self.channel.AoA)*np.arange(0,self.channel.M)/self.channel.lam)
        maxidx = np.argmax(np.abs(np.matmul(np.conj(self.W),x_eval)))
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
         
    
    def find_max_UCB(self):
        node_UCBs = []     
        for idx in self.nodes_ut:
            node_UCBs.append(self.graph.nodes[idx].UCB)
        return self.nodes_ut[np.argmax(node_UCBs)]
    
    def find_leader(self):
        node_means = []
        for idx in range(self.N):
            node_means.append(self.graph.nodes[idx].mu_hat)
        return np.argmax(node_means)

    def update_nuts(self,max_idx):
        nodes = self.graph.nodes
        self.nodes_ut = np.concatenate([[max_idx],nodes[max_idx].neighbors])
    
    def report(self):
        print('AoA: ' + str(self.channel.aoa))
        print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
        print('nodes UT: ' + str(self.nodes_ut))
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
        nodes = self.graph.nodes              #Shorthand nodes list
        self.nodes_ut = [0,1]
        P = []                                  #Initialize path   
        avg_power = 0
        peak_power = 0
        t = 1

        while t <= T+1:
            if t > T:
                print('TIME HORIZON REACHED')
                self.report()
                return P
  
            #Decisions are based on UCB and mean maxes
            max_mu_idx = self.find_leader() #WARNING: This is an index considering all N arms
            self.update_nuts(max_mu_idx)  
            max_UCB_idx = self.find_max_UCB() #WARNING: This is an index considering all N arms
            
            #leader is the max mean 
            nodes[max_mu_idx].lead_count += 1
            
            #This is straight out of the OSUB algorithm
            if np.mod((nodes[max_mu_idx].lead_count - 1)/(self.gamma + 1),1) == 0:
                k_test = dc(max_mu_idx)
            else:
                k_test = dc(max_UCB_idx) 
                    
            #Sample and beamforming application on ULA
            P.append((k_test,t))
            x,r= self.GetReward(t = t,mode = 'complex')
            y = np.abs(np.matmul(np.conj(self.W[k_test]),x))
            
            #Update rewards
            nodes[k_test].update(y)
            
            #Check Stopping Criteria
            avg_power = ((t-1) *avg_power + y)/t
            if y > peak_power:
                peak_power = dc(y)
            
            if __name__ == '__main__':
                print(peak_power/avg_power)           
            if peak_power/avg_power > self.Psi:
                self.STOP = True

            t += 1

            #Continue to next node or conclude sim
            if self.STOP:
                k_hat = dc(k_test)
                P.append((k_hat,'M',nodes[max_mu_idx].mu_hat))
                if k_hat in self.N_star: 
                    self.Nc[t::] = 1
                else: self.report()
                break
        return P



#%%
if __name__ == '__main__':
    M = 64
    max_resolution = 128
    AoA = 52.44918552111278
    SNR = 0
    T = 10000
    Psi = 4

    s = 974
    
    channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)

    agent = UBA(channel.sample_signal,channel,max_resolution,Psi = Psi,seed =s) 
    P = agent.run_sim(T = T,verbose = True)       

    plt.figure(0)
    plt.plot(agent.Nc)
    plt.show()
