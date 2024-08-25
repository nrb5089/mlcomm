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
import beams
import binary_tree
import rician
from copy import deepcopy as dc
import h5py
#import cdl
plt.close('all')


class Hier_UCB:
    def __init__(self,
                 GetReward,
                 Channel,
                 max_resolution,
                 c = 0,
                 b = 1,
                 zeta = 1,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.L = int(np.log2(self.N))           #Number of layers
        self.c = c
        self.b = b
        self.zeta = zeta
        self.seed = seed
        np.random.seed(seed = seed)

        #Set custom tree attributes
        setattr(binary_tree.Node,'update', update)
        setattr(binary_tree.Node,'UCB',np.inf)
        setattr(binary_tree.Node,'LCB',-np.inf)
        setattr(binary_tree.Node,'Nt',0)
        setattr(binary_tree.Node,'mu_hat',0)
        setattr(binary_tree.Node,'sigma_hat',0)
        
        #Initialize Tree
        self.tree = binary_tree.Binary_Tree(self.L)

        #Initialize Codebook
        self.W = array_play_codebooks.create_codebook(M = Channel.M,N = max_resolution, Q = 128)
        

    def run_sim(self, T = 1000, start_layer = 0, bidx = 0, tol =2,seed = 0,track_flg = False, verbose = False):
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
        Pr = []
        c0,c1 = current_node.children
        tc = 1                                  #local time index for individual binary bandits
        c_test = np.random.choice([c0,c1]) 
        c_prev = None
        
        
        
#        for t in range(1,T+1):
        t = 1
        while t <= T:
            
            #Routine for starting in a layer below the top
            if start_layer > 0 and t == 1:
                w_s = self.W[start_layer]
                idxs = []
                tc = 1
                
                #Build array consisting of node master indices for ever k_lth child node
                for kl in range(len(w_s)):
                    idxs.append(self.tree.get_idx(start_layer+1,kl))
                idxs = np.array(idxs)
                
                #Play arms for each time t. Eliminate arms as their UCB falls lower than the highest LCB
                while len(idxs) > 1:
                    if tc > T or t > T:
                        print('stalled in precursor')
                        break
                    for kk,idx in enumerate(idxs):
                        if tc > T or t > T:
                            print('stalled in precursor')
                            break
                        x,r = self.GetReward(bidx = bidx,t = tc-1,mode = 'complex')
                        y = 10*np.log10(np.abs(np.matmul(np.conj(w_s[kk]),x))) 
                        y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
                        nodes[idx].update(y,tc,self.b,self.c,self.zeta)
                        self.Nc.append(0)
                        P.append(nodes[idx].indices)
                        tc += 1         #one arm pull each timestamp
                        t += 1          #master time index
                        
                    #Gather arm information
                    UCBs = []
                    LCBs = []
                    Nts = []
                    for idx in idxs:
                        UCBs.append(nodes[idx].UCB)
                        LCBs.append(nodes[idx].LCB)
                        Nts.append(nodes[idx].Nt)
                    UCBs = np.array(UCBs)
                    LCBs = np.array(LCBs)
                    Nts = np.array(Nts)

                        
                    if np.all(Nts > 2): #minimum number of arm pulls condition
                        ridxs = np.where(UCBs < np.max(LCBs))[0]
                        idxs = np.delete(idxs,ridxs)
                        if len(idxs) == 1:
                            current_node = nodes[idxs[0]]
                            c0,c1 = current_node.children
                            c_test = np.random.choice([c0,c1]) 
                            tc = 1
                        if len(idxs) == 2:
                            c0,c1 = idxs[0],idxs[1]
                            c_test = np.random.choice([c0,c1]) 
                            tc = 1                                   
                               
                            
#            c_test = np.random.choice([c0,c1]) 
            if c_prev == c0:
                c_test = c1
            if c_prev == c1:
                c_test = c0
            

            l_hat, k_hat = nodes[c_test].indices
            if t > T:
                break
            w_hat = self.W[l_hat-1][k_hat]
            x,r = self.GetReward(bidx = bidx,t = t-1,mode = 'complex')
            y = 10*np.log10(np.abs(np.matmul(np.conj(w_hat),x))) 
            y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
            
            if y > 1:
                print('up')
            if y < 0:
                print('down')
            nodes[c_test].update(y,tc,self.b,self.c,self.zeta/(l_hat-start_layer))
            
            P.append(nodes[c_test].indices)
            Pr.append(y)
            c_prev = dc(c_test)
            
            #Update Trackers
            if track_flg:
                self.R.append(self.channel.r_star-y)
                self.r.append(y)
            if current_node.indices[1] in self.N_star and current_node.indices[0] == self.L-1: self.Nc.append(1) 
            else: self.Nc.append(0)
#            self.Nc.append(0)

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
                t += 1
            else:
                tc += 1
                t  += 1
                
        (l_hat,k_hat) = current_node.indices
        w_hat = self.W[l_hat-1][k_hat]
            
        for tc in range(t+1,T+1):
            if k_hat in self.N_star: 
                self.Nc.append(1)
            else: 
                if tc == t+1:
                    pass
#                    print(k_hat)
                self.Nc.append(0)             
            x,r= self.GetReward(bidx = bidx,t = tc-1,mode = 'complex')
            y = 10*np.log10(np.abs(np.matmul(np.conj(w_hat),x)))
            y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
            P.append(current_node.indices)    
            if track_flg:                
                self.R.append(self.channel.r_star-y)
                self.r.append(y)
                
        if np.abs(self.k_star-k_hat) > tol and self.k_star != self.N-1 and verbose:
            self.report(current_node,l_hat,k_hat)

        self.Nc = np.array(self.Nc)
        if track_flg:
            self.R = np.array(self.R)
            self.r = np.array(self.r) 
        return P,Pr
    
    def report(self,current_node,l_hat,k_hat):
        print('AoA: ' + str(self.channel.aoa))
        print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
        print('l: ' + str(l_hat) + '\t' + 'k_hat: ' + str(k_hat))
        print('k_star: ' + str(self.k_star))
        print('N_star: ' + str(self.N_star))
        print('Desc Ind: ' + str(self.k_star in current_node.final_descendents) + '\n')
        
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''
        self.k_star = int(self.N*0.5*np.cos(self.channel.aoa*np.pi/180))
        if self.k_star < 0: self.k_star += self.N    

        if self.k_star + tol > self.N:
            n_extra = self.k_star + tol - self.N
            self.N_star = np.concatenate([np.arange(self.k_star-tol,self.N),np.arange(0,n_extra + 1)])
            return
        if self.k_star - tol < 0:
            n_extra = self.k_star - tol + self.N
            self.N_star = np.concatenate([np.arange(n_extra,self.N),np.arange(0,self.k_star + tol+1)])    
            return
        self.N_star = np.arange(self.k_star - tol,self.k_star + tol + 1)
        
 
    
    def check_next(self,n0,n1,tc):
        cond0 = False
        if n0.LCB > n1.UCB:
            cond0 = True
        if n1.LCB > n0.UCB:
            cond0 = True
#       
        cond1 = True #fix
        
#        #need to sample at least twice to get a variance estimate
        cond2 = n0.Nt > 2 and n1.Nt > 2 and n0.Nt == n1.Nt         #need at least two samples to get 

            
        cond = cond0 > 0 and cond1 > 0 and cond2
        
        
        return cond
        
        
def update(self,y,t,b,c,z):
    self.Nt += 1
    self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
    self.sigma_hat = np.sqrt(((self.Nt-1)*self.sigma_hat**2 + (y-self.mu_hat)**2)/self.Nt) #std dev         
    self.UCB = self.mu_hat + np.sqrt(2*self.sigma_hat**2 * z*np.log(t)/self.Nt) + c * 3 * b * z*np.log(t)/self.Nt  
    self.LCB = self.mu_hat - np.sqrt(2*self.sigma_hat**2 * z*np.log(t)/self.Nt) - c * 3 * b * z*np.log(t)/self.Nt  
#    print('Node: ' +str(self.indices) + '\t mu: ' + str(self.mu_hat) + '\t UCB: ' + str(self.UCB) + '\t LCB: ' + str(self.LCB))


if __name__ == '__main__':
    M = 32
    max_resolution = 128
    b = 1
    c = .8e-1
    zeta = .9e-1
    AoA = 30.78251406115789
    tol = 1
    SNR = 0
    T = 300
    s = 124
#    channel = beams.Beams(M = M,AoA = AoA,N0 = -174,L=4,seed = 79)
    channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)
    
 

    

#    filename = '/S_' +  str(10000) + '_T_' + str(T) + '_M_' + str(M) + 'CDLabc.h5'  
#    with h5py.File('./../../cdl_dataset' + filename,'r') as f:    
#        signal_data = np.transpose(f['data'][:],[3,2,1,0])
#        max_data = f['labels'][:]
#    np.random.seed(seed = 0)
#    signal_data = signal_data[:,:,:,0] + 1j * signal_data[:,:,:,1]
#    idx = np.random.choice(np.arange(0,10000),10000, replace = False)
#    signal_data = signal_data[idx]
#    max_data = max_data[idx]
#    channel = cdl.CDL(signal_data = signal_data[s], M = M,kmax = max_data[s],SNR = SNR,seed = s) #-140.5 ~ 0 dB for SNR
    agent = Hier_UCB(channel.sample_signal,channel,max_resolution,c = c,b=b,zeta = zeta,seed =s)        
    P,Pr = agent.run_sim(T =300,start_layer =1,tol = tol,verbose = True)

    history = agent.node_history
    plt.figure(0)
    plt.plot(agent.Nc)
    plt.show()

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
    plt.show()
    
#    plt.figure(2)
#    plt.plot(np.cumsum(Pr[0::2]))
#    plt.plot(np.cumsum(Pr[1::2]))
#    plt.legend(['(1,0)','(1,1)'])