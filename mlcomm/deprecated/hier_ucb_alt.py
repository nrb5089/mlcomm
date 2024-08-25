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
                 epsilon = .01,
                 beta = 1,
                 delta = 1,
                 seed = 0):
        
        self.GetReward = GetReward
        self.channel = Channel
        self.N = max_resolution                 #Number of codewords for highest resolution codebook section
        self.L = int(np.log2(self.N))           #Number of layers
        self.epsilon = epsilon
        self.beta = beta

        self.delta = delta
        self.seed = seed
        self.beta = beta
        self.lam_p = ((2+beta)/beta)**2
#        self.lam_p = 1 + 10/self.N      #suggested by the paper outside the theory
        self.alpha =  self.lam_p* (1 + np.log(2*np.log(self.lam_p*self.N/delta))/np.log(self.N/delta))

#        self.alpha = 1.2
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
        

    def run_sim(self, T = 1000, start_layer = 0, bidx = 0, tol =2,seed = 0,track_flg = False, verbose = True):
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
        c_test = c0
        c_prev = None
        
        
#        for t in range(1,T+1):
        t = 1
        self.Deltas = np.zeros(self.L)
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
                kk = 0
                idx = 0
                
                #Play arms for each time t. Eliminate arms as their UCB falls lower than the highest LCB
                while len(idxs) > 1:
                    if tc > T or t > T:
                        print('stalled in precursor')
                        break

                    x,r = self.GetReward(bidx = bidx,t = tc-1,mode = 'complex')
                    y = np.abs(np.matmul(np.conj(w_s[kk]),x))
#                    y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
                    nodes[idx].update(y,tc,self.b,self.c,self.zeta,self.N)
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
                    kk = np.argmax(UCBs)
                    idx = idxs[kk]     
            
            if c_prev == c0:
                c_test = c1
            if c_prev == c1:
                c_test = c0
            c_prev = c_test
#            test_idx = np.argmax([nodes[c0].UCB,nodes[c1].UCB])
#            c_test = [c0,c1][test_idx]

            l_hat, k_hat = nodes[c_test].indices

            if t > T:
                break
            w_hat = self.W[l_hat-1][k_hat]
            x,r = self.GetReward(bidx = bidx,t = t-1,mode = 'complex')
            y = np.abs(np.matmul(np.conj(w_hat),x))
#            y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize

            nodes[c_test].update(y,tc,self.beta,self.epsilon,self.delta,2)
            P.append(nodes[c_test].indices)
            Pr.append(y)
            
            #Update Trackers
            if track_flg:
                self.R.append(self.channel.r_star-y)
                self.r.append(y)
            if current_node.indices[1] in self.N_star and current_node.indices[0] == self.L-1: self.Nc.append(1) 
            else: self.Nc.append(0)

            
            if self.check_next(nodes[c0],nodes[c1],tc):
#                print(current_node.indices[0])
                self.Deltas[current_node.indices[0]] = np.abs(nodes[c0].mu_hat - nodes[c1].mu_hat)
#                print(str(current_node.indices) + ': ' + str(np.abs(nodes[c0].mu_hat - nodes[c1].mu_hat)))
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
#        print('Steps to Final Arm: ' + str(t-1))
        for tc in range(t+1,T+1):
            if k_hat in self.N_star: 
                self.Nc.append(1)
            else: 
                if tc == t+1:
                    pass
#                    print(k_hat)
                self.Nc.append(0)             
            x,r= self.GetReward(bidx = bidx,t = tc-1,mode = 'complex')
            y = np.abs(np.matmul(np.conj(w_hat),x))
#            y = (y-self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  #Obtain rewards and normalize
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
        self.k_star = self.N*0.5*np.cos(self.channel.aoa*np.pi/180)
        if self.k_star < 0: self.k_star += self.N    
        self.k_star = int(self.k_star)  

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

        #Condition listed in Procedure
#        cond0 = False
#        if n0.Nt > 1 + self.lam_p * n1.Nt and n1.Nt > 0:
#            cond0 = True
#        if n1.Nt > 1 + self.lam_p * n0.Nt and n0.Nt > 0:
#            cond0 = True
            
        #Condition listed in Implementation  
        eps = (n0.indices[0]+.05)/4
        cond0 = False
#        print('LCB0: ' + str(n0.LCB) + '\t UCB1: ' + str(n1.UCB) + '\t' + str(n0.LCB > n1.UCB - eps))
        if n0.LCB > n1.UCB-eps:
            cond0 = True
#        print('LCB1: ' + str(n1.LCB) + '\t UCB0: ' + str(n0.UCB) + '\t' + str(n1.LCB > n0.UCB - eps))
        if n1.LCB > n0.UCB-eps:
            cond0 = True       
        cond1 = n0.Nt > 1 and n1.Nt > 1 and n0.Nt==n1.Nt       #need at least two samples to get            
        cond = cond0 and cond1

        
        
        return cond
        
        
def update(self,y,t,beta,epsilon,delta,N):
    self.Nt += 1
    self.mu_hat = ((self.Nt -1)*self.mu_hat + y)/self.Nt
    self.sigma_hat = np.sqrt(((self.Nt-1)*self.sigma_hat**2 + (y-self.mu_hat)**2)/self.Nt) #std dev         
    sigma = np.sqrt(0.15) #default worst case scenario
#    sigma = self.sigma_hat
    
    #max sigma
    #standard bias from procedure:
#    U = (1+ beta) * (1+ np.sqrt(epsilon)) * np.sqrt(2* sigma**2 *(1 + epsilon)*np.log(np.log((1 + epsilon)*self.Nt)/delta)/(self.Nt)) 
    
    #bias from simulations:
    U =  (1+ np.sqrt(epsilon)) * np.sqrt(2* sigma**2 *(1 + epsilon)*np.log(   2*np.log(((1 + epsilon)*self.Nt+2)/(delta/N))    )/(self.Nt)) 
    
    #empirical sigma
    #standard bias from procedure:
#    U = (1+ beta) * (1+ np.sqrt(epsilon)) * np.sqrt(2* self.sigma_hat**2 *(1 + epsilon)*np.log(np.log((1 + epsilon)*self.Nt)/delta)/(self.Nt)) 
    
    #bias from simulations:
#    U =  (1+ np.sqrt(epsilon)) * np.sqrt(2* self.sigma_hat**2 *(1 + epsilon)*np.log(   2*np.log(((1 + epsilon)*self.Nt+2)/(delta/N))    )/(self.Nt)) 
    
    if self.Nt > 0:
        self.UCB = self.mu_hat + U
        self.LCB = self.mu_hat - U
#    print('Node: ' +str(self.indices) + '\t mu: ' + str(self.mu_hat) + '\t UCB: ' + str(self.UCB) + '\t LCB: ' + str(self.LCB) + '\t U : ' + str(U))
#    print('Node: ' +str(self.indices) + '\t mu: ' + str(self.mu_hat) + '\t UCB: ' + str(self.UCB) + '\t LCB: ' + str(self.LCB) + '\t U : ' + str(U) + '\t sighat: ' + str(self.sigma_hat**2))
#    print('Node: ' +str(self.indices) + '\t mu: ' + str(self.mu_hat) + '\t UCB: ' + str(self.UCB) + '\t U : ' + str(U))

if __name__ == '__main__':
    M = 32
    max_resolution = 128
    beta = 1 #beta
    epsilon = .01 #epsilon
    nu = .01
#    delta_max = np.log(1 + epsilon)/np.exp(1)
    
    
    delta_sim = np.log(1+epsilon)*(nu*epsilon/(2+epsilon))**(1/(1+epsilon))
#    delta = .9 * delta_max #delta
    delta = .6 * delta_sim
    c_epsilon = (2 + epsilon)/epsilon * (1/np.log(1+ epsilon))**(1+epsilon)
    Pcalc = 1-4*np.sqrt(c_epsilon*delta) - 4*c_epsilon*delta
#    delta = .01
    AoA = 144.52526139916029
    tol = 1
    SNR = 0
    T = 300
    s = 16

    channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)

    agent = Hier_UCB(channel.sample_signal,channel,max_resolution,epsilon = epsilon, beta=beta, delta = delta,seed =s)        
    P,Pr = agent.run_sim(T =T,start_layer =0,tol = tol,verbose = True)
#    print('alpha: ' + str(agent.alpha))
    print('lam_p: ' + str(agent.lam_p))
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