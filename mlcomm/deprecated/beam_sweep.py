#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:19:43 2020

@author: nate
"""

import numpy as np
import matplotlib.pyplot
import array_play_codebooks
import rician
from copy import deepcopy as dc

class Agent:
    def __init__(self,
                 Channel,
                 GetReward,
                 resolution,
                 seed = 0):
        
        

        self.channel = Channel
        self.GetReward = GetReward
        self.M = Channel.M
        self.N = resolution
        self.mu_hats = np.zeros([self.N])
        self.seed = seed
        np.random.seed(seed = seed)
        #Initialize Codebook
#        self.W = array_play_codebooks.create_codebook(M = self.M, N = 128, Q = 128)[-1] 
    
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''   
        x_eval = np.exp(-1j* 2* np.pi * self.channel.d * np.cos(self.channel.AoA)*np.arange(0,self.channel.M)/self.channel.lam)
        maxidx = np.argmax(np.abs(np.matmul(np.conj(self.W),x_eval)))
        
        #Use this for dft codebook
#        map_idx_mat = np.argmax(np.abs(np.matmul(np.conj(self.W[-1]),np.transpose(self.athetai))),1)
#        self.k_star = map_idx_mat[maxidx]
#        
        #use this for thetaspecific codebook
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
        
    def run_sim(self,T = 300,tol = 1):
        self.neighborhood(1)
        bidx = 0
        t = 0
        k_hat = 0
        max_val = 0
        self.Nc = []
        for kk in range(self.N):
            x,r = self.GetReward(bidx = bidx,t = t,mode = 'complex')
            y = np.abs(np.matmul(np.conj(self.W[kk]),x))
            if y > max_val: 
                k_hat = dc(kk)
                max_val = dc(y)
            if k_hat in self.N_star: self.Nc.append(1)
            else: self.Nc.append(0)
        
        for tt in range(self.N,T):
            if k_hat in self.N_star: self.Nc.append(1)
            else: self.Nc.append(0)   
        self.Nc = np.array(self.Nc)
            
if __name__ == '__main__':
    path = './../../data/'
    S = 100
    T = 300
    SNRs = [0]
    Pcs = np.zeros(T)
    for SNR in SNRs:
        num_correct = 0
        for s in range(S):
            np.random.seed(seed = s)
            AoA = np.random.uniform(30,150)
            channel = rician.RicianAR1(M = 64,SNR = SNR,AoA = AoA,seed = s)
            agent = Agent(channel,channel.sample_signal, resolution = 128,seed = s)
            agent.run_sim(T = T,tol = 1)
            Pcs += agent.Nc