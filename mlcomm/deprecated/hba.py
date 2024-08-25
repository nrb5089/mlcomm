#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:36:34 2019

@author: nate

This script implements Hierarchical Beam Alignment (HBA) algorithm (Algorithm 1) from 2019-Wu-Fast mmWave Beam Alignment via Correlated Bandit Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import rician
import pickle
import array_play_codebooks
plt.close('all')


class hba:

    def __init__(self, 
                 GetRewards, 
                 Channel,
                 num_codewords = 128,
                 rho1 = 1, 
                 gamma = 0.5, 
                 zeta = .1,
                 seed = 0):
        
        self.GetRewards = GetRewards                                  #Function generating rewards
        self.channel = Channel                                      #Channel class
        self.N = num_codewords                                    #Number of beam directions
        self.rho1 = rho1                                            #First UCB parameter
        self.gamma = gamma                                          #Second UCB parameter
        self.zeta = zeta                                            #Stopping criteria parameter
        self.hmax = int(np.log2(self.N)+1)                       #Maximum Tree depth
        self.Q = np.inf * np.ones([self.hmax,self.N])            #Q-values for algorithm

        
        if __name__ == '__main__':
            self.W = array_play_codebooks.create_codebook(M = self.channel.M,N = self.N,Q = 128)[-1]
        self.Nt = np.zeros([self.hmax,self.N])                       
        self.Rt = np.zeros([self.hmax,self.N])
        self.Et = np.zeros([self.hmax,self.N])    
        self.seed = seed
        np.random.seed(seed = seed)      

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
        
  
    def report(self,l_hat,k_hat):
        print('AoA: ' + str(self.channel.aoa))
        print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
        print('l_hat: ' + str(l_hat) + '\t' + 'k_hat: ' + str(k_hat))
        print('k_star: ' + str(self.k_star))
        print('N_star: ' + str(self.N_star))
#        print('Desc Ind: ' + str(self.k_star in current_node.final_descendents) + '\n') 
        
    def run_sim(self,T = 1000, tol=1,track_flg = False):
        self.flops = np.zeros(T)
        self.Tcal = list([(0,0)])                               #Initialize tree to be constructed.
        self.neighborhood(tol)
        xH,xL = (1,0)
        self.Nc = np.zeros(T)
        for t in range(T):
            h,j = (0,0)
            Pcal = list([(h,j)])
            xH,xL = (1,0)
            while (h,j) in self.Tcal:
                if self.Q[h+1,2*j] > self.Q[h+1,2*j+1]:    
                    (h,j) = (h+1,2*j)
                    xL = xa(xH,xL)
                    self.flops[t-1] += 3
                elif self.Q[h+1,2*j] < self.Q[h+1,2*j+1]:  
                    (h,j) = (h+1,2*j+1)
                    xH = xa(xH,xL)     
                    self.flops[t-1] += 3
                else:
                    (h,j) = (h+1, 2*j + np.random.choice([0,1]))
                
                Pcal.append((h,j))

                if h==self.hmax-1 or j == 2**h:
                    break
                    
            (Ht,Jt) = (h,j)
            if (Ht,Jt) not in self.Tcal:
                self.Tcal.append((Ht,Jt))
#            if self.get_idx(Ht,Jt) in self.N_star and Ht == self.hmax-1:
#                
#                self.Nc.append(1)
#            else:
#                self.Nc.append(0)

            x,r  = self.GetRewards(t = t,mode = 'complex')
            y = np.abs(np.matmul(np.conj(self.W[self.get_idx(h,j)]),x))  
            self.flops[t] += 2*self.channel.M -1

            for (h,j) in Pcal:
                self.Nt[h,j] += 1                       #increment the number of times sampled
                self.Rt[h,j] = ((self.Nt[h,j] - 1)*self.Rt[h,j] + y)/self.Nt[h,j]
                self.flops[t] += 5.0
                
            for (h,j) in self.Tcal:
                if self.Nt[h,j] > 0:
#                    self.Et[h,j] = self.Rt[h,j] + np.sqrt(2*.0001 * np.log(t)/self.Nt[h,j]) + self.rho1*self.gamma**h
                    self.Et[h,j] = self.Rt[h,j] + np.sqrt(2* np.log(t)/self.Nt[h,j]) + self.rho1*self.gamma**h
                    self.flops[t] += 5.0
                else:
                    self.Et[h,j] = np.inf

            That = dc(self.Tcal)
            for (h,j) in self.Tcal:
                if self.Nt[h,j]>0:
                    if h != self.hmax-1:
                        self.Q[h,j] = np.min([self.Et[h,j],np.max([self.Q[h+1,2*j],self.Q[h+1,2*j+1]])])
                        self.flops[t] += 3.0
                    else:
                        self.Q[h,j] = self.Et[h,j]
                That.remove((h,j))
            
            if xH - xL < self.zeta/self.N: #Stopping criteria, still not clear what's going on
#                if  xH - xL == 1/self.channel.M:  
                max_el = np.max(self.Et)
#                    
                self.mh,self.mj = np.where(self.Et == max_el) #Find current arm providing max rewards
                self.mh = self.mh[0]
                self.mj = self.mj[0]
                if self.mh == self.hmax-1 and self.mj in self.N_star: self.Nc[t::] = 1
                else: self.report(self.mh,self.mj)
#              
                return self.get_idx(self.mh,self.mj)
        print('TIME HORIZON REACHED.')
        return 

    def get_idx(self,h,j):
        C = (j/2**h, (j+1)/2**h)
        mid_point =xa(C[0],C[1])
        idx = int(mid_point*self.N)
        return idx
 
                
#Helper functions:
def xa(xH,xL): return xL + (xH-xL)/2
    
if __name__=='__main__':
    
    M = 64
    N = 128
    L = 1
    AoA = 55
    SNR = 0
    s = 0
    
    channel = rician.RicianAR1(M = M,L = L,AoA = AoA,SNR = SNR,seed = s)
    GetRewards = channel.sample_signal
    agent = hba(GetRewards,channel,num_codewords =128,zeta = 1.1)
    agent.run_sim(T = 100, track_flg= False)
    print('Number of Flops: ' + str(agent.flops))
    plt.plot(agent.Nc)
    

    
    