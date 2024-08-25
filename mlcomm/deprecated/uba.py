#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:26:24 2019

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
import beams
import pickle 
import array_play_codebooks
import rician
plt.close('all')

class uba:
    def __init__(self,
                 GetRewards,
                 Channel,
                 num_codewords = 128,
                 Psi = 1.01,
                 gamma = 2):
        
        self.GetRewards = GetRewards
        self.channel = Channel
        self.N = num_codewords
        self.Psi = Psi                              #Stopping criteria threshold
        self.gamma = gamma                          #Governing multiple to pick leader
        self.rmat = np.zeros([1,self.N])
        self.lt = np.zeros([self.N])             #kth element denotes how many times the kth arms has been the leader  
        self.Lt = []                                #Track leader at time t
        self.sk = np.zeros([self.N])             #Number of times kth arm was selected
        self.bk = np.inf * np.ones([self.N])             #Arm selection criteria
        self.mode = 'search'                        #Mode indicating to continue to search for beam alignments
        self.c = 0.5                                #Constant not specified in paper c > 0
        self.q_vec = np.linspace(0,1,1000)          #Normalized vector for determining KL-UCB   
        self.muk = np.zeros([self.N])               #Empirical mean initialization
        self.peaks = np.zeros([self.N])
        
        if __name__=='__main__':
#            self.W = array_play_codebooks.create_codebook(M = self.channel.M,N = self.N,Q = 128)[-1]
            self.W = array_play_codebooks.create_alkhateeb_chiu_codebook(M = self.channel.M,N = self.N)[-1]
            
    def check_terminate(self,k,t):
#        pk_avg = np.sum(self.rmat[:,k])/t
        if self.peaks[k] < self.rmat[-1,k]:
            self.peaks[k] = self.rmat[-1,k]
#        psi = self.peaks[k]/self.muk[k]   #This still doesn't make sense to me
        psi = self.peaks[k]/np.mean(self.muk)
#        if t > self.N:
#            print(psi)
#            print(str(k) + '\n')
        if psi >= self.Psi and self.sk[k] > 1: 
            self.mode = 'found'
            self.k_f = k
        
    def KL(self,x,y):
        '''
        Kullback-Leibler (KL) Divergence for Bernoulli rvs
        '''
        return x*logspec(x/y) + (1-x)*logspec((1-x)/(1-y))
    
    def f(self,k):
        return logspec(self.lt[k]) + self.c * logspec(logspec(self.lt[k]))
    
#    def update_bk(self,k):
        '''
        KL-UCB update for bk - may not be valid for non-Bernoulli rewards
        '''
#        self.muk = np.sum(self.rmat,0)/self.sk                              #Calculate Empirical mean                                               
#        I_vec = self.KL(self.muk[k]/self.rmat[-1,k],self.q_vec)
#        idxs = np.where(I_vec <= self.f(k)/self.sk[k] )[0]
#        self.bk[k] = np.max(self.q_vec[idxs])
#        return
    
    def update_bk(self,k,t):
        '''
        Traditional UCB update for bk
        '''
        self.muk = np.sum(self.rmat,axis = 0)/self.sk                              #Calculate Empirical mean 
        self.bk[k] = self.muk[k] + np.sqrt(2*np.log(t)/self.sk[k])        #Calculate UCB
        return 
    
    
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
        
    def run_sim(self,T=1000,tol = 1,track_flg = False):
        self.Nc = []
        self.Nc = np.zeros(T)
        self.neighborhood(tol)
        for t in range(T):

            if self.mode == 'search':

#                self.Lt.append(np.argmax(np.sum(self.rmat,0)/self.sk))                          #Choose leader at time t
                self.Lt.append(np.argmax(self.muk))
                self.lt[self.Lt[-1]] += 1                                                       #Update number of times arm is the leader 

                if np.mod((self.lt[self.Lt[-1]]-1)/(self.gamma+1),1) == 0:          #if current leader is a multiple of the expression
                    k = self.Lt[-1]                                                 #Choose current leader if condition is true
#                    self.Lt.append(k)                                               #Update leader at time t
                else:
                    k = self.N_un[np.argmax(self.bk[self.N_un])]                          #Choose k in it's neighborhood based on max KL-UCB
                

                self.N_un = np.array([k-1,k,int(np.mod(k+1,self.N))])          #Update neighborhood for k  
                self.N_un = self.N_un[np.where(self.N_un >= 0)[0]]              #verify indices are above 0 for edge case
                self.N_un = self.N_un[np.where(self.N_un < self.N)[0]]          #verify indices are below N for edge case
                self.sk[k] += 1                                                     #Update number of times arm k played
                

                x,r  = self.GetRewards(t = t,mode = 'complex')
                y = np.abs(np.matmul(np.conj(self.W[k]),x))
                    
                if t == 0:
                    self.rmat = y*np_one_hot(k,self.N)                                 #Initialize rewards tracking if first iteration
                else:
                    self.rmat = np.concatenate([self.rmat,y*np_one_hot(k,self.N)],0)    #Append rewards tracked for time t
                if track_flg:
                    self.R.append(self.channel.r_star - self.rmat[-1,k])
                self.update_bk(k,t)
                self.check_terminate(k,t)                                                                         #Check to decide terminate search
                if t == T-1:
                    self.k_f = self.Lt[-1]
                    print('TIME HORIZON REACHED.')
            else:
                break
        if self.k_f in self.N_star: self.Nc[t::] = 1
               
        return 

def np_one_hot(x,depth,axis=0):
    '''numpy function to make one hot labels from a vector of floats'''
    try: N = len(x)
    except: N = 1
    if isinstance(x,int):
         xh = np.zeros((N,depth)).astype('float64')
         xh[np.arange(N),x] = 1.      
    else:
        x = x.astype('int')
        xh = np.zeros((N,depth)).astype('float64')
        xh[np.arange(N),x] = 1.
    return xh

    
def logspec(x):
    logx = np.log(x)
    zidx = np.where(np.isinf(logx))[0]
    nidx = np.where(np.isnan(logx))[0]
    try:
        logx[zidx] = 0
        logx[nidx] = 0
    except: 
        if np.isnan(logx):
            logx = 0
        if np.isinf(logx):
            logx = 0
    return logx
        
    
if __name__ == '__main__':
    
#    filename = '/S_' +  str(0) + '_BS_' + str(1000) + '_T_' + str(300) + '_M_' + str(32) + '_CDL.hdf5'  
#    with h5py.File('./../../cdla_dataset/' + filename,'r') as f:    
#        signal_data = f['data'][:]
#    channel = cdl.CDL(signal_data = signal_data, M = M,SNR = SNR,seed = s) #-140.5 ~ 0 dB for SNR
    
    channel = rician.RicianAR1(M = 128,AoA = 40, SNR = 0, seed = 0)


    agent = uba(channel.sample_signal,channel,num_codewords=128,Psi = 1.3)
    agent.run_sim(T = 300,track_flg = False)
    
    plt.figure(0)
    plt.plot(agent.Nc)
    
    
