#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:46:51 2019

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
sys.path.insert(0,'/home/nate/Dissertation/Modeling/Multi-User/BA')
import array_play_codebooks
import binary_tree
import rician
from copy import deepcopy as dc
plt.close('all')

class HPM:

    def __init__(self,
                 GetReward,
                 Channel,
                 eps=0.01,
                 delta=1/128,
                 sigma_alpha = .1,
                 seed = 0):  # time stopping criteria
        
        self.GetReward = GetReward
        self.channel = Channel
        self.eps = eps
        self.delta = delta
        self.sigma_alpha = sigma_alpha
        self.N = int(1/self.delta)
        self.S = int(np.log2(self.N)) 
        np.random.seed(seed = seed)
        self.seed = seed
        
        
        # codebook W_S - note that the codebook may have additional layers, but they will not be used.
        # The highest index that should be used is log2(self.target_res)-1
#        self.W = array_play_codebooks.create_codebook(M = self.channel.M, N = self.N, Q = 128)
        if __name__ == '__main__':
            self.W = array_play_codebooks.create_alkhateeb_chiu_codebook(M = self.channel.M,N = self.N)
#        self.thetas = np.linspace(30,150,128)
#        self.W = array_play_codebooks.create_theta_specific_codebook(self.thetas,Channel.M)
        
        # initialise pi(t)_i's - distribution P(phi = theta_i | z_1:t, w_1:t) i = 1,2,.., 1/delta
        self.pi = self.delta * np.ones(self.N)
        self.inffound = False
        #initialize array of angles [0,pi] and array response
#        self.theta = np.arange(0,self.N)*np.pi/self.N
#        self.theta = np.linspace(rads(30),rads(150),self.N)
        self.theta = np.arange(0,360,120/self.N)*np.pi/180
        self.theta = self.theta[32:160]
        #This worked, but makes no sense
#        thetap = 2* np.arange(0,self.N)/self.N #lam/2 * d = 2 if d = lam/2
#        thetap[thetap>1] = thetap[thetap>1] - 2 #wrap value to fit [-1,1]
#        self.theta = np.arccos(thetap)
        
#        print(180*self.theta/np.pi)
#        for i in range(0,len(self.theta)-1):
#            print(180*(self.theta[i]-self.theta[i+1])/np.pi)
        m_grid,theta_grid = np.meshgrid(np.arange(0,Channel.M),self.theta)
#        self.athetai = 1/np.sqrt(Channel.M) * np.exp(-1j * 2 * np.pi * self.channel.lam/2 *  np.cos(theta_grid) * m_grid/self.channel.lam) 
        self.athetai = np.exp(-1j * 2 * np.pi * self.channel.lam/2 *  np.cos(theta_grid) * m_grid/self.channel.lam) 
        
        
        
        
        #Channel effects 
        self.sigma_n = Channel.sigma_n          #AWGN noise std dev
        self.sigma = sigma_alpha                         #Fading coeff estimation std dev

        #Custom methods and variables
        setattr(binary_tree.Node,'pidkl',pidkl)
        self.tree = binary_tree.Binary_Tree(self.S)
        
    
    def q_func(self,y):
        '''Quantization function'''
        return y
    
    def multi_Gauss(self,z,w):
        '''
        z : quantized observation based on signal at timestamp and application of weights
        w : beamforming weights applied to get z
        '''
        alpha_hat = 1
        sigma_n_scaled =  self.sigma_n
        mus = alpha_hat * np.matmul(self.athetai,np.conj(w))
        pdf = 1/(2*np.pi * sigma_n_scaled**2) * np.exp(-np.abs(z-mus)**2/(2*sigma_n_scaled**2))        #USED IN FIRST BATCH OF SIMS
        if np.all(pdf ==0):
            return pdf, np.argmax(-np.abs(z-mus)**2/(2*sigma_n_scaled**2))
        return pdf, False

    def multi_Gauss_fade(self,z,w):
        '''
        z : quantized observation based on signal at timestamp and application of weights
        w : beamforming weights applied to get z
        alpha_hat : estimated fading parameter ~CN(alpha0,1)
        '''
        
#        sigma_n_scaled = np.sqrt(self.channel.M) * self.sigma_n
        sigma_n_scaled = self.sigma_n
        alpha_hat =  self.channel.alpha0 #+ self.sigma_alpha * self.sigma*randcn(1)  #might have to scale with with M
        mus = alpha_hat * np.matmul(self.athetai,np.conj(w))
        pdf = 1/(2*np.pi * sigma_n_scaled**2) * np.exp(-np.abs(z-mus)**2/(2*sigma_n_scaled**2))
        
        if np.all(pdf ==0):
            return pdf, np.argmax(-np.abs(z-mus)**2/(2*sigma_n_scaled**2))
        return pdf, False    
    
    def conclude_sim(self,t,T,k_hat):
        for tc in range(t+1,T):
            if k_hat in self.N_star: self.Nc.append(1)
            else: self.Nc.append(0) 
            self.k_hats.append(k_hat)   
        self.Nc = np.array(self.Nc)
        self.k_hats = np.array(self.k_hats)
        
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''   
        x_eval = np.exp(-1j* 2* np.pi * self.channel.d * np.cos(self.channel.AoA)*np.arange(0,self.channel.M)/self.channel.lam)
        maxidx = np.argmax(np.abs(np.matmul(np.conj(self.W[-1]),x_eval)))
        
        #Use this for dft codebook
        map_idx_vec = np.argmax(np.abs(np.matmul(np.conj(self.W[-1]),np.transpose(self.athetai))),1)
        self.k_star = map_idx_vec[maxidx]
        
        #use this for thetaspecific codebook
#        self.k_star = maxidx
        if self.k_star + tol >= self.N:
            n_extra = self.k_star + tol - self.N
            self.N_star = np.concatenate([np.arange(self.k_star-tol,self.N),np.arange(0,n_extra+1)])
            return
        if self.k_star - tol < 0:
            n_extra = self.k_star - tol + self.N
            self.N_star = np.concatenate([np.arange(n_extra,self.N),np.arange(0,self.k_star + tol+1)])    
            return
        self.N_star = np.arange(self.k_star - tol,self.k_star + tol + 1)


    def report(self,current_node,l_hat,k_hat):
        print('AoA: ' + str(self.channel.aoa))
        print('seed alg: ' + str(self.seed) + '\t seed chan: ' + str(self.channel.seed))
        print('l: ' + str(l_hat) + '\t' + 'k_hat: ' + str(k_hat))
        print('k_star: ' + str(self.k_star))
        print('N_star: ' + str(self.N_star))
        print('Desc Ind: ' + str(self.k_star in current_node.final_descendents))
        print('Inf Found: ' + str(self.inffound) + '\n')
        
    def run_sim(self,T = 1000,bidx = 0,tol = 1):
        nodes = self.tree.branches
        self.neighborhood(tol)
        self.Nc = []
        self.flops = np.zeros(T)
        self.k_hats = []
        P = list()                          #Track the path
        VL = False

        for t in range(T):
            current_node = nodes[0]             #start from the top
            k_tilde = False
            for l in range(1,self.S+1):
                c0,c1 = current_node.children
                if current_node.pidkl(self.pi) > 0.5:
                    l_star = dc(l)
                    c0_val = nodes[c0].pidkl(self.pi)   #left child value
                    self.flops[t] += len(nodes[c0].final_descendents) - 1
                    c1_val = nodes[c1].pidkl(self.pi)   #right child value
                    self.flops[t] += len(nodes[c1].final_descendents) - 1
                    if c0_val > c1_val:
                        current_node = nodes[c0]
                    if c0_val < c1_val:
                        current_node = nodes[c1]
                    if c0_val == c1_val:
                        current_node = nodes[np.random.choice([c0,c1])]
                else:
                    #options in (15) in Algorithm 1 of Chiu
                    sel_0 = self.tree.get_idx(l_star,np.ceil(current_node.indices[1]/2))
                    val_0 = np.abs(nodes[sel_0].pidkl(self.pi) - 0.5)
                    self.flops[t] += len(nodes[sel_0].final_descendents) - 1
                    sel_1 = self.tree.get_idx(l_star + 1,current_node.indices[1])       
                    val_1 = np.abs(nodes[sel_1].pidkl(self.pi) - 0.5)
                    self.flops[t] += len(nodes[sel_1].final_descendents) - 1
                    if val_0 < val_1:
                        current_node = nodes[sel_0]
                    if val_1 < val_0:
                        current_node = nodes[sel_1]
                    break
                
            (l_new,k_new) = current_node.indices
            P.append(current_node.indices)
            w_new = self.W[l_new-1][k_new]

            # Codeword selection result
            x,r= self.GetReward(bidx = bidx,t = t,mode = 'complex')
            y = np.matmul(np.conj(w_new),x)  
            self.flops[t] += 2*self.channel.M -1
            
            z = self.q_func(y)
            pdfs,k_tilde = self.multi_Gauss_fade(z,w_new)
            self.flops[t] += self.N*(5)
#            pdfs,k_tilde = self.multi_Gauss(z,w_new)
            
            
            if k_tilde == False:
                sum_vec = np.sum(self.pi*pdfs)-self.pi*pdfs
                self.pi = self.pi*pdfs / sum_vec
                self.flops[t] += 4*self.N -1
            else:
                k_hat = k_tilde
                t -= 1
                break
            
            if np.isinf(np.max(self.pi)) == False:
                self.pi = 1/np.sum(self.pi) * self.pi

            else:
                self.inffound = True
                
            if np.all(np.isnan(self.pi)):
                print('All nan scenario reached, breaking simulation.')
                print('s: ' + str(self.seed))
                print('AoA: ' + str(self.channel.AoA/np.pi * 180)+ '\n')
                for tc in range(t,T):
                    self.Nc.append(0)
                self.Nc = np.array(self.Nc)
                return P 
            else:
                k_hat = self.pi.argmax()
                

            if k_hat in self.N_star: self.Nc.append(1)
            else: self.Nc.append(0) 
            self.k_hats.append(k_hat)            
            
            if (VL and np.max(self.pi) > 1- self.eps) or self.inffound: # or np.isinf(np.max(self.pi)):
                break
            
        
        k_hat = self.pi.argmax()
        self.conclude_sim(t,T,k_hat) 
        if k_hat not in self.N_star:
#        if True:
            self.report(current_node,7,k_hat)
        self.Nc = np.array(self.Nc)
        self.k_hats = np.array(self.k_hats)
        return P 
    
def pidkl(self,pi_vec):
    return np.sum(pi_vec[self.final_descendents])

def randcn(N):
    return 1/np.sqrt(2) * (np.random.randn(N) + 1j* np.random.randn(N))

def rads(deg): return deg*np.pi/180

if __name__ == '__main__':
    T = 150
    AoA =50
#    AoA = 20
    N = 128
    delta = 1/N
    s = 0
#    channel = beams.Beams(L = 4,AoA = AoA)
    channel = rician.RicianAR1(M =64,AoA = AoA,SNR = 0,seed = s)
    k_star = int(channel.b_star * N/channel.M)
        
    agent = HPM(channel.sample_signal,channel,eps = .001,delta = delta,sigma_alpha = .1,seed = s)
    P = agent.run_sim(T = T)
    print(agent.flops)
    k_hats = agent.k_hats
    plt.figure(0)
    plt.plot(agent.Nc)
#    plt.figure(1)
#    plt.plot(agent.k_hats,'x')
#    plt.ylim([0,N])
    
    