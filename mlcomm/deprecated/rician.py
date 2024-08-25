#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:14:03 2019

@author: nate
"""

import numpy as np
import sys
sys.path.insert(0,'/media/nate/New Volume/DDrive/Education/PhD/Dissertation/Modeling/Multi-User/BA/mlcomm/mlcomm')
import matplotlib.pyplot as plt
import array_play_codebooks
from copy import deepcopy as dc
plt.close('all')
c = 3e8

class RicianAR1:
    def __init__(self,
                 M = 128,
                 mu= 1,
                 Kr = 10,
                 g = 0.024451,
                 L = 3,
                 sigrange = [1e-6,10],
                 SNR = 6,
                 fc = 60e9,
                 AoA = 20,
                 seed = 0,
                 mode = 'complex'):
        
        self.M = M
        self.mu = mu            
        self.Kr = Kr
        self.g = g
        self.L = L
        self.sigrange = sigrange 
        self.SNR = SNR
        self.fc = fc
        self.lam = c/fc
        self.d = self.lam/2
        self.seed = seed
        
        np.random.seed(seed)                                #Set RNG Seed
        self.alpha = np.sqrt(self.Kr/(1+self.Kr))*self.mu \
                    + np.sqrt(1/(1+self.Kr))*randcn(1)      #Initialize alpha (fading param)
        self.alpha0 = dc(self.alpha)
#        self.alpha = [1]
        self.sigma_n = np.sqrt(10**(-self.SNR/10))
        self.aoa = AoA                                      #AoA in degrees (needed for calls in sims)
        self.AoA = rads(AoA)                                #AoA for LOS Path
        self.AoAs_NLOS = np.random.uniform(30,120,self.L-1)
        self.AoAs = np.concatenate([[self.AoA],rads(self.AoAs_NLOS)]) 
        self.F = buildF(M = M,N = M)
        self.b_star = int(self.M * self.d * np.cos(self.AoA) / self.lam)
        if self.b_star < 0:
            self.b_star = self.b_star + int(self.M)
        

    def sample_signal(self,bidx = 0, idx = 0, t = 0, fading = True,mode = 'single'):
        self.update_alpha_time_varying()
#        self.update_alpha_static()
#        print(self.alpha[0])

#        #Fading
#        if fading:
#            xm = self.alpha[0] * np.exp(-1j* 2* np.pi * self.d * np.cos(self.AoA)*np.arange(0,self.M)/self.lam) \
#                + self.sigma_n * randcn(self.M)                                             #fading*signal + WGN 
        
        #Fading with multipath:
        gains = np.concatenate([[1.],.25*np.ones(self.L)])
        xm = np.zeros(self.M) + 0j
        if fading:
            for ll in range(self.L):
                xm += self.alpha[0] * gains[ll] *  np.exp(-1j* 2* np.pi * self.d * np.cos(self.AoAs[ll])*np.arange(0,self.M)/self.lam)
            xm += self.sigma_n * randcn(self.M)                                             #fading*signal + WGN 
                
        #No fading
        if not fading:
            xm = np.exp(-1j* 2* np.pi * self.d * np.cos(self.AoA)*np.arange(0,self.M)/self.lam) \
                + self.sigma_n * randcn(self.M)                                             #fading*signal + WGN
            
        y = np.abs(np.matmul(np.conj(self.F),xm))
        r = (y-self.sigrange[0])/(self.sigrange[1]-self.sigrange[0])
        if mode == 'single':
            return r[idx]
        if mode == 'complex':
            return xm,r
    
    def update_alpha_time_varying(self):
        self.alpha = np.sqrt(self.Kr/(1+self.Kr))*self.mu \
                + (self.alpha - np.sqrt(self.Kr/(1+self.Kr))*self.mu) * np.sqrt(1-self.g)\
                            + randcn(1) * np.sqrt(self.g/(1+self.Kr))
                            
#        print(self.alpha)
    def update_alpha_static(self):
        self.alpha = self.alpha

class RicianAR1_2D:
    def __init__(self,
                 Mx = 16,
                 My = 16,
                 mu= 1,
                 Kr = 10,
                 g = 0.024451,
                 sigrange = [1e-6,10],
                 SNR = 6,
                 fc = 60e9,
                 AoA = 20,
                 ZoA = 20,
                 seed = 0,
                 mode = 'complex'):
        
        self.Mx = Mx
        self.My = My
        self.mu = mu            
        self.Kr = Kr
        self.g = g
        self.sigrange = sigrange #signal power amplitude in dBm!
        self.SNR = SNR
        self.fc = fc
        self.lam = c/fc
        self.d = self.lam/2
        self.seed = seed
        
        np.random.seed(seed)                                #Set RNG Seed
        self.alpha = np.sqrt(self.Kr/(1+self.Kr))*self.mu \
                    + np.sqrt(1/(1+self.Kr))*randcn(1)      #Initialize alpha (fading param)
        self.alpha0 = dc(self.alpha)
        self.sigma_n = np.sqrt(10**(-self.SNR/10))
        self.aoa = AoA                                      #AoA in degrees (needed for calls in sims)
        self.zoa = ZoA
        self.AoA = rads(AoA)                                #AoA for LOS Path
        self.ZoA = rads(ZoA)
        self.Fx = buildF(M = Mx,N = Mx)
        self.Fy = buildF(M = My,N = My)
#        self.b_star = int(self.M * self.d * np.cos(self.AoA) / self.lam)
#        if self.b_star < 0:
#            self.b_star = self.b_star + int(self.M)
        
        #Build array response
        x = np.arange(0,Mx)
        y = np.arange(0,My)
        x,y = np.meshgrid(x,y)
        self.Omega = 1/np.sqrt(Mx*My) * np.exp(-1j * 2* np.pi* self.d/self.lam * np.sin(self.ZoA)*(np.cos(self.AoA)*x + np.sin(self.AoA)*y))

    def sample_signal(self,bidx = 0, idx = 0, t = 0, mode = 'single'):
    
        self.update_alpha_time_varying()
#        self.update_alpha_static()
#        xm = self.alpha[0] * self.Omega + self.sigma_n * randcn2D(self.Mx,self.My) #fading*signal + WGN  
        xm = 1 * self.Omega + self.sigma_n * randcn2D(self.Mx,self.My) #fading*signal + WGN   
        y = 10*np.log10(np.abs(np.matmul(np.matmul(np.conj(self.Fx),xm),np.transpose(np.conj(self.Fy)))))                     #convert to dBW
        r = (y-self.sigrange[0])/(self.sigrange[1]-self.sigrange[0])
        if mode == 'single':
            return r[idx]
        if mode == 'complex':
            return xm,r
    
    def update_alpha_time_varying(self):
        self.alpha = np.sqrt(self.Kr/(1+self.Kr))*self.mu \
                + (self.alpha - np.sqrt(self.Kr/(1+self.Kr))*self.mu) * np.sqrt(1-self.g)\
                            + randcn(1) * np.sqrt(self.g/(1+self.Kr))
                            
#        print(self.alpha)
    def update_alpha_static(self):
        self.alpha = self.alpha
        
def buildF(M = 128, N = 128):
    '''
    Builds a normalized DFT matrix F, ie. F'F = I and FF' = I
    N >= M should be 2^int
    '''
    Q = M/N
    n = np.arange(0,M,Q)
    m = np.arange(0,M)
    m,n = np.meshgrid(m,n)
    return 1/np.sqrt(M) * np.exp(-1j *  2* np.pi * m * n / M)
        
def randcn(M):
    #according to wikipedia, this shouldn't be normalized by 1/np.sqrt(M), ie. each component is CN(0,1)
    #and then each 
#    return  1/np.sqrt(2*M) * (np.random.randn(M) + 1j* np.random.randn(M))
    return  1/np.sqrt(2) * (np.random.randn(M) + 1j* np.random.randn(M))

def randcn2D(Mx,My):
    return 1/np.sqrt(2*Mx*My) * (np.random.randn(Mx,My) + 1j* np.random.randn(Mx,My))

def degs(rad):return 180*rad/np.pi
def rads(deg):return np.pi*deg/180
def db2lin(dbs):return 10**(dbs/10)
def lin2db(lins):return 10*np.log10(lins)

if __name__ =='__main__':
  
    if True:
        M = 64
        N = 128
        F = buildF(M = M, N = 128)
        W = array_play_codebooks.create_codebook(M = M,N = N, Q = 128)  
        Deltas = []
        Delta_nns = []
        var_all = []
        mu_all = []
        samples = []
        totals = []
        channel = RicianAR1(M = M,AoA = 20,L = 5, SNR= 0,seed = 2,Kr = 10)

        for ll in range(0,7):
            ys = []
            for ii in range(0,1000):
                x,_ = channel.sample_signal(mode = 'complex')
#                y = 10*np.log10(np.abs(np.matmul(np.conj(W[ll]),x))) 
                y = 10*np.log10(np.abs(np.matmul(np.conj(W[ll]),x)))
#                y = (y-channel.sigrange[0])/(channel.sigrange[1]-channel.sigrange[0])  #Obtain rewards and normalize
#                plt.figure(ll)
#                plt.plot(y)
                ys.append(y)
            mus = np.mean(ys,0)
            var = np.var(ys,0)
            Delta = np.max(mus)-mus
            Delta_nn = np.abs(mus[0::2] - mus[1::2])
            Deltas.append(Delta)
            Delta_nns.append(Delta_nn)
#            samp = 1/(Delta**2) * np.log(np.log(1/(Delta**2)))
#            samp[samp < 0] = 0
#            samples.append(samp)
#            totals.append(np.sum(samples[ll][samples[ll] != np.inf]))
            var_all.append(var)
            mu_all.append(mus)
    #        plt.ylim([0, 1])

        ys = np.vstack(ys)
#        var = np.var(ys,0)
#        mus = np.mean(ys,0)
        
        samples = []
        min_samps = []
        for ll in range(0,7):
#            samp = 1/(Delta_nns[ll]**2) * np.log(np.log(1/(Delta_nns[ll]**2)))
            samp = 1/(Delta_nns[ll]**2) * np.log(2/(.001*Delta_nns[ll]))
            min_samps.append(np.min(samp))
            samples.append(samp)
        print('Samples needed: ' + str(2*np.sum(min_samps)))
        
    for ii in range(7):  
        plt.figure(ii+7)
        plt.plot(mu_all[ii])
        plt.grid(axis = 'both')
    if False:
        Mx = 8
        My = 8
        AoA = 40
        ZoA = 40
        SNR = 0
        
        Fx = buildF(M = Mx, N = 128)
        Fy = buildF(M = My, N = 128)
        channel = RicianAR1_2D(Mx = Mx,My = My,AoA = AoA, ZoA = ZoA, SNR = SNR)
        for ii in range(0,1000):
            x,_ = channel.sample_signal(mode = 'complex')
            y = np.matmul(np.matmul(np.conj(Fx),x),np.transpose(np.conj(Fy)))
            y = 10*np.log10(np.abs(y))
            y = (y-channel.sigrange[0])/(channel.sigrange[1]-channel.sigrange[0])  #Obtain rewards and normalize
            if np.any(y > 1):
                print(np.max(y))
            if np.any(y < 0):
                print(np.min(y))
#        plt.figure(0)
#        plt.imshow(y)
#        print(np.where(y == np.max(y)))
#        plt.ylim([channel.sigrange[0], channel.sigrange[1]])
        ys = np.vstack(ys)
        var_all = np.var(ys,0)

