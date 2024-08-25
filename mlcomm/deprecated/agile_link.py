#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:59:29 2019

@author: nate

This class is intended to implement the Agile-Link aglorithm from 2018-Hassanieh
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/media/nate/New Volume/DDrive/Education/PhD/Dissertation/Modeling/Multi-User/BA/mlcomm/mlcomm')
import beams
import pickle
import array_play_codebooks
#import niceplots
#plt.close('all')
#remember that each arm is actually a bin that is a set of arms


def rads(degs): return np.pi * degs/180
def degs(rads): return 180 * rads/np.pi



channel = beams.Beams()
GetReward = channel.sample_signal

class AgileLink:
    '''
    This class manages and executes the Agile-Link algorithm from 2018-Hassanieh.  There are two major differences
    between this algorithm implementation and the original. First of which is the generation of permutations of bins.
    The original paper had a simplistic way to generate bin permutations based on the number of antenna elements N being
    a prime number and using mod(N*indices,N) to obtain permutations.  This is replaced with just permutations of the bin
    indices here.  Secondly, the setting that governs how many directions the individual "sub-beams" point R, can be mis-
    interpreted very easily.  While choosing R does indeed produce a sub-beam that points in R directions, these directions
    exist over the space [0,2pi], but a linear array is only valid to cover the space of [0,pi]. Hence only R/2 beams will
    actually be able to be utilized.
    
    Fcal is constructed such that the hth slice of it has already been multiplied by the permutation matrix and manipulated 
    by the phase shift.
    
    Only A[:,:,0] is actually used in the algorithm, but alternatively, one could use F with any other slice and obtain the same
    results as using Frho
    '''
    
    def __init__(self,
                 GetReward,
                 Channel,
                 num_codewords = 128,
                 num_segments = 2,                              #Option of zero forces max number of permutations (R!)
                 num_hashings = 10,
                 seed = 0):

        self.GetReward = GetReward                              #Reward Generation method
        self.channel = Channel                                  #Channel object
        self.N = num_codewords
        self.R = num_segments                                   #Number of sub-beams in each multi-armed beam
#        self.B = int(self.channel.M/self.R**2)                  #Number of bins
        self.B = int(self.N/self.R**2)
#        self.F = self.buildF()                                  #DFT Matrix   
        self.F = array_play_codebooks.create_codebook(M = self.channel.M, N = self.N,Q = 128)[-1]
        self.Pm = int(self.channel.M/self.R)                     #Beampattern indexing
        self.Pn = int(self.N/self.R)
        self.d = self.channel.lam/2                             #Array spacing
        self.H = num_hashings                                   #Number of bin hashings
        self.seed = seed                                        #Set RNG Seed
        np.random.seed(seed = seed)    
              
        #Construct beampatterns.  These indices are directly from 2018-Hassanieh please reference that. 
        self.Fcal = np.expand_dims(self.F,axis = 2) 
        self.buildperm()
        self.A = np.zeros([self.B,self.channel.M,self.H]) + 0j                    #Initialize beampatterns
        self.K = np.zeros([self.B,self.R,self.H])                                 #Track indices for directions
        self.thetas = np.zeros([self.B,self.R,self.H])                            #AoAs corresponding to track indices
        self.W = []
        for h in np.arange(0,self.H):                                 #Loop through hashings
            Fc = np.expand_dims(self.F[self.hashes[h]],2)               #Initialize current row-permuted DFT matrix
            if h !=0:                                                   #First iteration is original F, which is already in Fcal
                self.Fcal = np.concatenate([self.Fcal,Fc],axis = 2)     #Add to DFT tensor of each permutation
            for b in np.arange(0,self.B):                               #Loop through bins
                for r in range(0,self.R):                               #Loop through segments
                    self.A[b,r*self.Pm : (r+1)*self.Pm,h] = \
                    self.Fcal[r*self.Pn + self.R*b,r*self.Pm : (r+1)*self.Pm,h]*\
                    np.exp(-1j* 2 * np.pi * np.random.randint(0,self.channel.M)/self.channel.M)                #Random phases manipulate beams
                    
#                    self.K[b,r,h] = self.hashes[h,r*self.Pn + self.R*b]                                          #Tensor consisting of mappings to wavenumbers
                     
                        
#            self.W.append(np.matmul(self.A[:,:,h],np.transpose(np.conj(self.Fcal[:,:,0]))))        #Precalculations for simulation computations
#            self.W.append(np.matmul(self.A[:,:,h],np.conj(self.F)))                       #all intuition says use A[:,:,0] and then vary Fcal, but alas it doesn't seem to work.
            self.W.append(np.matmul(self.A[:,:,h],np.conj(np.transpose(self.F))))
        self.thetas = degs(np.pi*self.K/self.channel.M)
        self.W = np.dstack(self.W)
        self.Imat = np.abs(self.W)**2
        self.bins = np.transpose(np.reshape(self.hashes,[self.H,self.B,self.R**2]),[1,2,0])               #Tensor containing the bin to wavenumber configurations for each hashing
        self.hbins = np.append(np.sort(np.reshape(self.thetas[:,:,0],[self.B*self.R])),180)                 #Histogram bins for all possible discrete beam directions
        
     
    def buildF(self):
        '''
        Builds a DFT matrix F, ie. F'F =  I and FF' = I, normalized.
        '''
        k = np.arange(0,self.channel.M)
        n = np.arange(0,self.channel.M)
        n,k = np.meshgrid(k,n)
        return 1/np.sqrt(self.channel.M) * np.exp(-1j * 2 * np.pi * n * k / self.channel.M)
#        return np.exp(-1j * 2 * np.pi * n * k / self.channel.M)         
  
    
    def buildperm(self):
        '''
        This function builds a 2D matrix of indices permutations, the size of the 2D matrix, self.hashes 
        will be self.H X self.R.  This is slightly different from the way they do it in the paper, but
        provides permutations of the segments similarly.
        '''
#        hashlim = self.channel.M
        hashlim = self.N
        self.hashes = []                                                                    #Initialize list of bins
        self.hashes.append(np.arange(0,hashlim))                                             #Add standard bin indices [0,1,2,...,N-1]
        kk = 1                                                                              #While loop condition counter
        while kk<self.H:
            hash_c = np.random.choice(np.arange(0,hashlim),hashlim,replace = False)           #Make random permutation of indices
            cond = len(np.where((np.vstack(self.hashes)==hash_c).all(axis = 1))[0])         #Condition to check if this permutation is already in
            if cond == 0:                                                                   #If this permutation is not in, then append it to the list
                self.hashes.append(hash_c)
                kk +=1
            else: pass
        self.hashes = np.vstack(self.hashes)                                                #Convert list to 2D matrix.
        
    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct'''
        self.k_star = int(self.N*0.5*np.cos(self.channel.aoa*np.pi/180))
        if self.k_star < 0: self.k_star += self.N    

        if self.k_star + tol > self.N:
            n_extra = self.k_star + tol - self.N
            self.N_star = np.concatenate([np.arange(self.k_star-tol,self.N),np.arange(0,n_extra+1)])
            return
        if self.k_star - tol < 0:
            n_extra = self.k_star - tol + self.N
            self.N_star = np.concatenate([np.arange(n_extra,self.N),np.arange(0,self.k_star + tol+1)])    
            return
        self.N_star = np.arange(self.k_star - tol,self.k_star + tol + 1)
        
        
#    def run_sim(self,T=1000,thresh = -145,tol = 1):
#        '''
#        This is an implementation of the Agile Link Algorithm that utilizes all hashes 
#        and beam patterns for each sample received
#        '''
#
#        
#        self.Nc = []
#        Ds = np.zeros(self.channel.M)
#        ks = []
#
#        for t in range(T):
#            x,rt = self.GetReward(mode = 'complex')                                        #Receive array response from signal at time t
#            
#            Y = []
#            Tip = []
#            for h in range(0,self.H):
#                y = np.abs(np.matmul(np.conj(self.A[:,:,h]),x))                             #Take single measurement
#                Tip.append(1/self.B * np.sum(np.transpose(self.Imat[:,:,h])*y**2,1))                 #Voting mechanism for selecting direction
#                Y.append(y)                                                                 #Collect Observations
#            Y = np.transpose(np.vstack(Y))
#            Tip = 10*np.log10(np.vstack(Tip))+30                                            #Convert to dBm for a more sensible threshold
#            D = np.array(Tip > thresh).astype('int')    #Indicate where the threshold has been broken
#            Ds += np.sum(D,0)                           #Sum over all hashes and add to running total
#            k = np.argmax(Ds)                           #choose max
#            ks.append(k)
#            if k in range(self.channel.b_star-tol,self.channel.b_star+tol + 1): self.Nc.append(1)
#            else: self.Nc.append(0)
#   
#        return 

    def run_sim(self,T=1000,bidx = 0,thresh = 30,tol = 1):
        '''
        This simulation is designed to choose one bin from one hash each timestamp in order
        to better compare it with the other MAB algorithms.  A new sample is taken each timestamp.
        '''
        self.Nc = []
        self.neighborhood(tol)
        h = 0                                                   #Index indicating current hash 
        b = 0                                                   #Index indicating bin for current beamforming weights (row of A[:,:,h])
        Tip = np.zeros([self.N])
        Ds = np.zeros([self.N])
        for t in range(T):
            x,rt = self.GetReward(bidx = bidx,t = t,mode = 'complex')
            y = np.abs(np.matmul(np.conj(self.A[b,:,h]),x))
            Tip += (self.Imat[b,:,h] * y**2)
#            print(np.max(10*np.log10(Tip)) + 30)
            Ds += np.array(10*np.log10(Tip)+30 > thresh).astype('int')
            k = np.argmax(Ds)
#            print(k)
            if k in self.N_star: self.Nc.append(1)
            else: self.Nc.append(0)
            b += 1
            if b == self.B:
                b = 0
                h += 1
                if h == self.H:
                    h = 0
                    Tip = np.zeros([self.N])
        self.Nc = np.array(self.Nc)
        return

        
        
    def wn2deg(self,k):
        '''
        Convert wavenumber bin k into degrees quantized in accordance with 
        the existing codebook
        '''
        try:
            theta_f = self.channel.lam/self.d * k/self.channel.M                                #Convert to degrees from wavenumber
            theta_f[theta_f > 1] = theta_f[theta_f >1]-2                                        ##
            deg_theta = degs(np.arccos(theta_f))                                                ##
            qmat = np.transpose(np.tile(self.hbins[:-1],[len(deg_theta),1]))                    #Build matrix for quantizations         
            deltas = np.abs(deg_theta - qmat)                                                   #Find minimum distances between each quantizations
            q_mins = np.argmin(deltas,0)                                                        ##
            return self.hbins[q_mins]
        except:
            theta = self.channel.lam/self.d * k/self.channel.M
            if theta > 1: theta = theta - 2
            deg_theta = degs(np.arccos(theta))
            delta = np.abs(self.hbins[:-1] - deg_theta)
            q_min = np.argmin(delta)
            return self.hbins[q_min]
        
if __name__ == '__main__':
    T = 10000
    S = 10
    num_hashings = 10 #H
    num_segments = 2#R
    M = 32
    N = 128
    B = int(N/num_segments**2)
    print('Flops per measurement: ' + str(4*num_hashings*B*N))
    L = 0
#    AoA = 60
    tol = 2
    np.random.seed(seed = 0)
    SNR = 15
    Ncs = []
    for s in range(S):  
        AoA = np.random.uniform(0,180)
#        channel = beams.Beams(M = M,L = L, N0 = -190,AoA = AoA,seed = s)
        channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s) #-140.5 ~ 0 dB for SNR
        agent = AgileLink(channel.sample_signal,channel,num_codewords = N, num_segments=num_segments, num_hashings = num_hashings,seed = s)            
        agent.run_sim(T = T,tol = tol)
        Ncs.append(agent.Nc)
    Ncs = np.vstack(Ncs)
    Pc = np.mean(Ncs,0)
    plt.figure(0)
    plt.plot(Pc)
    plt.xlabel('Number of Measurements')
    plt.ylabel('Probability of Correct Detection')
    plt.title('Timestamps: ' + str(T) + ', Trials: ' + str(S) + ', Hashes: ' + str(num_hashings) + ', NLOS paths: ' + str(L))
    plt.grid(axis = 'both')
    
    #placeholders for dict
    R = 0
    r = 0
    datadict = {'Regret' : R, 'Reward' : r, 'Pc' : Pc}
    path = '../../data'
    filename = '/AL_T_' + str(T) + '_S_' + str(S) + '_N_' + str(channel.M) + '_NLOS_' + str(channel.L) + '_H_' + str(num_hashings) + '_R_' + str(num_segments)
#    f = open(path + filename +'.pkl','wb')
#    pickle.dump(datadict,f)
#    f.close() 
    
    fig0 = plt.figure(1)
    for ii in range(0,6):
        ax0 = fig0.add_subplot(int('32' + str(ii+1)),polar = False)
        for b in range(agent.B):
            plt.plot(degs(np.linspace(0,2*np.pi,agent.channel.M)),np.abs(agent.W[b,:,ii]))
            plt.ylim([0,0.5])
    
#    plt.figure()
#    plt.stem(agent.hbins[:-1],hist)

#    plt.figure(1)
#    plt.subplot(agent.B+1,1,1)
#    plt.plot(np.abs(Fpx))
#
#    for b in range(0,agent.B):
#        plt.subplot(agent.B+1,1,b+2)
#        plt.plot(degs(np.linspace(0,2*np.pi,agent.channel.M)),np.abs(agent.W[b,:,0]))
#        plt.ylabel([b])
#
#    plt.figure(2)
#    plt.subplot(agent.B+1,1,1)
#    plt.plot(np.abs(Fpx))
#
#    for b in range(0,agent.B):
#        plt.subplot(agent.B+1,1,b+2)
#        plt.plot(degs(np.linspace(0,2*np.pi,agent.channel.M)),np.abs(agent.W[b,:,0]))


    