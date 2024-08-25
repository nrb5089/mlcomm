#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:59:29 2019

@author: nate

This class is intended to implement the Agile-Link aglorithm from 2018-Hassanieh
"""

import numpy as np
import matplotlib.pyplot as plt
import beams
plt.close('all')
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
    results as using Fcal
    '''
    
    def __init__(self,
                 GetReward,
                 Channel,
                 num_segments = 4,                              #Option of zero forces max number of permutations (R!)
                 num_hashings = 1,
                 seed = 0):

        self.GetReward = GetReward                              #Reward Generation method
        self.channel = Channel                                  #Channel object
        self.R = num_segments                                   #Number of sub-beams in each multi-armed beam 1 <= R <= np.sqrt(N)
        self.B = int(self.channel.N/self.R**2)                  #Number of bins
        self.F = self.buildF()                                  #DFT Matrix        
        self.P = int(self.channel.N/self.R)                     #Beampattern indexing
        self.d = self.channel.lam/2                             #Array spacing
        self.H = num_hashings                                 #Number of bin hashings
        self.seed = seed                                        #Set RNG Seed
        np.random.seed(seed = seed)    
              
        #Construct beampatterns.  These indices are directly from 2018-Hassanieh please reference that. 
        self.Fcal = np.expand_dims(self.F,axis = 2) 
        self.buildperm()
        self.A = np.zeros([self.B,self.channel.N,self.H]) + 0j                    #Initialize beampatterns
        self.K = np.zeros([self.B,self.R,self.H])                                 #Track indices for directions
        self.thetas = np.zeros([self.B,self.R,self.H])                            #AoAs corresponding to track indices
        self.W = []
        for h in np.arange(0,self.H):                                   #Loop through hashings
            Fc = np.expand_dims(self.F[self.hashes[h]],2)               #Initialize current row-permuted DFT matrix
            if h !=0:                                                   #First iteration is original F, which is already in Fcal
                self.Fcal = np.concatenate([self.Fcal,Fc],axis = 2)     #Add to DFT tensor of each permutation
            for b in np.arange(0,self.B):                               #Loop through bins
                for r in range(0,self.R):                               #Loop through segments
                    self.A[b,r*self.P : (r+1)*self.P,h] = \
                    self.Fcal[r*self.P + self.R*b,r*self.P : (r+1)*self.P,h]*\
                    np.exp(-1j* 2 * np.pi * np.random.randint(0,self.channel.N)/self.channel.N)                #Random phases manipulate beams
                    
                    self.K[b,r,h] = self.hashes[h,r*self.P + self.R*b]                                          #Tensor consisting of mappings to wavenumbers
                     
                        
#            self.W.append(np.matmul(self.A[:,:,h],np.transpose(np.conj(self.Fcal[:,:,0]))))        #Precalculations for simulation computations
            self.W.append(np.matmul(self.A[:,:,h],np.conj(self.F)))                                 #all intuition says use A[:,:,0] and then vary Fcal, but alas it doesn't seem to work.
        self.thetas = degs(np.pi*self.K/self.channel.N)
        self.W = np.dstack(self.W)
        self.Imat = np.abs(self.W)**2
        self.bins = np.transpose(np.reshape(self.hashes,[self.H,self.B,self.R**2]),[1,2,0])               #Tensor containing the bin to wavenumber configurations for each hashing
        self.hbins = np.append(np.sort(np.reshape(self.thetas[:,:,0],[self.B*self.R])),180)                 #Histogram bins for all possible discrete beam directions
        
        self.UCB = np.inf * np.ones([self.B,self.H])                            #Matrix to keep track of UCB calculations
        self.Nk = np.zeros([self.B,self.H])                                     #Matrix to keep track of number of times chosen
        self.muk = np.zeros([self.B,self.H])                                    #Matrix to keep track of sample means
        
    def buildF(self):
        '''
        Builds a DFT matrix F, ie. F'F =  I and FF' = I, normalized.
        '''
        k = np.arange(0,self.channel.N)
        n = np.arange(0,self.channel.N)
        n,k = np.meshgrid(k,n)
        return 1/np.sqrt(self.channel.N) * np.exp(-1j * 2 * np.pi * n * k / self.channel.N)
#        return np.exp(-1j * 2 * np.pi * n * k / self.channel.N)         
    
    def buildperm(self):
        '''
        This function builds a 2D matrix of indices permutations, the size of the 2D matrix, self.hashes 
        will be self.H X self.R.  This is slightly different from the way they do it in the paper, but
        provides permutations of the segments similarly.
        '''
        hashlim = self.channel.N
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
        
    def tie_break(self,b,h):
        try:
            lB = len(b)
            if lB > 1:
                idx = np.random.randint(0,lB)
                b = b[idx]
            else:
                b = b[0]
        except: pass
    
        try:
            lH = len(h)
            if lH > 1:
                idx = np.random.randint(0,lH)
                h = h[idx]
            else:
                h = h[0]
        except: pass
        return b,h 
        
        
    def run_sim(self,T=1000,thresh = -154):
        self.rew = []
        self.Reg = []
        for t in range(T):
            #choose arm for time t
            if self.H > 1:
                b,h = (np.argmax(self.UCB,0),np.argmax(self.UCB,1))
            else:
                b,h = (np.argmax(self.UCB,1),0)
            b,h = self.tie_break(b,h)
            
            self.Nk[b,h] += 1

            x,_ = self.GetReward(mode = 'complex')                                         #Receive array response from signal x at time t
            
#            Y = []
#            Tip = []
#            for ii in range(0,self.H):
#                y = np.abs(np.matmul(self.W[:,:,ii],x))                                                #This is how they have it in the paper, doesn't make sense
#            y = 10*np.log10(np.abs(np.matmul(self.W[b,:,h],np.matmul(np.conj(self.F),x)))) + 30        #This is how I figured it out to maximally correlate
            y = 10*np.log10(np.abs(np.matmul(np.conj(self.A[b,:,h]),x)))+ 30                            #This is correct, DO NOT DELETE previous lines, needed for records.
            y = (y - self.channel.sigrange[0])/(self.channel.sigrange[1]-self.channel.sigrange[0])  
            self.rew.append(y)
            self.Reg.append(self.channel.r_star-y)
            self.muk[b,h] = ((self.Nk[b,h]-1)*self.muk[b,h] + y)/self.Nk[b,h]                   #Update empirical mean
            self.UCB[b,h] = self.muk[b,h] + np.sqrt(2*np.log10(t)/self.Nk[b,h])                 #Update UCB
            
#            Tip.append(np.sum(np.transpose(agent.Imat[b,:,h])*y**2,1))                 #Voting mechanism for selecting direction
#            Y.append(y)                                                                 #Collect Observations
#            Y = np.transpose(np.vstack(Y))
#            Tip = 10*np.log10(np.vstack(Tip))+30                                            #Convert to dBm for a more sensible threshold
#            _,iidxs = np.where(Tip>thresh)                                                  #Determine wavenumber bins where the threshold broke
#            angles = self.wn2deg(iidxs)                                                     #Convert wavenumbers to quantized angles
#            hist += np.histogram(angles,bins = self.hbins)[0]
            
        self.rew = np.array(self.rew)
        self.Reg = np.array(self.Reg)
        return
    
    def wn2deg(self,k):
        '''
        Convert wavenumber bin k into degrees quantized in accordance with 
        the existing codebook
        '''
        theta_f = self.channel.lam/self.d * k/self.channel.N                                #Convert to degrees from wavenumber
        theta_f[theta_f > 1] = theta_f[theta_f >1]-2                                        ##
        deg_theta = degs(np.arccos(theta_f))                                                ##
        qmat = np.transpose(np.tile(self.hbins[:-1],[len(deg_theta),1]))                    #Build matrix for quantizations         
        deltas = np.abs(deg_theta - qmat)                                                   #Find minimum distances between each quantizations
        q_mins = np.argmin(deltas,0)                                                        ##
        return self.hbins[q_mins]
    
if __name__ == '__main__':
    num_hashings = 10
    channel = beams.Beams(AoA = 120)
    agent = AgileLink(channel.sample_signal,channel,num_hashings = num_hashings)
#    x,y = agent.run_sim()
    agent.run_sim()

    plt.figure(0)
    plt.plot(np.cumsum(agent.rew))
    plt.title('cumulative rewards')
    plt.figure(1)
    plt.semilogy(np.cumsum(agent.Reg))
    plt.semilogy(np.arange(0,1000),'k--')
    plt.title('cumulative regret')
    plt.grid(axis = 'both')


    