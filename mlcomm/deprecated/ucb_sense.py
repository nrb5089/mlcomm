#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:43:20 2019

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
import beams
plt.close('all')

#class Channel: 
    
    
class Agent:
    '''Within T timestamps this simulation can be configured to sense or transmit
    at certain ratio intervals
    
    fft_length : int : number of bins in the FFT for spectrum sensing
    seed : int : rng seed for np.random
    mode_ratio : number of times to transmit between sensing
    mode  : kwarg : "equal" - ratio is irrelevant, transmits and senses alternating
                    "sense" - ratio dictates mode_ratio many senses before transmit
                    "transmit" - ratio dictates mode_ratio many transmissions before sense
    '''
    
    def __init__(self,
                 GetReward, 
                 Channel,
                 mode_ratio = 1, 
                 seed = 0, 
                 eta_star = 0.75,
                 mode_init = 'sense', 
                 mode = "equal"):
        
        self.GetReward = GetReward              #Reward Genearating function
        self.channel = Channel
        self.Ti = np.zeros(Channel.M)          #number of times particular band has been sensed
        self.t = 0                              #time index initialize
        self.UCB = np.inf*np.ones(Channel.M)
        self.It = []
        self.Lidxs = np.arange(0,Channel.M,1)
        self.L = len(self.Lidxs)
        self.eta_star = eta_star
        
        #Different initialization for each case.
        if mode_init == 'sense':
            self.channel_sense()
            self.txflg = False
        if mode_init == 'transmit':          
            self.channel_transmit(random = True)         
            self.txflg = True
            
    def channel_frame(self):
        '''Generates an outcome a vector of Bernoulli rvs according to parameter
        self.p which is length self.n
        
        output shape is [1,self.n]'''           
        self.t +=1
        return np.expand_dims(self.GetReward(mode = 'all'),0)
    
    def update_UCB(self,single = False): #need a mode for one or all
        '''updates sample mean and UCB based on channel sample.  If 'single' is true
        then it updates the band according to the last (or only) element of self.It'''            
        self.phat = 1/self.Ti * np.sum(self.h,0) #update sample mean
        if single:
            try: idx = int(self.It[-1])
            except: idx = int(self.It)
            self.UCB[idx] = self.phat[idx] + np.sqrt(2*np.log(fo(self.t))/self.Ti[idx])
        else:
            self.UCB = self.phat + np.sqrt(2*np.log(fo(self.t))/self.Ti)
      
    
    def channel_sense(self):
        '''Updates (or initializes) history and updates UCB'''
        
        try: 
            self.h = np.concatenate([self.h,self.channel_frame()],0)
            self.R = np.append(self.R,self.pmax)
            self.r = np.concatenate([self.r,np.zeros([1,self.n])],0)
        except: 
            self.h = self.channel_frame()
            self.R = self.channel.r_star
            self.r = np.zeros([1,self.channel.M])
        self.Ti += 1 
        self.update_UCB()
    
    def channel_transmit(self,random = False):
        ''' Handles all functions and updates associated with transmiting a packet:
            1. Chooses channel based on max UCB
            2. Reveals channel information for that band
            3. Compute regret of decision
            4. Updates reward of playing arm
            5. Updates UCB'''
        
        #1.
        if random:
            self.It = np.append(self.It,np.array(np.random.randint(0,self.channel.M)))  #initial choice is random
        else:
            self.It = np.append(self.It,np.argmax(self.UCB[self.Lidxs]))                    #concatenate to earlier choices
        try: idx = int(self.It[-1])
        except: idx = int(self.It) 
        
        
        #2.
        sample = self.channel_frame()[0,idx]                                    #reveal results for specific band
        
        #3. 
        try:
            self.R = np.append(self.R,self.channel.r_star - sample)
        except:
            self.R = self.channel.r_star - sample
            
        self.Ti[idx] +=1                                                        #increment sensing count for one bin
        

        
        #4.
        try: 
            self.h = np.concatenate([self.h,sample * np_one_hot(idx,self.channel.M)],0)  #history has a 1 at bin selected
            self.r = np.concatenate([self.r,sample * np_one_hot(idx,self.channel.M)],0)  #provide reward if band ACK is 1
        except: 
            self.h = sample * np_one_hot(idx,self.channel.M)                             #history is one hot for band idx, initialize if it does not exist
            self.r = sample * np_one_hot(idx,self.channel.M)                             #provide reward if band ACK is 1, initialize if it does not exist

            
        #5.
        self.update_UCB(single = True)
        
    def scale_procedure(self):
        '''
        Scaling procedure in 2019-Fouche
        '''
        Si = np.sum(self.h,axis = 0)
        muihat = Si/self.Ti
        etahat = np.mean(muihat[self.Lidxs])
        if etahat <= self.eta_star: #downscale
            self.L = np.max([self.L-1,1])
            self.txflg = True
        else: #upscale
            bihat = self.UCB[np.where(self.UCB==np.sort(self.UCB)[-(self.L)])[0]] #find L+1 largest element
            if len(bihat) > 1: bihat = bihat[0]
            Bthat = self.L/(self.L + 1) * etahat + 1/(self.L + 1) * bihat
            if Bthat > self.eta_star: self.L = np.min([self.L + 1,self.n])
            self.txflg = False
    
    def run_sim1(self,T):
        '''Traditional MAB where an arm is played each time'''
        for tt in range(0,T):
            self.channel_transmit()
        return
    
    
    def run_sim2(self,T,nuT = 20):
        '''sense the channel a certain number of times, then transmit.
        Collect reward if the chosen channel is optimal'''


        #Sense for first nuT timestamps
        for tt in range(0,T):
            if tt > nuT:
                self.channel_transmit()
            if  tt <= nuT:
                self.channel_sense()
        return
    
    def run_sim3(self,T,nuT=20):
        '''initially sense, and then sense every nuT timestamps'''
                
        #Sense every nuT timestamps
        for tt in range(0,T):
            if np.mod(tt,nuT) != 0:
                self.channel_transmit()
            if np.mod(tt,nuT) == 0:
                self.channel_sense()
        
        return
    
    def run_sim4(self,T,nuT=20):
        '''Sense until meeting scaling threshold, then downscale, then do traditional MAB'''
        count = 0
        for tt in range(T):
            if count == self.L: 
                self.scale_procedure()
                count = 0
            if self.txflg:
                self.channel_transmit()
                count += 1
            else:
                self.channel_sense()
                self.scale_procedure()
                    
        
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

def fo(t,basic = True):
    if basic: return t
    else: return  1 + t*np.log(t)**2
    
    

if __name__ == '__main__':
    T = 1000
    channel = beams.Beams()
    normal_agent = Agent(channel.sample_signal,channel,mode_init = 'transmit')
    normal_agent.run_sim1(T)
    sensing_agent = Agent(channel.sample_signal,channel)
    sensing_agent.run_sim4(T)
    
    plt.figure(0)
    plt.plot(np.cumsum(normal_agent.r))
    plt.figure(1)
    plt.semilogy(np.cumsum(normal_agent.R))
        
    plt.figure(0)
    plt.plot(np.cumsum(sensing_agent.r))
    plt.title('cumulative rewards')
    plt.figure(1)
    plt.semilogy(np.cumsum(sensing_agent.R))
    plt.semilogy(np.arange(0,T),'k--')
    plt.title('cumulative regret')
    plt.grid(axis = 'both')
    
    plt.semilogy(np.arange(0,T),'k--')