#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 08:32:20 2020

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/media/nate/New Volume/DDrive/Education/PhD/Dissertation/Modeling/Multi-User/BA/mlcomm/mlcomm')
#import niceplots
import array_play_codebooks
import rician
import pickle

class Agent:
    def __init__(self,
                 Channel,
                 GetReward,
                 max_resolution,
                 delta_p,
                 epsilon = .001,
                 beta = 1,
                 nu = .001,
                 seed = 0,
                 mode = 'proc'):
        
        

        self.channel = Channel
        self.GetReward = GetReward
        self.M = Channel.M
        self.N = max_resolution
        self.l = int(np.log2(max_resolution))-1
        self.epsilon = epsilon
        self.beta = beta
        self.nu = nu
        self.lam_p = ((beta + 2)/beta)**2
        if mode == 'sim':
            delta_sim = np.log(1+epsilon)*(nu*epsilon/(2+epsilon))**(1/(1+epsilon))
            self.delta = delta_p * delta_sim
        else:
            delta_max = np.log(1 + epsilon)/np.exp(1)
            self.delta = delta_p * delta_max
            
        self.Nts = np.zeros([self.N])
        self.mu_hats = np.zeros([self.N])
        self.sigma_hats = np.zeros([self.N])
        self.UCBs = np.inf * np.ones([self.N])
        self.LCBs = -np.inf * np.ones([self.N])
        
        self.Omega = np.arange(0,self.N)            #initialize list to track arms
        
        #Initialize Codebook
        self.W = array_play_codebooks.create_codebook(M = self.M, N = 128, Q = 128)
        self.seed = seed
        
    def update(self,y,t,idx):
        self.Nts[idx] += 1
        self.mu_hats[idx] = ((self.Nts[idx] -1)*self.mu_hats[idx] + y)/self.Nts[idx]
        self.sigma_hats[idx] = np.sqrt(((self.Nts[idx]-1)*self.sigma_hats[idx]**2 + (y-self.mu_hats[idx])**2)/self.Nts[idx]) #std dev         
        sigma = np.sqrt(0.15) #default worst case scenario
    #    sigma = self.sigma_hat
        
        #max sigma
        #standard bias from procedure:
#        U = (1+ self.beta) * (1+ np.sqrt(self.epsilon)) * np.sqrt(2* sigma**2 *(1 + self.epsilon)*np.log(np.log((1 + self.epsilon)*self.Nts[idx])/self.delta)/(self.Nts[idx])) 
        
        #bias from simulations:
        U =  (1+ np.sqrt(self.epsilon)) * np.sqrt(2* sigma**2 *(1 + self.epsilon)*np.log(   2*np.log(((1 + self.epsilon)*self.Nts[idx]+2)/(self.delta/self.N))    )/(self.Nts[idx])) 
        
        #empirical sigma
        #standard bias from procedure:
#        U = (1+ self.beta) * (1+ np.sqrt(self.epsilon)) * np.sqrt(2* self.sigma_hats[idx]**2 *(1 + self.epsilon)*np.log(np.log((1 + self.epsilon)*self.Nts[idx])/self.delta)/(self.Nts[idx]))  
        
        #bias from simulations:
    #    U =  (1+ np.sqrt(self.epsilon)) * np.sqrt(2* self.sigmas[idx]**2 *(1 + self.epsilon)*np.log(   2*np.log(((1 + self.epsilon)*self.Nts[idx]+2)/(self.delta/self.N))    )/(self.Nts[idx]))
        
        if self.Nts[idx] > 0:
            self.UCBs[idx] = self.mu_hats[idx] + U
            self.LCBs[idx] = self.mu_hats[idx] - U
    #    print('Node: ' +str(self.indices) + '\t mu: ' + str(self.mu_hat) + '\t UCB: ' + str(self.UCB) + '\t LCB: ' + str(self.LCB) + '\t U : ' + str(U))
#        print('Arm: ' +str(idx) + '\t mu: ' + str(self.mu_hats[idx]) + '\t UCB: ' + str(self.UCBs[idx]) + '\t U : ' + str(U))
    

    def neighborhood(self,tol):
        '''Determines set of beams for which the estimate is correct
        '''
        self.k_star = self.N*0.5*np.cos(self.channel.aoa*np.pi/180)
        if self.k_star < 0: self.k_star += self.N    
        self.k_star = int(self.k_star)

# 
#        print('AoA: ' + str(self.channel.aoa) + '\t k_star calc: ' + str(tempk) + '\t' + 'k_star actual: ' + str(self.k_star))      
        
        if self.k_star + tol > self.N:
            n_extra = self.k_star + tol - self.N
            self.N_star = np.concatenate([np.arange(self.k_star-tol,self.N),np.arange(0,n_extra + 1)])
            return
        if self.k_star - tol < 0:
            n_extra = self.k_star - tol + self.N
            self.N_star = np.concatenate([np.arange(n_extra,self.N),np.arange(0,self.k_star + tol+1)])    
            return
        self.N_star = np.arange(self.k_star - tol,self.k_star + tol + 1)
        
        
        
    def run_sim(self,T,tol = 1):
        self.neighborhood(tol)

        Pc = []
        P = []
        for tt in range(T):
            It = np.argmax(self.UCBs)
            P.append(self.Omega[It])
            w_hat = self.W[self.l][It]
            x,r = self.GetReward(t = tt+1,mode = 'complex')
            y = np.abs(np.matmul(np.conj(w_hat),x))
            
            self.update(y,tt+1,It)
                
            if self.Omega[It] in self.N_star: Pc.append(1)
            else: Pc.append(0)
            
            if np.any(self.UCBs[It] < self.LCBs):
                np.delete(self.UCBs,It)
                np.delete(self.LCBs,It)
                np.delete(self.mu_hats,It)
                np.delete(self.Nts,It)
                np.delete(self.sigma_hats,It)
        return np.array(Pc),P
        



if __name__ == '__main__':
    path = './../../data/StopStudy/'
    M = 32
    max_resolutions = [2,4,8,16,32,64,128]
#    max_resolutions = [128]
    beta = 1 #beta
    epsilon = .01 #epsilon
    nu = .001
    delta_p = .6

    tol = 0
    SNR = 0
    T = 1000
    S = 10000
    
    #Routine for running simulations and saving
    if True:
        saveflg = True   
        
        Pcs = np.zeros(T)
        sigs = np.zeros(T)
        for max_resolution in max_resolutions:
            filename = '/M_' + str(M) + '_N_' + str(max_resolution) + '_S_' + str(S) + '_T_' + str(T) + '_SNR_' + str(SNR)
            print('starting ' + str(max_resolution))
            for s in range(S):
                AoA = np.random.uniform(20,160)
                channel = rician.RicianAR1(M = M,AoA = AoA,SNR = SNR,seed = s)
                agent = Agent(channel,channel.sample_signal,max_resolution = max_resolution,delta_p = delta_p, mode = 'sim')
                Pc,P = agent.run_sim(T,tol = tol)
    
                Pcs = (s*Pcs + Pc)/(s+1)
                sigs = np.sqrt((s*sigs**2 + (Pc - Pcs)**2)/(s+1))
            conf = 4.302653 * np.array(sigs)/np.sqrt(S)
            
            if saveflg:
                datadict = {'Pcs' : Pcs, 'conf' : conf, 'delta_p' : delta_p, 'epsilon' : epsilon, 'beta' : beta, 'sigs' : sigs}
                f = open(path + filename + '_ae_rician.pkl','wb')
                pickle.dump(datadict,f)
                f.close()          

    #Routine to run plots
    if True:
        colors = ['r','g','b','c','y','m','k']
        mkstr = ['>','o','s','D','*','1','h']
        for ii,max_resolution in enumerate(max_resolutions):
            filename = '/M_' + str(M) + '_N_' + str(max_resolution) + '_S_' + str(S) + '_T_' + str(T) + '_SNR_' + str(SNR) + '_ae_rician.pkl'
            data = pickle.load(open( path + filename, 'rb' ))
            plt.figure(0)
            plt.plot(data['Pcs'],mkstr[ii],ls = '-',markevery = 50,color = colors[ii])
#            plt.fill_between(np.arange(0,T), 
#                     data['Pcs'] - data['conf'], 
#                     data['Pcs'] + data['conf'],
#                     color = (0,0,0,0.2))
        plt.grid(axis = 'both')
        plt.legend(['2','4','8','16','32','64','128'])
        plt.title('Individual Layer MAB Games with Empirical Var')
        plt.xlim([0,T])
        plt.ylim([0,1])
        