#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:49:06 2019

@author: nate
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/media/nate/New Volume/DDrive/Education/PhD/Dissertation/Modeling/Multi-User/BA/mlcomm/mlcomm')
import os
#plt.close('all')
#import beams
import uba2
import hba
#import hier_ucb
#import hier_ucb2
#import hier_ucb_alt
#import hier_ucb3
#import hier_successive_elim
import OSUB2
#import hier_rapid_descent
import agile_link
import hpm2
import beam_sweep
import pickle
import rician
import array_play_codebooks

T = 150
S = 10000
tol = 1 
M = 64
N = 128
L = 5
#alg = 'HUCB'  
#algs = ['HUCB','UBA','HPM','AL','HBA']
#algs = ['HPM']
alg = 'HBA'

#OSUB and GHS
approach = '2'


    
start_layers = [3]
#start_layers = [1,2,3,4]

almode = 2
Psi = 1 # 1.05
zeta = 1.1
num_segments = 2
num_hashings = 10

#HUCB:
delta = .0001
c = 5
epsilons = [7]
#epsilons = [1.4,2.1,2.2,2.4,2.6,2.8,3.2,3.6,4.5,7]


eps = .001

pltflg = True
saveflg = True

W = array_play_codebooks.create_alkhateeb_chiu_codebook(M,N)
#W = array_play_codebooks.create_codebook(M,N,Q = 128)

SNRs = [-5,-2.5,0,2.5]
#SNRs = [0]
if S <= 1000: 
    Paths = []
#Deltas = np.zeros(7)
for SNR in SNRs:
    for start_layer in start_layers:
        for epsilon in epsilons:

            print('Algorithm: ' + alg + '\n' 'T: ' + str(T) + '\n' + 'S: ' + str(S) + '\n' + 'SNR: ' + str(SNR) + '\n' + 'tol: ' + str(tol) + '\n')
            mu_Ncs = np.zeros(T)
            sig_Ncs = np.zeros(T)
            mu_flops = np.zeros(T)
            for s in range(0,S):
    
    
                AoA = np.random.uniform(30,150)
    #            print('s: ' + str(s) + ' AOA: ' + str(AoA))
                channel = rician.RicianAR1(M = M,L = L,AoA = AoA,SNR = SNR,seed = s) 
            
                if alg =='UBA':       
#                    agent = uba.uba(channel.sample_signal,channel,num_codewords = N,Psi = Psi)
                    agent = uba2.UBA(channel.sample_signal,channel,max_resolution,Psi = Psi,seed =s)
                if alg == 'HBA':
                    agent = hba.hba(channel.sample_signal,channel,num_codewords = N,seed = s,zeta = zeta)
                if alg == 'AL':
                    agent = agile_link.AgileLink(channel.sample_signal,channel,num_segments = num_segments,num_hashings = num_hashings,seed = s)
                if alg == 'HUCB':
    #                agent = hier_ucb3.Hier_UCB(channel.sample_signal,channel,max_resolution = N,zeta = .63,c = .01,seed = s)  #zeta = .12 and c = .05    
                    agent = hier_successive_elim.Hier_UCB(channel.sample_signal,channel,max_resolution = N,c=c,delta = delta, seed = s)
                if alg == 'GHS':
                    agent = G_LSE2.LSE(channel.sample_signal,channel,max_resolution = N, start_layer = start_layer, epsilon = epsilon,delta = delta,seed =s)
                if alg == 'OSUB':
#                    agent = OSUB.OSUB(channel.sample_signal,channel,max_resolution = N, start_layer = start_layer,delta = delta,seed =s)
                    agent = OSUB2.OSUB(channel.sample_signal,channel,max_resolution = N, start_layer = start_layer,delta = delta,seed =s)
                if alg == 'HRD':
                    agent = hier_rapid_descent.HierRapidDescent(channel.sample_signal,channel,max_resolution = N,seed =s)
                if alg == 'HPM':
                    agent = hpm2.HPM(channel.sample_signal,channel,eps = eps,delta = 1/N,sigma_alpha = .1,seed = s)
                if alg =='BS':
                    agent = beam_sweep.Agent(channel,channel.sample_signal, resolution = N, seed = s)                    
                    
                if True:
                    agent.W = W
                    if alg == 'UBA' or alg == 'HBA' or alg == 'BS':
                        agent.W = W[-1]
    
                    

                    
                        
                
                if alg == 'HUCB':
                    agent.run_sim(T = T,start_layer = start_layer,tol = tol)
                else:
                    if S <= 1000:  
                        Paths.append(agent.run_sim(T = T,tol = tol))
                    else:
                        agent.run_sim(T = T, tol = tol)
                        
                if s > 0:
                    mu_Ncs = (s*mu_Ncs + agent.Nc)/(s+1)
                    sig_Ncs = np.sqrt((s*sig_Ncs**2 + (agent.Nc - mu_Ncs)**2)/(s+1))
                    mu_flops = (s*mu_flops + agent.flops)/(s+1)
                else:
                    mu_Ncs = agent.Nc
                    mu_flops = agent.flops
                    
    #        Deltas = Deltas/S
            Pc = np.array(mu_Ncs)
            conf = 4.302653 * np.array(sig_Ncs)/np.sqrt(S+1) #two degrees of freedom with 95% confidence for statistical variance est
               
            
            if pltflg:
                plt.figure(2)
                plt.plot(Pc)
                plt.title('Probability of correct arm chosen (' + str(S) + ' trials)')
                plt.xlabel('Time')
                plt.ylabel('Probability of Correct Arm Chosen')
                plt.ylim([0,1])
                plt.fill_between(np.arange(0,T), 
                                 Pc - conf, 
                                 Pc + conf,
                                 color = (0,0,0,0.2))
                plt.grid(axis = 'both')

            path = './../../data/' + alg + '_performance_comp'

#            win_path = 'D:\\DDrive\\Education\\PhD\\Dissertation\\Modeling\\Multi-User\\BA\\data\\' + alg + '_performance'
            if alg =='UBA':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol) + '_Psi_' + str(agent.Psi)
                datadict = {'Pc' : Pc, 'conf' : conf}
            if alg == 'HBA':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol)  + '_zeta_' + str(agent.zeta) 
                datadict = {'Pc' : Pc, 'conf' : conf, 'mu_flops' : mu_flops}
            if alg == 'AL':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol) + '_H_' + str(num_hashings) + '_R_' + str(num_segments) + '_mode_' + str(almode)
            if alg == 'HUCB':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol)  + '_sl_' + str(start_layer)    
                datadict = {'Pc' : Pc, 'conf' : conf, 'c' : c, 'delta' : delta}
            if alg == 'HPM':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol) + '_eps_' + str(eps) 
                datadict = {'Pc' : Pc, 'conf' : conf, 'mu_flops' : mu_flops}
            if alg == 'BS':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol)
                datadict = {'Pc' : Pc, 'conf' : conf}
            if alg == 'GHS':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol) + '_eps_' + str(epsilon) + '_app_' + approach + '_sl_' + str(start_layer)
                datadict = {'Pc' : Pc, 'conf' : conf}
            if alg == 'OSUB':
                filename = '/' + alg + '_T_' + str(T) + '_S_' + str(S) + '_N_' + str(N) + '_tol_' + str(tol) + '_app_' + approach + '_sl_' + str(start_layer)
                datadict = {'Pc' : Pc, 'conf' : conf, 'mu_flops' : mu_flops}
            if saveflg:
                if not os.path.exists(path):
                    os.makedirs(path)
                f = open(path + filename + '_SNR_' + str(SNR) + '_L_' + str(L) + '_rician.pkl','wb')
                pickle.dump(datadict,f)
                f.close()       
#plt.legend(['3','4','5'])


