#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:59:29 2019

@author: nate

This script is designed to provide a sandbox to gain insights into what the 
beamforming process is doing with a linear array.  Two different matrices 
can be constructed that give the true angle (W) and the bin that can be mapped to
that angle (F).  This model works for finding AoAs in the range (0,180), a single
linear array is not capable of disambiguating between the signals that may arrive
at theta and -theta

The patterns created for omnidirectional and semi-omni directional patterns look 
kind of rough, so they are synthesized by averaging several patterns of 
patterns that have more directions.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
#plt.close('all')


def rads(degs): return np.pi * degs/180
def degs(rads): return 180 * rads/np.pi

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

def findNls(N):
    B = []
    B.append(N)
    while N/2 > 1:
        B.append(int(N/2))
        N = N/2
    B.reverse()
    return B
   

def create_codebook_chunks(N=128):                         #Number of antenna elements 
    Nls = findNls(N)                   #On the bth level the ith segment covers 1/b of the whole swath. 
    F = buildF(N)                   #Build DFT Matrix
    W = []                          #Codebook initialization
    
    for Nl in Nls:                             #Loop through bins that have b sectors, corresponding to the l = log2(b) layer
        R = int(N/Nl)                    #Number of segments for each direction
        P = int(N/R)                    #Length of each segment       
        Wl = np.zeros([Nl,N]) + 0j       #Initialize codes for level l
        
        for i in range(0,Nl):            #Loop through sectors
            for r in range(0,R):        #Loop through segments
                Wl[i,r*P:(r+1)*P] = F[r+i*R,r*P:(r+1)*P]

       
        W.append(Wl)                        #Append to final codebook.
    W.reverse()
    return W


def create_codebook(M = 128, N = 128, Q = 128):
    '''
    M : is the number of antenna elements
    N : highest number of codewords (bottom layer)
    Q : upsample factor for summing
    '''
    F = buildF(M = M,N = N*Q)               #Build upsampled MxN DFT matrix
    Nls = findNls(N)                           #Build array indicating number of codewords in indexed layer
    W = []                                  #Initialize codebook

    for Nl in Nls:
        Wl = np.zeros([Nl,M]) + 0j           #Initilaize codebook section for this layer
        P = int((N*Q)/Nl)                        #Number of DFT rows to sum and use in each codeword
        for ii in range(Nl):
            norm_fact = np.linalg.norm(np.sum(F[ii*P:(ii+1)*P,:],0))
            Wl[ii,:] = 1/norm_fact * np.sum(F[ii*P:(ii+1)*P,:],0)       
#            Wl[ii,:] = np.sum(F[ii*P:(ii+1)*P,:],0)  
        W.append(Wl)
    return W
    
def create_2Dcodebook(M = 128, N = 128, Q = 128):
    '''
    M : is the number of antenna elements
    N : highest number of codewords (bottom layer)
    Q : upsample factor for summing
    '''
    F = buildF(M = M,N = N*Q)               #Build upsampled MxN DFT matrix
    Nls = findNls(N)                           #Build array indicating number of codewords in indexed layer
    W = []                                  #Initialize codebook

    for Nl in Nls:
        Wl = np.zeros([Nl,M]) + 0j           #Initilaize codebook section for this layer
        P = int((N*Q)/Nl)                        #Number of DFT rows to sum and use in each codeword
        for ii in range(Nl):
            norm_fact = np.linalg.norm(np.sum(F[ii*P:(ii+1)*P,:],0))
            Wl[ii,:] = 1/norm_fact * np.sum(F[ii*P:(ii+1)*P,:],0)       
#            Wl[ii,:] = np.sum(F[ii*P:(ii+1)*P,:],0)  
        W.append(Wl)
        for k in range(int(len(W[-1])**2)):
            pass
            
    return W

def create_theta_specific_codebook(thetas, M = 128):
    N = len(thetas)
    m_mat,theta_mat = np.meshgrid(np.arange(0,M),thetas*np.pi/180)
    F = np.exp(-1j*np.pi*m_mat*np.cos(theta_mat)) #lambda/2 spacing
    Nls = findNls(N)
    W = []
    for Nl in Nls:
        Wl = np.zeros([Nl,M]) + 0j
        P = int(N/Nl)
        for ii in range(Nl):
            norm_fact = np.linalg.norm(np.sum(F[ii*P:(ii+1)*P,:],0))
            Wl[ii,:] = 1/norm_fact * np.sum(F[ii*P:(ii+1)*P,:],0) 
        W.append(Wl)
    return W

#%% Alkhateeb Codebook
def create_alkhateeb_codebook(M,N):
    W = []
    No = dc(N) #original N
    N = 8*N #This needs to be done to make the codebook work because N must be 4x M.
    L = int(np.log2(No))
    for l in range(1,L+1):
        Wl = []
        for k in range(int(2**(l-1))):
            for m in range(2):
                Wl.append(F(l+1,k+1,m+1,M,N,2))
        W.append(np.vstack(Wl))
    return W

def create_alkhateeb_chiu_codebook(M,N,Qint = 1024):
    W = []
    L = np.log2(N).astype('int')
    Q = 120/Qint
    phi = np.arange(0,360,Q)*np.pi/180 #120/1024
#    phi = np.arange(0,360,.9375)*np.pi/180 #120/128
    phi_idx_min = np.argmin(np.abs(rads(30)-phi))
    phi_idx_max = phi_idx_min + Qint
#    phi_idx_min = 32
#    phi_idx_max = 160
    
    Af = Afact(M,len(phi))
    for l in range(0,L):
        Wl = []
        num_part = int(2**(l+1)) #number of partitions
        len_part = int(Qint/num_part) #len of each partition
#        len_part = int(128/num_part)
        u = np.reshape(np.arange(phi_idx_min,phi_idx_max),[num_part,len_part]).astype('int')
        for k in range(num_part): 
            F_p = np.sum(Af[:,u[k]],1)               
            Wl.append(1/np.linalg.norm(F_p) * F_p)
        W.append(np.vstack(Wl))
    return W

#Helper functions
def Afact(M,N):
    A = []
    mode = 'chiu'
    if mode == 'alkhateeb':
        phi = np.arange(0,N)*2*np.pi/N
    if mode == 'chiu':
        phi = np.arange(0,360,0.1171875)*np.pi/180 #120/1024
    
    for l in range(len(phi)):
#        A.append(1/np.sqrt(M) * np.exp(1j*2*np.pi*d*np.arange(0,M)/lam * np.sin(phi[l])))
        A.append(1/np.sqrt(M) * np.exp(-1j*2*np.pi*0.5*np.arange(0,M) * np.cos(phi[l])))
    A = np.transpose(np.vstack(A))
    return np.matmul(np.linalg.inv(np.matmul(A,npH(A))),A)


#def Afact_lim(M,N):
#    '''Since this doesn't account for anything outside of the interval, that part
#    just automatically gets crazy amplified for some reason'''
#    phi_min = 30*np.pi/180
#    phi_max = 150*np.pi/180
#    phi = np.linspace(phi_min,phi_max,N)
#    A = []
#    for l in range(len(phi)):
##        A.append(1/np.sqrt(M) * np.exp(1j*2*np.pi*d*np.arange(0,M)/lam * np.sin(phi[l])))
#        A.append(1/np.sqrt(M) * np.exp(-1j*2*np.pi*0.5*np.arange(0,M) * np.cos(phi[l])))
#    A = np.transpose(np.vstack(A))
#    return np.matmul(np.linalg.inv(np.matmul(A,npH(A))),A)

def F(s,k,m,M,N,K):
    Af = Afact(M,N)
    u = Im(s,k,m,N,K)
    Afu = np.sum(Af[:,u],1)
    norm_fact = 1/np.linalg.norm(Afu)
#    norm_fact = 1
    return  norm_fact * Afu



def npH(X): return np.conj(np.transpose(X))
def I(s,k,N,K): return np.arange((k-1)*N/(K**(s-1)),k*N/(K**(s-1))).astype('int')
def Im(s,k,m,N,K): return np.arange((N/K**(s))*(K*(k-1) + m-1),(N/K**(s))*(K*(k-1) + m)).astype('int') #There's a +1 in the lower term in the paper
def phiu(u,N): return 2*np.pi*u/N  
#%%

if __name__ == '__main__':
    M = 64                                 #Number of antenna elements
    N = 128
    Q = 128

    fc = 60e9
    lam = 3e8/fc
    d = lam/2
#    W = create_2Dcodebook(M = M, N = N, Q = Q)
    W = create_alkhateeb_chiu_codebook(M = M,N = N)
    
    #Specific parameters for antenna pattern or frequency response
    Qz = 512                                        #resolution of frequency response    
    Nz = int(M*Qz)                                  #Upsampled number of rows
    P = int(Nz/N)                                   #number of rows per codeword
    Fz = buildF(M, int(M*Qz))
    
    Atest = Afact_lim(M,N)
    #Translate from kspace to radians/degs
    w = lam*np.linspace(0,N,int(Nz))/(d*N)          #Translate from kspace to radians/degs
    idx = np.where(w>1)[0]                          #Find where w is outside of [0,1]
    w[idx] = w[idx]-2                               #Force w into [-1,1] where arccos is defined
    w = np.arccos(w)                                #arccos operation                                 #
    w[idx] = w[idx]                          #rotate portion to correct angle so the coverage is [-pi/2,pi/2]
    w = np.concatenate([w, -w])              #concatenate to get full 360 pattern  
    sidxs = np.argsort(w)                       #Retrieve sorting indices to make plots look nice
    w = np.sort(w)
    #Plot pattern
#    layer = 1
#    for ii in range(0,len(W[layer])):
    for layer in [2]:
        for ii in range(len(W[layer])):
            Hz = np.matmul(np.conj(Fz),W[layer][ii])
            Hz = 10*np.log10(np.abs(Hz))
            Hz = np.concatenate([Hz,Hz])[sidxs] #Apply sorting to make plot look nice
            plt.figure(0)
            plt.polar(w,Hz,linewidth = 1)
            plt.ylim([-10,0])
    #        plt.savefig('../../figs' + '/M_' + str(M) + '_N_' + str(N) +  '_Q_' + str(Q) + '_lvl_all' +  '.pdf')
#            plt.savefig('../../figs' + 'l_' + str(layer) + '_k_' + str(ii) +    '.png')

#    create_theta_specific_codebook(thetas = np.linspace(30,150,128),M = 64)