    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class Beams:
    '''
    The Beams class creates a spatial channel for a steered beam pattern by a linear array
    which incorporates shadow fading and a variable number of NLOS paths which for now are
    allocated randomly.

    Notes:
        -Units are very important, some are initially set in dBm which needs to be offset
        by 30 prior to being converted to linear units.
        
    
    '''
    def __init__(self, 
                 N0 = -174,
                 W = 2.16e9,
                 f = 60e9,
                 xi = 1.74,
                 sigma = 2,
                 sigrange = [-120,-40],
                 Tssw = 15.8e-6,
                 TBI = 100e-3,
                 M = 128,
                 Pe = 50,
                 L = 5,
                 U = [7,13],
                 d = 20,
                 AoA = 20,
                 seed = 0):
        
        #Simulation Parameters from Table 1 of 2019-Wu
        self.N0 = N0                                        #Noise spectrum Density (dBm/Hz)
        self.W = W                                          #System Bandwidth (Hz)
        self.f = f                                          #Carrier Frequency (Hz)
        self.xi = xi                                        #Path loss exponent (unitless)
        self.sigma = sigma                                  #Shadowing fading std deviation (dB) 
        self.sigrange = sigrange                            #Signal Range (dBm)
        self.Tssw = Tssw                                    #Frame Duration (s)
        self.TBI = TBI                                      #Beacon interval duration (s)
        self.M = M                                          #Number of beams
        self.Pe = Pe                                        #EIRP (dBm)
        self.L = L                                          #Number of NLOS paths
        self.U = U                                          #Extra NLOS path loss (dB)
        self.d = d                                          #Transmission Distance (m)
        self.aoa = AoA                                      #Angle of arrival LOS beam
        self.seed = seed                                    #RNG Seed
        
        np.random.seed(seed)                                #Set RNG Seed
        self.P_dbm = Pe - 10*np.log10(M)                    #Transmit Power dBm
        self.P = 10**((self.P_dbm-30)/10)                   #Transmit Power (W)
        self.sigma_n = np.sqrt(10**((N0-30)/10)/2)          #Noise std deviation for channel
        
        self.F = self.buildF()                              #DFT Matrix

        self.theta0 = np.cos(np.pi*AoA/180)                                             #AoA for LOS Path
        self.thetals = np.cos(np.pi*np.array(np.random.uniform(0,180,self.L))/180)      #AoAs for NLOS Paths
        self.lam = 3e8/self.f                                                           #Transmitter Wavelength (m)
        self.g0_db = 22 - 20*np.log10(self.lam) + 10*self.xi*np.log10(self.d)           #Gain of LOS Path based on FSPL Equation
        
        #Form NLOS channel response based on part of (1) and (2) in 2019-Wu
        self.sum_NLOS = np.zeros(self.M) + 0j
        for thetal in self.thetals:
            self.sum_NLOS += np.exp(-1j * 2 * np.pi * self.lam/2 * np.arange(0,self.M) * thetal / self.lam)             
        
        #Calculate the mean rewards 
        g0_lin = 10**((self.g0_db)/10)                                                                              #Linear channel gain LOS
        gl_lin = 10*g0_lin                                                                                          #Linear channel gain NLOS
        h0 =  1/g0_lin *np.exp(-1j * 2 * np.pi * self.lam/2 * np.arange(0,self.M) * self.theta0 / self.lam) + 1/gl_lin * self.sum_NLOS                                                                                     #Channel reponse LOS and NLOS Eq. (2) from 2019-Wu
        h0_spec = np.sqrt(self.P)*np.matmul(self.F,np.conj(h0))   
        h0_spec_dbm = (10*np.log10(np.abs(h0_spec))+30)                                                             
        self.h_avg = (h0_spec_dbm - self.sigrange[0])/(self.sigrange[1]-self.sigrange[0])                           #Average rewards over all arms                      
        self.b_star,self.r_star =  (np.argmax(self.h_avg),np.max(self.h_avg))                                       #Index and Value of Max Reward
        


    def sample_signal(self,idx=0,mode = 'single'):
        '''
        This function returns the magnitude response of the LOS and NLOS channel components.  This is a time sample
        taken by each array element (array response to beacon signal).  The magnitude response is normalized to [0,1] 
        in logorithmic units.
        
        :param idx : int    : index of single returned value, disregarded if mode == 'all' or mode == 'complex'
        :param mode: string : 'single', 'all' : indicates to return a single indexed value or the entire array 
        '''
        g0_ins = 10**((self.g0_db + self.sigma * np.random.randn(1))/10)                                                    #instantaneous channel gain LOS
        gl_ins = 10**((self.g0_db + np.random.uniform(self.U[0],self.U[1]) + self.sigma * np.random.randn(1))/10)           #instantaneous channel gain NLOS
        
#        hn = 1/(np.sqrt(self.L+1)) * (1/g0_ins *np.exp(-1j * 2 * np.pi * self.lam/2 * np.arange(0,self.M) * self.theta0 / self.lam) + 1/gl_ins * self.sum_NLOS)  #Channel reponse LOS and NLOS Eq. (2) from 2019-Wu
        hn =  (1/g0_ins *np.exp(-1j * 2 * np.pi * self.lam/2 * np.arange(0,self.M) * self.theta0 / self.lam) + 1/gl_ins * self.sum_NLOS)  #Channel reponse LOS and NLOS Eq. (2) from 2019-Wu
                
#        hn_noise = np.sqrt(self.P)*hn + np.matmul(self.F,self.sigma_n *  (np.random.randn(self.M) + 1j*np.random.randn(self.M)))        #Scale by transmit power and add channel nosie  
        hn_noise = np.sqrt(self.P)*hn + self.sigma_n * (np.random.randn(self.M) + 1j*np.random.randn(self.M))        #Scale by transmit power and add channel nosie  
        Hn_noise = np.matmul(np.conj(self.F),hn_noise)
        Hn_noise_dbm = (10*np.log10(np.abs(Hn_noise))+30)                                                                               #Convert magnitude response to dBm
        Hn_noise_dbm_scaled = (Hn_noise_dbm - self.sigrange[0])/(self.sigrange[1]-self.sigrange[0])                                     #Scale to take values between [0,1]       
        if mode == 'all':
            return Hn_noise_dbm_scaled       #Return all rewards 
        if mode == 'complex':
            return hn_noise, Hn_noise_dbm_scaled #Return raw signal and rewards
        else:
            return Hn_noise_dbm_scaled[idx]  
        
    def sample_signal_awgn(self):
        g0_ins = 10**(self.g0_db/10) 
        hn = np.sqrt(self.P) * 1/g0_ins * np.exp(-1j * 2 * np.pi * self.lam/2 * np.arange(0,self.M) * self.theta0 / self.lam)
        hn_noise = hn + self.sigma_n *  (np.random.randn(self.M) + 1j*np.random.randn(self.M))  
        return hn_noise
    
    def buildF(self):
        '''
        Builds a DFT matrix F, ie. F'F =  I and FF' = I, normalized.
        '''
        k = np.arange(0,self.M)
        m = np.arange(0,self.M)
        m,k = np.meshgrid(k,m)
        return 1/np.sqrt(self.M) * np.exp(-1j * 2 * np.pi * m * k / self.M)    
    
def mult_gauss(z,mu,sigma):
    return 1/(2*np.pi * sigma**2) * np.exp(-(np.matmul(z-mu,z-mu))/(2*sigma**2))
    
def gauss(z,mu,sigma):
    return 1/np.sqrt(2*np.pi * sigma**2) * np.exp(-(z-mu)**2/(2*sigma**2))

if __name__=='__main__':
    plt.close('all')
    sim = Beams(M = 128)
    samples = list()
    for ii in range(0,10):
        _,sample = sim.sample_signal(mode = 'complex')
        samples.append(sample)
        plt.figure(0)
        plt.plot(sample)
        
    plt.ylim([0,1])
    plt.xlim([0,sim.M-1])
    plt.grid(axis = 'both')
    
#    samples = np.vstack(samples)
#    samples_r = np.real(samples)
#    samples_i = np.imag(samples)
#    var_r = np.var(samples_r,0)
#    mu_r = np.mean(samples_r,0)
#    var_i = np.var(samples_i,0)
#    mu_i = np.mean(samples_i,0)
#    var = var_r + 1j * var_i
#    mu = mu_r + 1j * mu_i
#    var_a = np.abs(var)
#    
#    idx = 0
#    z = np.array([np.real(sample[idx]),np.imag(sample[idx])])
#    mu_s = np.array([mu_r[idx],mu_i[idx]])
#    
#    mu_t = mu_r[idx]
#    sigma_t = np.sqrt(var_r[idx])
#    Pz = gauss(np.real(sample[idx]),mu_r[idx],np.sqrt(var_r[idx]))
##    plt.figure(1)
##    plt.plot(sample_c)
##    plt.grid(axis = 'both')
#    x = mu_t + sigma_t * np.linspace(-5,5,10000)
#    plt.plot(x,gauss(x,mu_t,sigma_t))