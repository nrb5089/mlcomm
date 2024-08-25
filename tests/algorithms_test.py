"""
Adaptive Communication Decision and Information Systems (ACDIS) - User License
https://bloch.ece.gatech.edu/researchgroup/

Copyright (c) 2024-2025 Georgia Institute of Technology 

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software. Users shall cite 
ACDIS publications regarding this work.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
OTHER LIABILITY, WHETHER INANACTION OF CONTRACT TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
sys.path.insert(0,'../mlcomm')
import numpy as np
from copy import deepcopy as dcp
from scipy.special import lambertw as W
from scipy.special import erf
from algorithms import *
from codebooks import *
from channels import *
from util import *


NUM_PATHS = 5
SNR = 20 #in dB

def main():
    hosub_multi_run()
    #hpm_multi_run()
    #abt_run()
    #nphts_multi_run()
    #dbz_calculate_sample_windows_demo()
    #dbz_run_stationary()
    #dbz_run()
    #tasd_beam_subset_demo()
    #nphts_multi_run()
    return 0


def hosub_multi_run():
    for seed in np.arange(100):
        if seed == 0: print(f'Initialized RNG in main loop. Seed = {seed}')
        np.random.seed(seed = seed)
        cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
        aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1]) * 180/np.pi
        
        #Channel Option
        #channel = NYU1({'num_elements' : 64, 'angle_degs' : aoa_aod_degs, 'environment' : 'LOS', 'scenario' : 'Rural', 'center_frequency_ghz' : 28, 'propagation_distance' : 50, 'noise_variance' : 1e-10, 'seed' : 0})
        channel = RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : seed})
        #channel = AWGN({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'snr' : 20, 'seed' : seed})
        
        bandit = HOSUB({'cb_graph' : cb_graph, 'channel' : channel, 'time_horizon' : 150, 'starting_level' : 2, 'c' : 1, 'delta' : .01})
        bandit.run_alg()
        report_bai_result(bandit)
    



def dbz_calculate_sample_windows_demo():
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='./')
    data_dict = calculate_dbz_sample_window_lengths(cb_graph, eps = .01, delta = .1, a = 1, b = .001, c = .001)
    pickle.dump(data_dict, open('../mlcomm/demo_samples.pkl','wb'))
    pickle.dump(data_dict, open('demo_samples.pkl','wb'))

def calculate_dbz_sample_window_lengths(cb_graph, snr_dbs = np.arange(0,50), sigma_us_degs = [.001,.0005,.00025,.0001], tau = 1, eps = .001, delta = .1, a = 1, b = .1, c = .1):
        """
        
        Description
        ------------
        Calculates sample windown lengths for both alignment time and sample complexity required.
        
        
        Parameters
        ----------
        cb_graph : object
            The codebook graph associated with the simulation.
        snr_dbs : list or array of floats
            list of array of signal-to-noise ratio values in dB.
        sigma_us_degs : list or array of floats
            list or array of sigma_u paramter that governs the random acceleration associated with the WGNA channel
        tau : float
            length of time in seconds between channel fluctuations
        delta : float
            confidence term
        epsilon : float
            tolerance term that is scaled based on the level h
        a : float
            Threshold parameter
        b : float
            First confidence parameter
        c : float 
            Secondary confidence parameter
        
        Returns 
        --------
        data_dict : dict
            Dictionary with keys, parallel lists with the value of sigma_u or snr_db
                - 'snr_dbs' 
                - 'sigma_us_degs' 
                - 'samples_alignment' - Sample window lengths for the values of sigma_u prescribed
                - 'samples_complexity' - Sample window lengths for values of snr_db prescribed
                
        """
        
        #Simulation with uniformly distributed phi with no truncation
        def unif_mean_normal_pdf(x,sigma,a,b): 
            ''' In my setting a = -b, where b is the coverage limit'''
            return  (erf((x-a)/np.sqrt(2)/sigma) - erf((x-b)/np.sqrt(2)/sigma))/2/(b-a)

        def unif_mean_normal_cdf(sigma,a): 
            '''Probability that x is less than a and greater than -a
            '''
            c = np.sqrt(2)*sigma
            return  1/4/a * (2*np.sqrt(2)*c/np.sqrt(np.pi) * (np.exp(-4*a**2/c**2)-1) + 4*a*erf(2*a/c))
        
        bws = cb_graph.beamwidths
        nns = np.arange(1,1001)
        sigma_us = [sigma_us_degs[ii]  * np.pi/180 for ii in np.arange(len(sigma_us_degs))]
        probs_coh = []
        for ii,sigma_u in enumerate(sigma_us):
            probs_sigma = []
            for jj,bw in enumerate(bws):
                probs_bw = []
                a = bw/2
                for nn in nns:
                    if nn < 2:
                        probs_bw.append(1)
                    else:
                        sigma_n = np.sqrt(tau**4/4 * sigma_u**2 * (4*nn**3/3 - 4*nn**2 + 11*nn/3 -1) + tau**2*(nn-1)**2*sigma_u**2)
                        probs_bw.append(unif_mean_normal_cdf(sigma_n, a))
                probs_sigma.append(probs_bw)
            probs_coh.append(probs_sigma)
            
        H = cb_graph.H
        samples_alignment = []
        print(f'Confidence Value: {1-delta/(2*H)}')
        for probs_sigma in probs_coh:
            Ncos_sigma = []
            for probs_bw in probs_sigma:
                kidx = 0
                while probs_bw[kidx] > 1 - delta/(2*H):
                    kidx += 1
                kidx -= 1
                Ncos_sigma.append(kidx)
            samples_alignment.append(Ncos_sigma)
            
            
        thetas = np.linspace(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1],1024) 
        samples_complexity = []
        for snr_db in snr_dbs:
            samples_snr = []
            print(f'Starting processing for SNR: {snr_db}')
            for theta in thetas:
                samples_theta = []
                #Populate stats based on channel state
                channel = AWGN({'num_elements' : cb_graph.M, 'angle_degs' : theta * 180/np.pi, 'snr' : snr_db, 'seed' : 0})
                best_midxs = []
                mus = []
                for hh in np.arange(cb_graph.H):
                    mus_h = []
                    for midx in cb_graph.base_midxs[hh]:
                        zeta = np.abs(np.conj(cb_graph.nodes[midx].f)@channel.ht)**2 
                        cb_graph.nodes[midx].mu = zeta + channel.sigma_v**2
                        cb_graph.nodes[midx].var = 2*channel.sigma_v**2 * zeta + channel.sigma_v**4
                        mus_h.append(cb_graph.nodes[midx].mu)
                    mus.append(mus_h)
                    best_midxs.append(cb_graph.level_midxs[hh][np.argmax(mus_h)])
        
            #Get complexity terms just one time since we assume that it's constant over different angles
                g = cb_graph.g
                for hh in np.arange(cb_graph.H):
                    epsh = float(g)**(-(cb_graph.H-hh-1))*eps
                    base_mus = np.array([cb_graph.nodes[midx].mu for midx in cb_graph.base_midxs[hh]])
                    best_mu_midx_h = cb_graph.base_midxs[hh][np.argmax(base_mus)]
                    
                    #If you're at the broadest beam level, use all base midxs that cover the swath
                    if hh == 0: 
                        Deltas = np.max(base_mus) - base_mus
                        size_Ih = float(len(cb_graph.base_midxs[hh]))
                    else:
                        Deltas = np.array([cb_graph.nodes[best_mu_midx_h].mu - cb_graph.nodes[cb_graph.nodes[best_mu_midx_h].prior_sibling].mu, cb_graph.nodes[best_mu_midx_h].mu - cb_graph.nodes[cb_graph.nodes[best_mu_midx_h].post_sibling].mu,0.0])
                        size_Ih = 3
                    Deltas[np.where(Deltas==0)[0][0]] = np.sort(Deltas)[1] #the index corresponding to the max mu is originally zero, make it the smallest non-zero Delta
                    for ii,Delta in enumerate(Deltas):
                        Deltas[ii] = np.max([(Delta + epsh)/4,epsh/2])
                    aleph_h_eps = 0.0
                    for Delta_eps in Deltas:
                        aleph_h_eps +=(2*b*cb_graph.nodes[midx].var + 2*np.sqrt(2*b*c)*Delta_eps + np.sqrt(4*b**2 *cb_graph.nodes[midx].var**2 + 2*np.sqrt(2*c)*cb_graph.nodes[midx].var*Delta_eps*b**(3/2)))/Delta_eps**2
                        
                    x = 1/(4*aleph_h_eps)
                    y = (15*cb_graph.NH/(2*delta))**(1/4)*np.exp((2*size_Ih-1)/(4*aleph_h_eps))  
                    samples_level_h = np.ceil(np.real(-1/x * W(-x/y,k=-1))) + 1
                    samples_theta.append(samples_level_h)
                samples_snr.append(samples_theta)
            samples_complexity.append(np.ceil(np.mean(np.vstack(samples_snr),axis = 0)))
        data_dict = {}
        data_dict['snr_dbs'] = snr_dbs
        data_dict['sigma_us_degs'] = sigma_us
        data_dict['samples_alignment'] = samples_alignment
        data_dict['samples_complexity'] = samples_complexity
        return data_dict
    
def dbz_multi_run_stationary():
    for seed in np.arange(100):
        if seed == 0: print(f'Initialized RNG in main loop. Seed = {seed}')
        np.random.seed(seed = seed)
        cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
        aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1]) * 180/np.pi
        
        #Channel Option
        #channel = NYU1({'num_elements' : 64, 'angle_degs' : aoa_aod_degs, 'environment' : 'LOS', 'scenario' : 'Rural', 'center_frequency_ghz' : 28, 'propagation_distance' : 50, 'noise_variance' : 1e-10, 'seed' : 0})
        channel = RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : seed})
        #channel = AWGN({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'snr' : 20, 'seed' : seed})
        
        bandit = DBZ({'cb_graph' : cb_graph, 'channel' : channel, 'delta' : .1, 'epsilon' : .05, 'sample_window_lengths' : [0 for hh in range(cb_graph.H)], 'mode' : 'stationary', 'levels_to_play' : [0,0,0,1,0,1], 'a' : 1, 'b' : .01, 'c' : .001})
        bandit.run_alg(0)
        report_bai_result(bandit)
    

def dbz_run():
    init_figs()
    
    data_dict = pickle.load(open('demo_samples.pkl','rb'))
    snr_idx = 20
    snr_db = data_dict['snr_dbs'][snr_idx]
    print(f"SNR = {snr_db} chosen based on index {snr_idx}.")
    sample_window_lengths = data_dict['samples_complexity'][snr_idx]
    print(f'Sample window lengths: {sample_window_lengths}')
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='./')
    channel = DynamicMotion({'num_elements' : cb_graph.M, 'sigma_u_degs' : .001, 'initial_angle_degs' : 90,  'fading' : .995, 'time_step': 1, 'num_paths' : NUM_PATHS, 'snr' : snr_db, 'mode' : 'WGNA', 'seed' : 0})
    bandit = DBZ({'cb_graph' : cb_graph, 'channel' : channel, 'delta' : .1, 'epsilon' : .01, 'sample_window_lengths' : sample_window_lengths, 'mode' : 'non-stationary', 'levels_to_play' : [1,1,1,1], 'a' : 1, 'b' : .005, 'c' : .0001})
    
    bandit.run_alg(2000)
    fig,ax = plt.subplots()
    ax.plot(bandit.log_data['relative_spectral_efficiency'])
    ax.set_ylim([0,1])
    ax.set_xlim([0,2000])
    ax.set_xlabel('n')
    ax.set_ylabel('Relative Spectral Efficiency')

def hpm_multi_run():
    for seed in np.arange(100):
        if seed == 0: print(f'Initialized RNG in main loop. Seed = {seed}')
        np.random.seed(seed = seed)
        cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
        aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1]) * 180/np.pi
        
        #Channel Option
        #Option between Rician fading and just AWGN
        channel = RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : seed})
        #channel = AWGN({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'snr' : SNR, 'seed' : seed})
        
        bandit = HPM({'cb_graph' : cb_graph, 'channel' : channel, 'time_horizon' : 100, 'delta' : .01, 'fading_estimation' : 'exact', 'mode' : 'VL'})
        bandit.run_alg()
        report_bai_result(bandit)

def abt_run():
    init_figs()
    cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
    channel = DynamicMotion({'num_elements' : cb_graph.M, 'sigma_u_degs' : .001, 'initial_angle_degs' : 90,  'fading' : .995, 'time_step': 1, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'mode' : 'WGNA', 'seed' : 0})
    bandit = ABT({'cb_graph' : cb_graph, 'channel' : channel, 'delta' : .01, 'fading_estimation' : 'exact'})
    
    bandit.run_alg(2000)
    fig,ax = plt.subplots()
    ax.plot(bandit.log_data['relative_spectral_efficiency'])
    ax.set_ylim([0,1])
    ax.set_xlim([0,2000])
    ax.set_xlabel('n')
    ax.set_ylabel('Relative Spectral Efficiency')
    
    
def tasd_beam_subset_demo():
    cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
    
    #Option between Rician fading and just AWGN
    channel = RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : 90, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : 0})
    
    bandit = TASD({'cb_graph' : cb_graph, 'channel' : channel, 'delta' : .01, 'epsilon' : .001})
    bandit.run_base_alg([90,91,92,93,94,95,96])  #These are the known midxs with the codebook used that surround the best midx (93).
    report_bai_result(bandit)
    
def nphts_multi_run():
    for seed in np.arange(100):
        if seed == 0: print(f'Initialized RNG in main loop. Seed = {seed}')
        np.random.seed(seed = seed)
        cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
        aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1]) * 180/np.pi
        
        #Channel Option
        #channel = NYU1({'num_elements' : 64, 'angle_degs' : 90, 'propagation_condition' : 'LOS', 'setting' : 'Rural', 'center_frequency_ghz' : 28, 'propagation_distance' : 50, 'noise_variance' : 1e-10, 'seed' : 0})
        channel = RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : seed})
        #channel = AWGN({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'snr' : 20, 'seed' : seed})
        
        bandit = NPHTS({'cb_graph' : cb_graph, 'channel' : channel, 'levels_to_play' : [0, 0, 0, 1, 0, 1],'delta' : [0, 0, 0, .05, 0, .05], 'epsilon' : .001})
        bandit.run_alg()
        report_bai_result(bandit)
    
def report_bai_result(bandit):
    """
    Description
    -----------
    Prints several outputs of the resultant simulation.

    Parameters
    ----------
    bandit : object
        Object corresponding to best arm identification algorithm post simulation.

    """
    log_data = bandit.log_data
    print(f'Estimated Best Node midx: {log_data["path"][-1]} after {np.sum(log_data["samples"])} samples')
    print(f'Actual Best Node midx: {bandit.best_midx}')
    try: 
        print(f'Resultant Relative Spectral Efficiency: {bandit.calculate_relative_spectral_efficiency(bandit.cb_graph.nodes[log_data["path"][-1]])}')
    except:
        print(f'Resultant Relative Spectral Efficiency: {log_data["relative_spectral_efficiency"][-1]}')
    print('\n')
    
if __name__ == '__main__': 
    main()