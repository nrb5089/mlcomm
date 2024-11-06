import sys
sys.path.insert(0,'../mlcomm')
sys.path.insert(0,'../tests')
import os
from time import time
import pickle
import numpy as np
from copy import deepcopy as dcp
from scipy.special import lambertw as W
from scipy.special import erf
import multiprocessing as mp

from codebooks_test import *

from algorithms import *
from codebooks import *
from channels import *
from util import *


def main():
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='../tests/')
    #run_single_ia_demo()
    
    #run_multi_parallel_ia_demo(cb_graph, int(config_idx))
    run_multi_parallel_ia(cb_graph,0)
    
    return

    
def run_single_ia_demo(cb_graph):
    bandit = run_single_ia(2, {'snr_db' : 20, 'sigma_u' : 0, 'mode' : 'stationary', 'eps' : 0, 'cb_graph' : cb_graph})
    samples,rse = bandit.log_data['samples'], bandit.log_data['relative_spectral_efficiency']
    print(f"Number of samples required: {samples}, resulting RSE: {rse}")
    
def run_single_ia(seed,params_dict):
    sigma_u = params_dict['sigma_u']
    snr_db = params_dict['snr_db']
    mode = params_dict['mode']
    eps = params_dict['eps']
    cb_graph = params_dict['cb_graph']
    
    #print(f'Initializing random number generator seed within run_single: {seed}.')
    np.random.seed(seed = seed)
    
    # codebook_ver = 0
    # #if seed == 0: print(f"Using Codebook v{codebook_ver}.")
    # cb_graph = load_codebook(filename=f'tst_codebook_v{codebook_ver}', loadpath='./')
    aod = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1])
    # vel = (0,0, 0)
    channel = DynamicMotion({'num_elements' : cb_graph.M, 
                             'sigma_u_degs' : sigma_u, 
                             'initial_angle_degs' : aod * 180/np.pi,  
                             'fading' : .995, 
                             'time_step': 1, 
                             'num_paths' : 5, 
                             'snr' : snr_db, 
                             'mode' : 'WGNA', 
                             'seed' : seed})
    
    bandit = NPHTS({'cb_graph' : cb_graph, 
                       'channel' : channel, 
                       'delta' : [.025, .025, .025, .025], 
                       'epsilon' : eps, #Testing different values of epsilon .001, was fine.
                       # 'sample_window_lengths' : [45, 13, 10,  9,],
                       'mode' : mode, 
                       'levels_to_play' : [1,1,1,1]})
    t0 = time()
    bandit.run_alg()
    rse = bandit.log_data['relative_spectral_efficiency']
    bandit.log_data['runtime'] = time() - t0
    # return (np.sum(bandit.log_data['samples']),rse[-1])
    return bandit

def run_multi_parallel_ia_demo(config_idx = 0):
    samples, rse = run_multi_parallel_ia(config_idx)
    print(f"Number of samples required: {samples}, resulting RSE: {rse}")
    
def run_multi_parallel_ia(cb_graph,config_idx = 0):
    
    num_sims = 2000
    
    params_dicts = []
    
    
    for snr_db in np.arange(20,51,5)[-1::-1]:
        for eps in [0,1e-9,1e-7,1e-4]:
            params_dicts.append({'snr_db' : snr_db, 'sigma_u' : 0, 'mode' : 'stationary', 'eps' : eps, 'cb_graph' : cb_graph})
    
    
    batch_size = mp.cpu_count()-1
    num_batches = int(np.ceil(num_sims/batch_size))
    last_batch_size = num_sims % batch_size
    
    #Based on command line argument specified, choose that entry in the params_dict
    # params_dict = params_dicts[config_idx]
    
    os.makedirs("./data/ia/", exist_ok = True)
    for params_dict in params_dicts:
        print(f"Starting simulations for {params_dict['snr_db']} SNR, {params_dict['eps']} eps.")
        for batch_idx in range(num_batches):
            
            #Determine seeds for particular batch
            if batch_idx != num_batches-1: batch_seeds = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size) 
            else: batch_seeds = np.arange(batch_idx * batch_size, batch_idx * batch_size +  last_batch_size) 
            
            
            #Build params tuple
            params_tuples = [(seed,params_dict) for seed in batch_seeds]
            
            #Run Simulations
            with mp.Pool(batch_size) as p:
                # out = p.starmap(run_single_ia, params_tuples)
                bandits = p.starmap(run_single_ia, params_tuples)
            
            #Process outputs, bandits is a list of bandit objects of length num_cpus
            out = [(np.sum(bandit.log_data['samples']),bandit.log_data['relative_spectral_efficiency'][-1]) for bandit in bandits]
            # for bandit in bandits: print(f"Run times: {bandit.log_data['runtime']}")
            out_runtimes = np.array([bandit.log_data['runtime'] for bandit in bandits])
            out_stacked = np.vstack(out)
            if batch_idx == 0: 
                out_summed = np.sum(out_stacked,axis = 0)
                out_sq_summed = np.sum(out_stacked**2,axis = 0)
                runtime_summed = np.sum(out_runtimes)
                runtime_sq_summed = np.sum(out_runtimes**2)  #For calculating empirical variance
            else: 
                out_summed = out_summed + np.sum(out_stacked,axis = 0)
                out_sq_summed = out_sq_summed + np.sum(out_stacked**2,axis = 0)
                runtime_summed = runtime_summed + np.sum(out_runtimes)
                runtime_sq_summed = runtime_sq_summed + np.sum(out_runtimes**2)
            pickle.dump({'out_summed': out_summed, 'out_sq_summed' : out_sq_summed, 'num_sims' : num_sims, 'runtime_summed' : runtime_summed, 'runtime_sq_summed' : runtime_sq_summed, 'last_seed' : batch_seeds[-1]},open(f"./data/ia/snr_{params_dict['snr_db']}_eps_{params_dict['eps']}.pkl",'wb'))
    return out_summed[0]/num_sims,out_summed[1]/num_sims


if __name__ =='__main__':
    main()