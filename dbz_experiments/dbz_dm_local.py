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
    init_figs()
    
    
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='../tests/')
    #run_multi_parallel_dm_demo(cb_graph,int(config_idx))
    run_multi_parallel_dm(cb_graph,0)
    
    #run_single_dm_demo(cb_graph)
    
    return

def run_single_dm_demo(cb_graph):
    bandit = run_single_dm(0, {'snr_db' : 40, 'sigma_u' : .005, 'scenario' : 'LOS', 'cb_graph' : cb_graph})
    rse = bandit.log_data['relative_spectral_efficiency']
    fig,ax = plt.subplots()
    ax.plot(rse)
    ax.set_ylim([0,1])
    plt.show()
    
def run_single_dm(seed,params_dict):
    sigma_u = params_dict['sigma_u']
    snr_db = params_dict['snr_db']
    #cb_graph = dcp(load_codebook(filename=f'tst_codebook_v0', loadpath='../dev/'))
    cb_graph = dcp(params_dict['cb_graph'])
    scenario =params_dict['scenario']
    eps =params_dict['eps']
    num_timesteps = 1000
    np.random.seed(seed = seed)
    
    aod = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1])
    # vel = (0,0, 0)
    channel = DynamicMotion({'num_elements' : cb_graph.M, 
                             'sigma_u_degs' : sigma_u, 
                             'initial_angle_degs' : aod * 180/np.pi,  
                             'fading' : .995, 
                             'time_step': 1, 
                             'num_paths' : 5, 
                             'snr' : snr_db, 
                             'scenario' : scenario,
                             'mode' : 'WGNA', 
                             'seed' : seed})
    
    bandit = DBZ({'cb_graph' : cb_graph, 
                  'channel' : channel, 
                  'delta' : .1, 
                  'epsilon' : eps, 
                  'sample_window_lengths' : np.array([45, 30, 15,  9,]), #Pass as nparray or fails, fix this later. 
                  'mode' : 'non-stationary', 
                  'levels_to_play' : [1,1,1,1], 
                  'a' : 1, 
                  'b' : 1e-2, 
                  'c' : 1e-5})
    t0 = time()
    bandit.run_alg(num_timesteps)
    bandit.log_data['runtime'] = time() - t0
    bandit.log_data['relative_spectral_efficiency'] = bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    #print(rse)
    # return bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    return bandit

def run_multi_parallel_dm_demo(cb_graph,config_idx = 0):
    rse = run_multi_parallel_dm(cb_graph, config_idx)
    fig,ax = plt.subplots()
    ax.plot(rse)
    ax.set_ylim([0,1])
    plt.show()
    
def run_multi_parallel_dm(cb_graph,config_idx = 0):
    
    num_sims = 2000
    
    params_dicts = []
    
    for scenario in ['LOS']:
        for snr_db in np.arange(20,51,5)[-1::-1]:
            for eps in [1e-9,1e-7,1e-4]:
                for sigma_u in [.0005,.001,.005,.01]:
                    params_dicts.append({'snr_db' : snr_db, 'sigma_u' : sigma_u, 'scenario' : scenario, 'eps' : eps, 'cb_graph' : dcp(cb_graph)})
    
    
    batch_size = mp.cpu_count()-1
    num_batches = int(np.ceil(num_sims/batch_size))
    last_batch_size = num_sims % batch_size
    
    os.makedirs("./data/dm/",exist_ok = True)
    # params_dict = params_dicts[config_idx] #if just running one, need to change indent
    for params_dict in params_dicts:
        print(f"Starting simulations for {params_dict['snr_db']} SNR, {params_dict['sigma_u']} sigma_u, {params_dict['eps']} eps.")
        for batch_idx in range(num_batches):
            
            #Determine seeds for particular batch
            if batch_idx != num_batches-1: batch_seeds = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size) 
            else: batch_seeds = np.arange(batch_idx * batch_size, batch_idx * batch_size +  last_batch_size) 
            
            #Each batch uses a random map
            
            #Build params tuple
            params_tuples = [(seed,params_dict) for seed in batch_seeds]
            with mp.Pool(batch_size) as p:
                bandits = p.starmap(run_single_dm, params_tuples)
            
            rse = [bandit.log_data['relative_spectral_efficiency'] for bandit in bandits]
            out_runtimes = np.array([bandit.log_data['runtime'] for bandit in bandits])
            rse_stacked = np.vstack(rse)
            if batch_idx == 0: 
                rse_summed = np.sum(rse_stacked,axis = 0)
                rse_sq_summed = np.sum(rse_stacked**2,axis = 0)
                runtime_summed = np.sum(out_runtimes)
                runtime_sq_summed = np.sum(out_runtimes**2)  #For calculating empirical variance
            else: 
                rse_summed = rse_summed + np.sum(rse_stacked,axis = 0)
                rse_sq_summed = rse_sq_summed + np.sum(rse_stacked**2,axis = 0)
                runtime_summed = runtime_summed + np.sum(out_runtimes)
                runtime_sq_summed = runtime_sq_summed + np.sum(out_runtimes**2)
    
            pickle.dump({'rse_summed': rse_summed, 'rse_sq_summed' : rse_sq_summed,  'num_sims' : num_sims, 'runtime_summed' : runtime_summed, 'runtime_sq_summed' : runtime_sq_summed,'last_seed' : batch_seeds[-1]},open(f"./data/dm/snr_{params_dict['snr_db']}_sigma_u_{params_dict['sigma_u']}_eps_{params_dict['eps']}_scenario_{params_dict['scenario']}.pkl",'wb'))
            # pickle.dump({'out_summed': out_summed, 'num_sims' : num_sims, 'runtime_summed' : runtime_summed, 'runtime_sq_summed' : runtime_sq_summed,'last_seed' : batch_seeds[-1]},open(f"./data/dm_tst_codebook/dm/snr_{params_dict['snr_db']}_sigma_u_{params_dict['sigma_u']}_scenario_{params_dict['scenario']}.pkl",'wb'))
    return rse_summed/num_sims
    
if __name__ == '__main__':
    main()
    
    
