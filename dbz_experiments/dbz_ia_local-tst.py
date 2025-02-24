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
    #run_multi_parallel_ia_demo(cb_graph,int(config_idx))
    run_multi_parallel_ia(cb_graph,0)
    # run_multi_parallel_ia(cb_graph,int(config_idx),resume = True)
    
    #run_single_ia_demo(cb_graph)
    
    return

def run_single_ia_demo(cb_graph):
    bandit = run_single_ia(5, {'snr_db' : 50, 'sigma_u' : 0, 'mode' : 'stationary', 'eps' : 1e-9, 'scenario' : 'LOS','cb_graph' : cb_graph})
    samples,rse = bandit.log_data['samples'], bandit.log_data['relative_spectral_efficiency'][-1]
    print(f"Number of samples required: {samples}, resulting RSE: {rse}")
    
def run_single_ia(seed,params_dict):
    sigma_u = params_dict['sigma_u']
    snr_db = params_dict['snr_db']
    cb_graph = dcp(params_dict['cb_graph'])
    scenario =params_dict['scenario']
    eps = params_dict['eps']
    num_timesteps = 1000000  #Arbitrary high time horizon, will terminate after finding beam.
    np.random.seed(seed = seed)
    
    aod = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1])
    channel = DynamicMotion({'num_elements' : cb_graph.M, 
                              'sigma_u_degs' : 0, 
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
                  'delta' : .01, 
                  'epsilon' : eps, 
                  'sample_window_lengths' : [0 for hh in range(cb_graph.H)],
                  'mode' : 'stationary', 
                  'levels_to_play' : [1,1,1,1], 
                  'a' : 1, 
                  'b' : 1e-1, 
                  'c' : 5e-6})
    t0 = time()
    bandit.run_alg(0)
    bandit.log_data['runtime'] = time() - t0
    #print(rse)
    # return bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    return bandit

def run_multi_parallel_ia_demo(cb_graph,config_idx = 0):
    samples, rse = run_multi_parallel_ia(cb_graph, config_idx)
    print(f"Number of samples required: {samples}, resulting RSE: {rse}")
    
def run_multi_parallel_ia(cb_graph,config_idx = 0,resume = False):
    
    num_sims = 2000
    
    params_dicts = []
    
    for scenario in ['LOS']:
        for snr_db in np.arange(20,51,5)[-1::-1]:
        # for snr_db in [35,25]:
            for eps in [1e-9,1e-7,1e-4]:
                params_dicts.append({'snr_db' : snr_db, 'sigma_u' : 0, 'eps' : eps,'scenario' : scenario, 'cb_graph' : dcp(cb_graph)})
    
    
    
    
    os.makedirs("./data/ia/",exist_ok = True)
    batch_size = mp.cpu_count()-1
    # params_dict = params_dicts[config_idx] #if just running one, need to change indent
    for params_dict in params_dicts:
        print(f"Starting simulations for {params_dict['snr_db']} SNR with eps {params_dict['eps']}.")
        if resume:
            data_dict = pickle.load(open(f"./data/ia/snr_{params_dict['snr_db']}_eps_{params_dict['eps']}.pkl",'rb'))
            start_seed = data_dict['last_seed'] + 1
            num_sims = num_sims - start_seed
            out_summed = data_dict['out_summed']
            out_sq_summed = data_dict['out_sq_summed']
            runtime_summed = data_dict['runtime_summed']
            runtime_sq_summed = data_dict['runtime_sq_summed']
        else: start_seed = 0
        
        num_batches = int(np.ceil(num_sims/batch_size))
        last_batch_size = num_sims % batch_size
        for batch_idx in range(num_batches):
            
            #Determine seeds for particular batch
            if batch_idx != num_batches-1: batch_seeds = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size) + start_seed
            else: batch_seeds = np.arange(batch_idx * batch_size, batch_idx * batch_size +  last_batch_size) + start_seed
            
            #Each batch uses a random map
            
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
            if batch_idx == 0 and not resume: 
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
    
if __name__ == '__main__':
    main()
    
    
