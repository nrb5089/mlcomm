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
    
    
    cb_graph = load_codebook(filename=f'demo_binary_codebook', loadpath='../tests/')
    # cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='./')
    #run_multi_parallel_ia_demo(cb_graph,int(config_idx))
    run_multi_parallel_ia(cb_graph,0)
    
    #run_single_ia_demo(cb_graph)
    
    return

def run_single_ia_demo(cb_graph):
    bandit = run_single_ia(2, {'snr_db' : 30, 'sigma_u' : 0, 'mode' : 'stationary', 'eps' : .001, 'scenario' : 'LOS','cb_graph' : cb_graph})
    samples,rse = bandit.log_data['samples'], bandit.log_data['relative_spectral_efficiency'][-1]
    print(f"Number of samples required: {samples}, resulting RSE: {rse}")
    
def run_single_ia(seed,params_dict):
    sigma_u = params_dict['sigma_u']
    snr_db = params_dict['snr_db']
    cb_graph = dcp(params_dict['cb_graph'])
    scenario =params_dict['scenario']
    fading_estimation = params_dict['fading_estimation']
    num_timesteps = 1000000  #Arbitrary high time horizon, will terminate after finding beam.
    np.random.seed(seed = seed)
    
    aod = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1])
    # vel = (0,0, 0)
    channel = DynamicMotion({'num_elements' : cb_graph.M, 
                             'sigma_u_degs' : 0, 
                             'initial_angle_degs' : aod * 180/np.pi,  
                             'fading' : .995, 
                             'time_step': 1, 
                             'num_paths' : 1, 
                             'snr' : snr_db, 
                             'scenario' : scenario,
                             'mode' : 'WGNA', 
                             'seed' : seed})
    
    bandit =  HPM({'cb_graph' : cb_graph, 
                   'channel' : channel, 
                   'time_horizon' : num_timesteps, 
                   'delta' : .01, 
                   'fading_estimation' : fading_estimation,
                   'mode' : 'VL'})
    t0 = time()
    bandit.run_alg()
    bandit.log_data['runtime'] = time() - t0
    #print(rse)
    # return bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    return bandit

def run_multi_parallel_ia_demo(cb_graph,config_idx = 0):
    samples, rse = run_multi_parallel_ia(cb_graph, config_idx)
    print(f"Number of samples required: {samples}, resulting RSE: {rse}")
    
def run_multi_parallel_ia(cb_graph,config_idx = 0):
    
    num_sims = 2000
    
    params_dicts = []
    
    for scenario in ['LOS']:
        for snr_db in np.arange(20,51,5)[-1::-1]:
            for fading_estimation in ['exact']:
                params_dicts.append({'snr_db' : snr_db, 'fading_estimation' : fading_estimation, 'sigma_u' : 0, 'scenario' : scenario, 'cb_graph' : dcp(cb_graph)})
    
    
    batch_size = mp.cpu_count()-1
    num_batches = int(np.ceil(num_sims/batch_size))
    last_batch_size = num_sims % batch_size
    
    os.makedirs("./data/ia/",exist_ok = True)
    params_dict = params_dicts[config_idx] #if just running one, need to change indent
    for params_dict in params_dicts:
        print(f"Starting simulations for {params_dict['snr_db']} SNR.")
        for batch_idx in range(num_batches):
            
            #Determine seeds for particular batch
            if batch_idx != num_batches-1: batch_seeds = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size) 
            else: batch_seeds = np.arange(batch_idx * batch_size, batch_idx * batch_size +  last_batch_size) 
            
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
            pickle.dump({'out_summed': out_summed, 'out_sq_summed' : out_sq_summed, 'num_sims' : num_sims, 'runtime_summed' : runtime_summed, 'runtime_sq_summed' : runtime_sq_summed, 'last_seed' : batch_seeds[-1]},open(f"./data/ia/snr_{params_dict['snr_db']}_fading_estimation_{params_dict['fading_estimation']}.pkl",'wb'))
    return out_summed[0]/num_sims,out_summed[1]/num_sims
    
if __name__ == '__main__':
    main()
    
    
