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
# import multiprocessing as mp

# from codebooks_test import *

from dl_algorithms import *
from codebooks import *
from channels import *
from util import *

def main():
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='../tests/')
    run_multi_dm(cb_graph,resume = True)
    
    # run_single_dm_demo(cb_graph)
    # view_data()
    return

def run_single_dm_demo(cb_graph):
    bandit = run_single_dm(1, {'snr_db' : 50, 'sigma_u' : .0001, 'scenario' : 'LOS', 'outage_recovery' : True, 'cb_graph' : cb_graph})
    rse = bandit.log_data['relative_spectral_efficiency']
    pickle.dump(rse, open('./mydata.pkl','wb'))
    fig,ax = plt.subplots()
    ax.plot(rse)
    ax.set_ylim([0,1])
    plt.show()

def view_data():
    
    data = pickle.load(open('mydata.pkl','rb'))
    print(np.mean(data))
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show()


def run_single_dm(seed,params_dict):
    sigma_u = params_dict['sigma_u']
    snr_db = params_dict['snr_db']
    cb_graph = dcp(params_dict['cb_graph'])
    scenario =params_dict['scenario']
    outage_recovery = params_dict['outage_recovery']
    num_timesteps = 2000
    np.random.seed(seed = seed)
    
    #Avoid previously trained seeds
    seed = seed + 60000
    
    aod = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1])
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
    
    bandit = LSTMTracking({'cb_graph' : cb_graph, 
                  'channel' : channel,'outage_recovery': outage_recovery})
    t0 = time()
    bandit.run_alg(num_timesteps)
    bandit.log_data['runtime'] = time() - t0
    bandit.log_data['relative_spectral_efficiency'] = bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    #print(rse)
    # return bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    return bandit


def run_multi_parallel_dm(cb_graph,resume = False):
    
    num_sims = 2000
    
    params_dicts = []
    
    
    for snr_db in np.arange(15,51):
        for outage_recovery in [True]:
            for scenario in ['LOS','NLOS']:
                for sigma_u in [.0001,.00025,.0005,.001]:
                    params_dicts.append({'snr_db' : snr_db, 'sigma_u' : sigma_u, 'scenario' : scenario, 'outage_recovery': outage_recovery, 'cb_graph' : dcp(cb_graph)})
    
    
    
    os.makedirs("./data/dm/",exist_ok = True)
    
    for params_dict in params_dicts:
        print(f"Starting simulations for {params_dict['snr_db']} SNR, {params_dict['sigma_u']} sigma_u.")
        
        if resume:
            try:
                data_dict = pickle.load(open(f"./data/dm/snr_{params_dict['snr_db']}_sigma_u_{params_dict['sigma_u']}_scenario_{params_dict['scenario']}.pkl",'rb'))
                start_seed = data_dict['last_seed'] + 1
                num_sims = num_sims - start_seed
                rse_summed = data_dict['rse_summed']
                rse_sq_summed = data_dict['rse_sq_summed']
                runtime_summed = data_dict['runtime_summed']
                runtime_sq_summed = data_dict['runtime_sq_summed']
                if start_seed == num_sims: return
            except:
                print("No previous data found, initializing...")
                start_seed = 0
                rse_summed = np.zeros(num_sims)
                rse_sq_summed = np.zeros(num_sims)
                runtime_summed = 0.0
                runtime_sq_summed = 0.0
        else: 
            start_seed = 0
            rse_summed = np.zeros(num_sims)
            rse_sq_summed = np.zeros(num_sims)
            runtime_summed = 0.0
            runtime_sq_summed = 0.0
            
        
        for seed in np.arange(start_seed,num_sims):
            t0 = time()
            #Run single sim
            
            bandit = run_single_dm(seed,params_dict)
            rse = bandit.log_data['relative_spectral_efficiency'] 
            out_runtimes = bandit.log_data['runtime'] 
            rse_summed = rse_summed + np.array(rse)
            rse_sq_summed = rse_sq_summed + np.array(rse)**2
            runtime_summed = runtime_summed + out_runtimes
            runtime_sq_summed = runtime_sq_summed + out_runtimes**2  #For calculating empirical variance
            print(f"Estimated Time Remaining: {convert_seconds((time()-t0) * (num_sims-seed-1))}")
            # if np.mod(seed+1,20) == 0:
            pickle.dump({'rse_summed': rse_summed, 'rse_sq_summed' : rse_sq_summed,  'num_sims' : num_sims, 'runtime_summed' : runtime_summed, 'runtime_sq_summed' : runtime_sq_summed,'last_seed' : seed},open(f"./data/dm/snr_{params_dict['snr_db']}_sigma_u_{params_dict['sigma_u']}_scenario_{params_dict['scenario']}.pkl",'wb'))
    return rse_summed/num_sims
    
if __name__ == '__main__':
    main()
    
    
