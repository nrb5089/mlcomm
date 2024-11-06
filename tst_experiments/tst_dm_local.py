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
    # init_figs()
    
    #calculate_average_Tchar()
    #test_run_ns_tas_dm()
    
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='../tests/')
    #run_multi_parallel_dm_demo(cb_graph,int(config_idx))
    run_multi_parallel_dm(cb_graph,0)
    
    #run_single_dm_demo()
    
    return

def run_single_dm_demo(cb_graph):
    rse = run_single_dm(1, {'snr_db' : 20, 'sigma_u' : .001, 'eps' : 1e-7, 'cb_graph' : cb_graph})
    fig,ax = plt.subplots()
    ax.plot(rse)
    ax.set_ylim([0,1])
    plt.show()
    
def run_single_dm(seed,params_dict):
    sigma_u = params_dict['sigma_u']
    snr_db = params_dict['snr_db']
    eps = params_dict['eps']
    cb_graph = dcp(params_dict['cb_graph'])
    scenario =params_dict['scenario']
    
    num_timesteps = 1000
    np.random.seed(seed = seed)
    
    codebook_ver = 0
    aod = np.random.uniform(0,2*np.pi)
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
    
    bandit = MotionTS({'cb_graph' : cb_graph, 
                       'channel' : channel, 
                       'delta' : [.025, .025, .025, .025], 
                       'epsilon' : eps, #Testing different values of epsilon .001, was fine.
                       #'sample_window_lengths' : [45, 13, 10,  9,],
                       'sample_window_lengths' : [45, 30, 15,  9,],
                       'mode' : 'non-stationary', 
                       'levels_to_play' : [1,1,1,1]})
    t0 = time()
    bandit.run_alg(num_timesteps)
    bandit.log_data['runtime'] = time() - t0
    bandit.log_data['relative_spectral_efficiency'] = bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    
    #print(rse)
    # return bandit.log_data['relative_spectral_efficiency'][:num_timesteps]
    return bandit

def run_multi_parallel_dm_demo(cb_graph,config_idx = 0):
    bandit = run_multi_parallel_dm(cb_graph, config_idx)
    rse = bandit.log_data['relative_spectral_efficiency']
    fig,ax = plt.subplots()
    ax.plot(rse)
    ax.set_ylim([0,1])
    plt.show()
    
def run_multi_parallel_dm(cb_graph,config_idx = 0):
    
    num_sims = 2000
    
    params_dicts = []
    
    for scenario in ['LOS']:
        for snr_db in np.arange(20,51,5)[-1::-1]:
            for sigma_u in [.0005,.001,.005,.01]:
                for eps in [0,1e-9,1e-7,1e-4]:
                    params_dicts.append({'snr_db' : snr_db, 'sigma_u' : sigma_u, 'eps' : eps, 'scenario' : scenario, 'cb_graph' : dcp(cb_graph)})
    
    
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
    
            pickle.dump({'rse_summed': rse_summed, 'rse_sq_summed' : rse_sq_summed,    'num_sims' : num_sims, 'runtime_summed' : runtime_summed, 'runtime_sq_summed' : runtime_sq_summed, 'last_seed' : batch_seeds[-1]},open(f"./data/dm/snr_{params_dict['snr_db']}_sigma_u_{params_dict['sigma_u']}_eps_{params_dict['eps']}_scenario_{params_dict['scenario']}.pkl",'wb'))
    return rse_summed/num_sims
    
def test_run_ns_tas_dm():
    init_figs()
        
    data_dict = pickle.load(open('../tests/demo_samples.pkl','rb'))
    snr_idx = 20
    snr_db = data_dict['snr_dbs'][snr_idx]
    print(f"SNR = {snr_db} chosen based on index {snr_idx}.")
    # snr_db = 0
    # print(f"SNR = {snr_db} chosen based on index {snr_idx}. OVERRITING WHAT THE ARRAY HAS!")
    
    sample_window_lengths = data_dict['samples_complexity'][snr_idx]
    print(f'Sample window lengths: {sample_window_lengths}')
    # cb_graph = load_codebook(filename='../tests/demo_ternary_codebook', loadpath='./')
    cb_graph = load_codebook(filename=f'tst_codebook_v0', loadpath='./')
    channel = DynamicMotion({'num_elements' : cb_graph.M, 'sigma_u_degs' : .001, 'initial_angle_degs' : 90,  'fading' : .995, 'time_step': 1, 'num_paths' : 30, 'snr' : snr_db, 'mode' : 'WGNA', 'seed' : 0})
    bandit = MotionTS({'cb_graph' : cb_graph, 'channel' : channel, 'delta' : [.025, .025, .025, .025], 'epsilon' : .00001, 'sample_window_lengths' : sample_window_lengths,'mode' : 'non-stationary', 'levels_to_play' : [1,1,1,1]})
    
    bandit.run_alg(1000)
    fig,ax = plt.subplots()
    ax.plot(bandit.log_data['relative_spectral_efficiency'])
    ax.set_ylim([0,1])
    ax.set_xlim([0,1000])
    ax.set_xlabel('n')
    ax.set_ylabel('Relative Spectral Efficiency')

    fig,ax = plt.subplots()
    angles = [angle[0] * 180/np.pi for angle in bandit.channel.log_data['angles']]
    ax.plot(angles)
    ax.set_ylim([30,150])


def calculate_average_Tchar():
    d_co = 10 #Not used, but needs it to be sepcified to load
    codebook_ver = 0
    scenario = 'RMa'
    num_points = 200
    aods = np.linspace(0,np.pi/2,num_points)
    eps = 1e-10
    Tchars = []
    ii = 0
    for aod in aods:
        if np.mod(ii+1,10) == 0 and ii != 0: print(f"{num_points - ii} aods remaining")
        pos = (100 * np.cos(aod), 100 * np.sin(aod), 20) 
        cb_graph = load_codebook(filename=f'tst_codebook_v{codebook_ver}', loadpath='./')
        nodes = cb_graph.nodes
        channel = NYU2({'num_elements' : cb_graph.M, 
                        'scenario' : scenario, 
                        'center_frequency_ghz' : 28, 
                        'initial_me_position' : pos, 
                        'initial_me_velocity' : (0,0,0), 
                        'sigma_u_m_s2' : .001,  #Not used, but needs to be specified
                        'correlation_distance' : d_co, 
                        'noise_variance' : 1e-10,  # According to the numbers in Section V-C of [1] is 1e-10
                        'seed' : 0, 
                        'map_generation_mode' : 'load', 
                        'map_index' : 0, 
                        'maps_dir' : f'./{scenario}_maps_{d_co}'})
        current_midxs = dcp(cb_graph.base_midxs[0])
        Tchars_aod = []
        for hh in range(cb_graph.H):
            
            mus = [np.abs(np.conj(nodes[midx].f)@channel.ht)**2 + channel.sigma_v**2 for midx in current_midxs]
            Tchar = get_Tchar(mus,eps,channel.sigma_v)
            current_midxs = nodes[current_midxs[np.argmax(mus)]].zoom_in_midxs
            Tchars_aod.append(Tchar)
        Tchars.append(Tchars_aod)
        ii += 1
        print(np.sum(np.vstack(Tchars),axis = 0)/ii)
    Tchars = np.sum(np.vstack(Tchars),axis = 0)/num_points
    print(Tchars)
    return

def get_Tchar(mus,eps,sigma_v):
    
    def d(mu1, mu2):
        if mu1 < 0 or mu2 < 0: print('Empirical value mu2-eps < 0, invalid value, need smaller epsilon')
        myd = 1/2 * np.log(mu2/mu1) + mu1/2/mu2 + (mu1 - mu2)**2 / 4 / mu2 / sigma_v**2 - 1/2
        #myd = 
        return myd

    # Define the right lambdaX (minimizer) function depending on epsilon and on the distributions
    def lambdaX(x,mua,mub,epsilon,pre=1e-12):
        # computes the minimizer for lambda in (mu^- ; mu^+ - epsilon) of d(mua,lambda)+d(mub,lambda+epsilon) 
        # has be be used when mua > mub-epsilon !!
    # 	print('in lambdaX')
        if (epsilon==0):
            return (mua + x*mub)/(1+x) 
        elif (x==0):
            return mua
        else:
            #func(lambda)=(lambda-mua)/variance(lambda)+x*(lambda+epsilon-mub)/variance(lambda+epsilon)
            # def func(lam)=(lam-mua)*variance(lam+epsilon)+x*(lam+epsilon-mub)*variance(lam)
            def func(lam): return d(mua,lam)  + x * d(mub,lam + epsilon) #From (6.3)
            return dicoSolve(func, np.max([mub-epsilon,pre]),mua,pre) #Again, assumes that muminus and muplus are -inf and +inf 

    def gb(x,mua,mub,epsilon,pre=1e-12):
    # 	print('in gb')
        # compute the minimum value of d(mua,lambda)+d(mub,lambda+epsilon)
        # requires mua > mub - epsilon
        if (x==0):
            return 0
        else:
            lam = lambdaX(x,mua,mub,epsilon,pre)
            return d(mua,lam)+x*d(mub,lam+epsilon)

    def AdmissibleAux(mu,a,epsilon): 
        #Assumes that muminus is -inf and muplus is inf
    # 	return d(mu[a],np.min(np.array([mu[i] for i in np.arange(len(mu)) if i!=a])-epsilon)), d(mu[a],np.max(np.array([mu[i] for i in np.arange(len(mu)) if i!=a])-epsilon))
        return 0, d(mu[a],np.max(np.array([mu[i] for i in np.arange(len(mu)) if i!=a])-epsilon))
        #return 0, d(mu[a]+ epsilon,np.max(np.array([mu[i] for i in np.arange(len(mu)) if i!=a])))  #This just stalled forever despite you get all positive

    # COMPUTING THE OPTIMAL WEIGHTS BASED ON THE FUNCTIONS G AND LAMBDA

    def xbofy(y,mua,mub,epsilon,pre = 1e-12):
        # return x_b(y), i.e. finds x such that g_b(x)=y
        # requires mua > mub - epsilon
        # requires [0 < y < d(mua,max(mb-epsilon,muminus))]
        def g(x): return gb(x,mua,mub,epsilon) - y
        xMax=1
        while g(xMax)<0:
            xMax=2*xMax
            if xMax>1000000:
                break
        xMin=0
        return dicoSolve(g, xMin, xMax,pre)

    def dicoSolve(f, xMin, xMax, pre=1e-11):
        # find m such that f(m)=0 using binary search
        l = xMin
        u = xMax
        sgn = f(xMin)
        while (u-l>pre):
            m = (u+l)/2
            if (f(m)*sgn>0):
                l = m
            else:
                u = m
        return (u+l)/2

    def auxEps(y,mu,a,epsilon,pre=1e-12):
        # returns F_mu(y) - 1
        # requires a to be epsilon optimal!
        # y has to satisfy 0 < y < d(mua,max(max_{b\neq a} mub - epsilon,mumin))
        # (the function AdmissibleAux computes this support)
        K = len(mu)
        Indices = np.arange(K)
        Indices = np.delete(Indices,a)
        x = [xbofy(y,mu[a],mu[b],epsilon,pre) for b in Indices]
        m = [lambdaX(x[k],mu[a], mu[Indices[k]], epsilon,pre) for k in np.arange(K-1)]
        return (np.sum([d(mu[a],m[k])/(d(mu[Indices[k]], m[k]+epsilon)) for k in np.arange(K-1)])-1)

    def aOpt(mu,a,epsilon, pre = 1e-12):
        # returns the optimal weights and values associated for the epsilon optimal arm a
        # a has to be epsilon-optimal!
        # cannot work in the Bernoulli case if mua=1 and there is another arm with mub=1
        K=len(mu)
        yMin,yMax=AdmissibleAux(mu,a,epsilon)
        def fun(y): return auxEps(y,mu,a,epsilon,pre)
        if yMax==np.inf:
            yMax=1
            while fun(yMax)<0:
                yMax=yMax*2
        ystar = dicoSolve(fun, yMin, yMax, pre)
        x = np.zeros(K)
        for k in np.arange(K):
            if (k==a):
                x[k]=1
            else:
                x[k]=xbofy(ystar,mu[a],mu[k],epsilon,pre)
        nuOpt = x/np.sum(x)
        return nuOpt[a]*ystar, nuOpt

    def OptimalWeightsEpsilon(mu,epsilon,pre=1e-11):
        # returns T*(mu) and a matrix containing as lines the candidate optimal weights
        K=len(mu)
        # find the epsilon optimal arms
        IndEps=np.where(np.array(mu) >= np.max(mu)-epsilon)[0]
        L=len(IndEps)
        if (L>1) and (epsilon==0):
            # multiple optimal arms when epsilon=0
            vOpt=np.zeros(K)
            vOpt[IndEps]=1/L
            return np.inf,vOpt
        else:
            Values = np.zeros(L)
            Weights = []
            for i in np.arange(L):
                # dval,weights=aOpt(mu,IndEps[i][2],epsilon,pre)
                dval,weights=aOpt(mu,IndEps[i],epsilon,pre)
                Values[i]=1/dval
                Weights.append(weights)
            # look at the argmin of the characteristic times
            Tchar = np.min(Values)
            # iFmu=findall(x->x==Tchar, Values)
            iFmu=np.where(Values==Tchar)[0]
            M=len(iFmu)
            WeightsFinal = []
            for i in np.arange(M):
                # WeightsFinal[i,:]=Weights[iFmu[i][2],:]
                WeightsFinal.append(Weights[iFmu[i]])
            return Tchar,WeightsFinal
        
    Tchar, weights = OptimalWeightsEpsilon(mus, eps)
    return Tchar


if __name__ == '__main__':
    main()
    
    
