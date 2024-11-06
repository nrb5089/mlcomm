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


import numpy as np
from copy import deepcopy as dcp
import types
import pickle
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.special import erf
from codebooks import *
from channels import *
from util import *

class AlgorithmTemplate:
    """
    Description
    ------------
    AlgorithmTemplate is a class to represent the simulation of an algorithm 
    that interacts with a communication channel and an associated codebook graph.
    
    Attributes
    ----------
    cb_graph : object
        The codebook graph associated with the simulation.
    channel : object
        The communication channel used in the simulation.
    best_midx : int
        midx corresponding to the node with the highest mean_reward
    log_dat : dict
        Algorithm-specific dictionary for storing simulation data.
        
    Methods
    -------
    sample(self, node, with_noise=True):
        Samples the node's response with optional noise.
    set_best(self)
        sets attribute best_midx, the midx with the largest mean reward
    calculate_relative_spectral_efficiency(self,node)
        Calculates the relative spectral efficiency with respect to the node with the highest mean rewards
    """

    def __init__(self, params):
        """
        Description
        ------------
        Initializes the AlgorithmTemplate with the provided parameters.
        
        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph': object
                The codebook graph associated with the simulation.
            - 'channel': object
                The communication channel used in the simulation.
            - 'log_data' : dict
                Used to track performance metrics over time
                - 'relative_spectral_efficiency' 
                    Normalized with respect to best beamforming vector.
                - 'path'
                    History of beamforming vectors chosen during algorithm execution.
                - 'samples' 
                    Number of samples required to terminate for the algorithm.
                - 'flops'
                    Number of floating point operations for the algorithm.
        """
        self.cb_graph = params['cb_graph']
        self.channel = params['channel']
        self.set_best()
        self.log_data = {'relative_spectral_efficiency' : [], 
                         'path' : [],
                         'samples' : [],
                         'flops' : [],
                         'runtime' : []
                         }
        #print(self.best_midx)
        
    def set_best(self):
        """
        Description
        ------------
        Sets the attribute best_midx, which is the midx belonging to the node with the highest mean reward.
        """
        self.best_midx = np.argmax([self.sample(node,with_noise=False) for node in self.cb_graph.nodes.values()])
        self.best_node = self.cb_graph.nodes[self.best_midx]
        
    def sample(self, node, transmit_power_dbm = 1, with_noise=True, mode='rss'): 
        """
        Description
        ------------
        Samples the node's response with optional noise.

        This method computes the absolute squared value of the conjugate
        transpose of the node's field vector multiplied by the channel's array 
        response. Noise can be optionally included in the computation.

        Parameters
        ----------
        node : object
            The node to be sampled.
        transmit_power_dbm : float
            Transmit power over the channel in dbw, not required for BasicChannel
        with_noise : bool, optional
            A flag to indicate whether noise should be included in the sample 
            (default is True).
        mode : str
            Valid choices are 'rss' and 'complex', default to 'rss'.  Dictates reward returned, some Bayesian algorithms require complex value.

        Returns
        -------
        float
            The absolute squared value of the sampled response or complex value within.
        """
        assert mode == 'complex' or mode == 'rss', 'Parameter Selection Error: Valid entries for parameter "mode" are "complex" and "rss" (default)'
        if mode == 'rss':
            return np.abs(np.conj(node.f).T @ self.channel.array_response(transmit_power_dbm = transmit_power_dbm,with_noise=with_noise))**2
        elif mode == 'complex':
            return np.conj(node.f).T @ self.channel.array_response(with_noise=with_noise)
        
    def calculate_relative_spectral_efficiency(self,node):
        """
        Description
        ------------
        Calculates relative spectral efficiency with respect to node specified and node with highest mean reward, attribute best_node

        Parameters
        ----------
        node : object
            The node to be used in the relative spectral efficiency calculation.

        Returns
        -------
        float
            The relative spectral efficiency.
        """
        return np.log2(1 + self.sample(node,with_noise = False)/self.channel.sigma_v**2)/np.log2(1 + self.sample(self.best_node,with_noise = False)/self.channel.sigma_v**2)
    
## Multi-Armed Bandit Algorithms
class HOSUB(AlgorithmTemplate):
    """
    Description
    -----------
    Hierarchical Optimal Sampling for Unimodal Bandits (HOSUB) [1] is a class that extends 
    AlgorithmTemplate to implement a hierarchical optimization algorithm for selecting 
    the best beamforming vector.  HOSUB is find the best arm in a user-designated fixed 
    bugdet of time steps, where one sample is taken each time step.
    
    [1] Blinn, Nathan, Jana Boerger, and Matthieu Bloch. "mmWave Beam Steering with Hierarchical Optimal Sampling for Unimodal Bandits." ICC 2021-IEEE International Conference on Communications. IEEE, 2021.

    Attributes
    ----------
    cb_graph : object
        The codebook graph associated with the simulation.
    N : int
        Time horizon for the algorithm.
    h0 : int
        Starting level in the hierarchical structure.
    c : float
        Exploration-exploitation trade-off parameter.
    delta : float
        Confidence parameter for upper confidence bound calculations.

    Methods
    -------
    __init__(self, params):
        Initializes the HOSUB algorithm with the provided parameters.
    run_alg(self):
        Runs the HOSUB algorithm to find the best beamforming vector.
    update_node(self, node, r):
        Updates the node's empirical mean reward and upper confidence bound.
    """

    def __init__(self, params):
        """
        Initializes the HOSUB algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph': object
                The codebook graph associated with the simulation.
            - 'channel': object
                The communication channel used in the simulation.
            - 'time_horizon': int
                Time horizon for the algorithm.
            - 'starting_level': int
                Starting level in the hierarchical structure.
            - 'c': float
                Exploration-exploitation trade-off parameter.
            - 'delta': float
                Confidence parameter for upper confidence bound calculations.
        """
        super().__init__(params)
        self.N = params['time_horizon']
        self.h0 = params['starting_level']
        self.c = params['c']
        self.delta =params['delta']
        
        #Initialize node attributes and method for bandit algorithm
        for node in self.cb_graph.nodes.values():
            node.mean_reward = self.sample(node,with_noise = False)
            node.empirical_mean_reward = 0.0
            node.num_pulls = 0.0
            node.num_lead_select = 0
            node.ucb = np.inf
        
        self.set_best()
            
    def run_alg(self):
        """
        Description
        -----------
        Runs the HOSUB algorithm to find the best beamforming vector.
        
        This method iteratively selects and samples nodes in the codebook graph,
        updating their empirical mean rewards and upper confidence bounds, until
        the time horizon is reached or the narrowest beamforming vector is found.
        
        Returns
        -------
        int
            The midx corresponding to the estimated best beamforming vector as indexed by the codebook graph.
            
        """
        
        
        
        #Establish aliases
        nodes = self.cb_graph.nodes
        
        #Set initial starting nodes
        current_neighborhood_midxs = self.cb_graph.level_midxs[self.h0]
        
        #Initialize time step counter
        nn = 0
        
        #Sample each starting level beamforming vector once
        for midx in current_neighborhood_midxs:
            self.update_node(nodes[midx],self.sample(nodes[midx]))
            nn += 1
        
        #Fixed Budget, therefore fixed number of samples
        self.log_data['samples'] = self.N
        
        while nn < self.N:
            
            #Find current node with highest empricial mean reward
            current_best_midx = np.argmax([node.empirical_mean_reward for node in nodes.values()])
            self.log_data['path'].append(current_best_midx)
            
            #Return if reached narrowest beamforming vector
            if nodes[current_best_midx].h == self.cb_graph.H-1: return current_best_midx
            
            #Update neighborhood
            if nodes[current_best_midx].h == self.h0:
                current_neighborhood_midxs = [nodes[current_best_midx].prior_sibling,nodes[current_best_midx].post_sibling, nodes[current_best_midx].zoom_in_midxs[0], nodes[current_best_midx].zoom_in_midxs[1]]
            else:
                current_neighborhood_midxs = [nodes[current_best_midx].zoom_out_midx, nodes[current_best_midx].zoom_in_midxs[0], nodes[current_best_midx].zoom_in_midxs[1]]
            
            nodes[current_best_midx].num_lead_select += 1
            
            #Maximum degree of any node is 4, hence we use the integer check from line 12 in Algorithm 1 in [1].  Otherwise, sample the highest UCB in the neighborhood.
            if np.mod((nodes[current_best_midx].num_lead_select-1)/4,1) == 0:
                node_to_sample = nodes[current_best_midx]
            else:
               node_to_sample = nodes[current_neighborhood_midxs[np.argmax([nodes[midx].ucb for midx in current_neighborhood_midxs])]]
              
            #Sample node and update statistics
            self.update_node(node_to_sample,self.sample(node_to_sample))
            self.channel.fluctuation(nn)
            self.set_best()
            
            nn += 1
            
        est_best_midx =  np.argmax([node.empirical_mean_reward for node in nodes.values()])
        self.calculate_relative_spectral_efficiency(nodes[est_best_midx])
        
    def update_node(self,node,r):
        """
        Description
        -----------
        Updates the node's empirical mean reward and upper confidence bound.

        This method updates the number of pulls, empirical mean reward, and upper
        confidence bound for the specified node based on the new sample reward.

        Parameters
        ----------
        node : object
            The node to be updated.
        r : float
            The new sample reward.

        Returns
        -------
        None
        """
        node.num_pulls += 1
        node.empirical_mean_reward = ((node.num_pulls-1) * node.empirical_mean_reward + r)/node.num_pulls
        node.ucb = node.empirical_mean_reward + np.sqrt(self.c * np.log(2/self.delta)/2/node.num_pulls)

class DBZ(AlgorithmTemplate):
    """
    Dynamic Beam Zooming (DBZ) takes a hierarchical approach to intial alignment and beam tracking for an integrated sensing and communication approach.  
    DBZ uses a base algorithm, Lower-Upper Confidence Bound (LUCB) [1,2] for best arm identification in fixed confidence.  May be run in the traditional sense, where all
    sample over time are considered for the mean, or a sample window length is specified where only recent samples are considered. The 
    latter is used for non-stationary reward structures.  Different from [1,2], we are only interested in sets of arms with cardinality 1.
    
    Our confidence term and exploration rate differ due to the reward probability distribution.
    
    [1] Kalyanakrishnan, Shivaram, et al. "PAC subset selection in stochastic multi-armed bandits." ICML. Vol. 12. 2012.
    [2] Gabillon, Victor, Mohammad Ghavamzadeh, and Alessandro Lazaric. "Best arm identification: A unified approach to fixed budget and fixed confidence." Advances in Neural Information Processing Systems 25 (2012).
    
    Parameters
    ----------
    params : dict
        dict with params
        
    Attributes
    ----------
    cb_graph : object
        The codebook graph associated with the simulation.
    delta : float
        confidence term
    epsilon : float
        tolerance term that is scaled based on the level h
    mode : str
        Specifies 'stationary' or 'non-stationary' setting, either or is a valid entry
    p : nparray 
        array of 1s and 0s indicating which levels to play
    W : nparray of ints
        array where each element indicates the sample window length associated with the particular level
    a : float
        Threshold parameter
    b : float
        First confidence parameter
    c : float 
        Secondary confidence parameter
        
    Notes
    -----
    
    Example
    -------
    bandit = DBZ({'cb_graph' : cb_graph, 'channel' : channel, 'delta' : .1, 'epsilon', .001, 'mode' : stationary, levels_to_play' : [1,1,1,1], 'a' : 1, 'b' : .1, 'c' : .1})
    """
    
    def __init__(self,params):
        """
        """
        super().__init__(params)
        self.delta = params['delta']
        self.epsilon = params['epsilon']
        self.p = params['levels_to_play']
        self.mode = params['mode']
        assert self.mode == 'stationary' or self.mode == 'non-stationary', 'Parameter Selection Error: Valid entries for parameter "mode" are "stationary" (default) and "non-stationary" '
        if self.mode == 'stationary':
            self.W = np.inf * np.ones(self.cb_graph.H)
        elif self.mode == 'non-stationary':
            self.W = params['sample_window_lengths']
        assert np.all(self.W > 3), 'Parameter Selection Error: "sample_window_length" should be chosen to be larger than 3.'
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        if 'transmit_power_dbm' in params: self.transmit_power_dbm = params['transmit_power_dbm']
        else: self.transmit_power_dbm = 0
        
        assert self.p[self.cb_graph.H-1] == 1, "Error: Last value of 'levels_to_play' (self.p) must be 1."
        
        #Terms for the statistics are time-varying, hence must be calculated live as node methods
        # def exploration(self,nn): 
        #     nn = np.min([float(nn),self.W])
            
        #     return np.log(15.0*self.NH * (np.min([nn,self.W]))**4.0/4.0/self.delta)
            
        # def confidence_term(self,nn): 
        #     num_pulls = len(np.where(np.array(self.sample_history) != 0)[0])
        #     if num_pulls < 2: 
        #         return np.inf
        #     else:
        #         empirical_variance = np.sum(np.array(self.sample_history)**2)/num_pulls - np.sum(self.sample_history)**2/num_pulls**2
        #         return np.sqrt(4 * self.b * empirical_variance * self.exploration(nn) / num_pulls) + 2 * np.sqrt(2 * self.b * self.c) * self.exploration(nn) / (num_pulls-1) 
        
        # def ucb(self,nn): 
        #     num_pulls = len(np.where(np.array(self.sample_history) != 0)[0])
        #     if num_pulls < 2: 
        #         return np.inf
        #     else:
        #         return 1/num_pulls * np.sum(self.sample_history) + self.confidence_term(nn)
        
        # def lcb(self,nn): 
        #     num_pulls = len(np.where(np.array(self.sample_history) != 0)[0])
        #     if num_pulls < 2: 
        #         return -np.inf
        #     else:
        #         return 1/num_pulls * np.sum(self.sample_history)  - self.confidence_term(nn)
        
        #Initialize node attributes and method for bandit algorithm
        for node in self.cb_graph.nodes.values():
            node.sample_history = []
            node.delta = self.delta
            node.b = self.b
            node.c = self.c
            node.W = self.W[node.h]
            node.NH = float(len(self.cb_graph.level_midxs[0]) + 3 * (self.cb_graph.H-1))  #Total number of beams, note that this is hard-coded for our purposes
            # node.exploration = types.MethodType(exploration,node)
            # node.confidence_term = types.MethodType(confidence_term,node)
            # node.ucb = types.MethodType(ucb,node)
            # node.lcb = types.MethodType(lcb,node)
    
    def run_alg(self,time_horizon):
        if self.mode == 'stationary': time_horizon = 10000000 #Hard stop algorithm time ceiling
        nodes = self.cb_graph.nodes
        current_midxs = dcp(self.cb_graph.base_midxs[0])
        while self.p[nodes[current_midxs[0]].h] != 1:
            current_midxs = np.hstack([nodes[midx].zoom_in_midxs for midx in current_midxs])
        H = self.cb_graph.H
        thresholds = [-np.inf]
        self.comm_node = None
        gamma_midx,u_midx,nn = self.initialize(0.0,current_midxs)
        
        Zmax = False
        while nn < time_horizon:
            nn_flops = 0.0
            if not Zmax: G = nodes[u_midx].ucb(nn-1) - nodes[gamma_midx].lcb(nn-1)
            else: G = np.inf
            nn_flops += 2*(2*nodes[gamma_midx].W + 13) + 1
            epsh = self.epsilon * self.cb_graph.g**(-(H-nodes[current_midxs[0]].h-1))
            #print(f'{nn} - G: {G}, epsh : {epsh}')
            
            #Case for Zooming In
            if G < epsh and not Zmax:
                self.comm_node = nodes[gamma_midx]
                
                thresholds.append(nodes[gamma_midx].lcb(nn-1) + self.a * epsh)
                current_midxs = nodes[gamma_midx].zoom_in_midxs
                while self.p[nodes[current_midxs[0]].h] != 1:
                    current_midxs = np.hstack([nodes[midx].zoom_in_midxs for midx in current_midxs])
                if self.comm_node.h == H-1:
                    Zmax = True
                    gamma_midx, u_midx = dcp(self.comm_node.midx), None
                    if self.mode == 'stationary': 
                        self.log_data['path'].append(gamma_midx)
                        self.log_data['samples'] = [nn]
                        self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
                        return gamma_midx,nn
                else:
                    gamma_midx,u_midx = self.get_gamma_u(nn, current_midxs)
                    gamma_midx,u_midx,nn = self.initialize(nn,current_midxs)
                    
            # Case for Zooming Out
            elif np.any(nodes[gamma_midx].ucb(nn-1) < np.array(thresholds)):
                Zmax = False
                thresholds.pop(-1)
                if self.comm_node.h == 0:
                    self.comm_node = None
                    current_midxs = self.cb_graph.base_midxs[0]
                    gamma_midx,u_midx,nn = self.initialize(nn,current_midxs)
                else:
                    self.comm_node = nodes[self.comm_node.zoom_out_midx]
                    current_midxs = self.comm_node.zoom_in_midxs
                gamma_midx,u_midx = self.get_gamma_u(nn,current_midxs)
            
            #Otherwise
            else:
                if not Zmax: gamma_midx,u_midx = self.get_gamma_u(nn,current_midxs)
                else: gamma_midx,u_midx = dcp(self.comm_node.midx),None
                
            #Perform sampling iteration and increment time
            self.sampling_iteration(nn,gamma_midx,u_midx)
            nn += 1
            
        
            
            
    def sampling_iteration(self,nn,gamma_midx,u_midx):
        """
        Description
        -----------
        Subordinate algorithm for sampling in LUCB algorithms.  If there is only one arm being considered for sampling, which will happen when the active beam is one of the narrowest beamforming patterns, sample it.
        Otherwise, proceed through the selection process outlined in [1,2]
        
        Parameters
        ----------
        nn : int
            time step of the algorithm
        gamma_midx : int 
            corresponds to the largest LCB 
        u_midx : int
            corresponds to the largest UCB that is not gamma_midx, determined with method 'get_gamma_u'
            
        Notes
        -----
        Requires 2*(2*nodes[u_midx].W + 11) + 1 to compute the confidence term and 2*self.cb_graph.M +1 from ``perform_sample_update_channel``
        """
        nodes = self.cb_graph.nodes
        if u_midx == None:
            node_to_sample = nodes[gamma_midx]
        else:
            # gamma_midx, u_midx = self.get_gamma_u(nn,current_midxs)
            if nodes[gamma_midx].confidence_term(nn) > nodes[u_midx].confidence_term(nn):
                node_to_sample = nodes[gamma_midx]
            elif nodes[gamma_midx].confidence_term(nn) < nodes[u_midx].confidence_term(nn):
                node_to_sample = nodes[u_midx]
            else:
                node_to_sample = nodes[np.random.choice([gamma_midx,u_midx])]
        self.perform_sample_update_channel(nn,node_to_sample)
        
    def initialize(self,nn,current_midxs):
        """
        Description
        -----------
        Samples all nodes within current_midxs twice, and determines initial gamma and u from the LUCB algorithm.  gamma and u correspond to the largest LCB and the largest UCB that is not 
        the same midx as gamma.
        
        Parameters
        ----------
        current_midxs : list or array 
            midxs corresponding to cb_graph object attribute in which the bandit game will play those nodes
        
        """
        nodes = self.cb_graph.nodes
        if not np.all(self.W == np.inf):
            if np.any(self.W < 6):
                print("Warning: sample window (attribute W) considers too few samples to encompass initial round robin sampling, consider making it larger")
        
        #Sample each arm twice
        for _ in [0,1]:
            for midx in current_midxs:
                node_to_sample = nodes[midx]
                self.perform_sample_update_channel(nn,node_to_sample)
                nn += 1
                
        gamma_midx, u_midx = self.get_gamma_u(nn,current_midxs)
        
        return gamma_midx, u_midx, nn
            
    def get_gamma_u(self,nn,current_midxs):
        """
        Description
        -----------
        Determines the midx corresponding to gamma and u from the LUCB algorithm.  gamma and u correspond to the largest LCB and the largest UCB that is not 
        the same midx as gamma.
        
        Parameters
        ----------
        nn : int
            time step of the algorithm
        current_midxs : list or array 
            midxs corresponding to cb_graph object attribute in which the bandit game will play those nodes
        
        Notes
        -----
        
        """
        nodes = self.cb_graph.nodes
        #LUCB arm sample selection process
        gamma_midx = current_midxs[np.argmax([nodes[midx].lcb(nn) for midx in current_midxs])]
        arg_sort_ucbs = np.argsort([nodes[midx].ucb(nn) for midx in current_midxs])[-1::-1]
        if current_midxs[arg_sort_ucbs[0]] == gamma_midx:
            u_midx = current_midxs[arg_sort_ucbs[1]]
        else:
            u_midx = current_midxs[arg_sort_ucbs[0]]
        return gamma_midx, u_midx
    
    def update_node(self,node,r):
        """
        Description
        -----------
        Updates the node's sample history, and deletes "old" samples as specified by the sampling window length "W"
        
        In the non-stationary setting, this "ages out" older samples of other nodes
        
        Parameters
        ----------
        node : object
            The node to be updated.
        r : float
            The new sample reward.
            
        Returns
        -------
        None
        
        Notes
        -----
        Simulated buffered memory, not consuming flops
        """
        
        #Update Node that was just sampled
        node.sample_history.append(r)
        if len(node.sample_history) > node.W:
            node.sample_history.pop(0)
        
        #Age-out older samples on other nodes
        if self.mode == 'non-stationary':
            for other_node in self.cb_graph.nodes.values():
                if other_node.midx != node.midx:
                    other_node.sample_history.append(0.0)
                    if len(other_node.sample_history) > node.W:
                        other_node.sample_history.pop(0)
                        

    def perform_sample_update_channel(self,nn,node_to_sample):
        """
        Description
        -----------
        Wrapper function to perform operations necessary during sampling.
        
        Notes
        -----
        Requires 2* self.cb_graph.M +1 flops for the inner product and absolute value squared.
        """
        self.update_node(node_to_sample,self.sample(node_to_sample,transmit_power_dbm=self.transmit_power_dbm))
        
        if self.comm_node == None:
            self.log_data['relative_spectral_efficiency'].append(0.0)
            self.log_data['path'].append(np.nan)
        else:
            # self.set_best()
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
            self.log_data['path'].append(self.comm_node.midx)
        self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
        self.set_best()


## Bayesian Algorithms
class HPM(AlgorithmTemplate):
    """
    Description
    -----------
    Implements the Hierarchical Posterior Matching (HPM) [1] algorithm.  Given our fixed codebook resolution, we only use the
    fixed resolution (FR) impelmentation in their algorithm.

    [1] Chiu, Sung-En, Nancy Ronquillo, and Tara Javidi. "Active learning and CSI acquisition for mmWave initial alignment." 
        IEEE Journal on Selected Areas in Communications 37.11 (2019): 2474-2489.
        
        
    Attributes
    ----------
    
    delta : float
        The target confidence for the algorithm.
    fading_estimation : str
        Fading estimation mode, valid choices are 'exact', 'estimate', and 'unitary'
    mode : str
        Fixed length (FL) or variable length (VL) where the algorithm stops after a certain number of timesteps or after achieving confidence 1-delta
        
    Notes
    -----
    Works only with binary tree codebook.
    """

    def __init__(self, params):
        """
        Initializes the HPM algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the initialization parameters.
                - 'time_horizon'
                    Maximum number of samples before terminating algorithm.  Only used in 'FL' mode
                - 'delta'
                    The target confidence for the algorithm.  Only used in 'VL' mode
                - 'fading_estimation'
                    Fading estimation mode, valid choices are 'exact', 'estimate', and 'unitary'
                - 'mode'
                    Fixed length (FL) or variable length (VL) where the algorithm stops after a certain number of timesteps or after achieving confidence 1-delta
        """
        super().__init__(params)
        self.time_horizon = params['time_horizon']
        self.delta = params['delta']
        self.fading_estimation = params['fading_estimation']
        self.mode = params['mode']
        # initialise pi(t)_i's - distribution P(phi = theta_i | z_1:t, w_1:t) i = 1,2,.., 1/resolution
        
        
    def run_alg(self):
        """
        Executes the HPM algorithm for the simulation and logs various data points.

        This method runs the core HPM algorithm, capturing key data points at various stages of the process. 
        The logged data is stored in a dictionary and returned at the end of the method execution.

        Returns
        -------
        dict
            A dictionary containing logged data from the simulation run.
            The keys in the dictionary represent different data points, and the values represent the recorded data for those points.

        Raises
        ------
        SimulationError
            If the simulation encounters an error during execution.

        """
        log_data = {
            'relative_spectral_efficiency': [],
            'flops': []
        }
        
        def pidkl(node):
            """
            Returns posterior probability aggregate of the specified node.

            This function calculates the sum of the posterior probabilities 
            corresponding to the nodes with the narrowest beamforming patterns 
            at a given hierarchy level in the codebook graph.

            Parameters
            ----------
            node (Node)
                A node in the codebook graph.

            Returns
            -------
            float
                The aggregated posterior probability for the specified node.
            """
            #aggregated_indices = np.arange(ii * 2**(self.H-hh-1), (ii+1)*2**(self.H-hh-1))
            return np.sum(posteriors[np.arange(node.i * 2**(H-node.h-1), (node.i+1)*2**(H-node.h-1))])
        
        def q_func(y):
            """
            Generic quantization function.

            This function performs quantization on the input value. 
            The implementation can be customized based on specific quantization needs.

            Parameters
            ----------
            y (float)
                The input value to be quantized.

            Returns
            -------
            float
                The quantized value.
            """
            return y
        
        nodes = self.cb_graph.nodes #Shorthand for nodes
        H = self.cb_graph.H # Number of levels of the beamforming codebook
        N = len(self.cb_graph.level_midxs[-1]) # Number of steered angle positions for the beamforming codebook
        M = self.cb_graph.M
        posteriors = np.ones(N)/N  #Establish priors
        VL = True
        nn = 1
        current_node = nodes[0]
        while True:
            nn_flops = 0.0
            
            #Make initial zoom-in node selection at h = 0
            z0,z1 = 0,1
            l_star = 0
            if pidkl(nodes[z0]) > pidkl(nodes[z1]):
                current_node = nodes[z0]
            elif pidkl(nodes[z0]) < pidkl(nodes[z1]):
                current_node = nodes[z1]
            else:
                current_node = nodes[np.random.choice([z0,z1])]
                
            nn_flops += 2 * 2**(H -nodes[z0].h-1) -1
            for hh in np.arange(1,H):
                z0,z1 = current_node.zoom_in_midxs
                if pidkl(current_node) > 0.5 or hh == 0:
                    l_star = dcp(hh)
                    if pidkl(nodes[z0]) > pidkl(nodes[z1]):
                        current_node = nodes[z0]
                    elif pidkl(nodes[z0]) < pidkl(nodes[z1]):
                        current_node = nodes[z1]
                    else:
                        current_node = nodes[np.random.choice([z0,z1])]
                    nn_flops += 2 * 2**(H -nodes[z0].h-1) -1
                else:
                    #options in (15) in Algorithm 1 of Chiu
                    sel_0 = current_node.zoom_out_midx
                    val_0 = np.abs(pidkl(nodes[sel_0]) - 0.5)
                    nn_flops += 2**(H-nodes[sel_0].h-1) - 1
                    sel_1 = self.cb_graph.level_midxs[l_star + 1][current_node.i]
                    val_1 = np.abs(pidkl(nodes[sel_1]) - 0.5)
                    nn_flops += 2**(H-nodes[sel_1].h-1) - 1 
                    if val_0 < val_1:
                        current_node = nodes[sel_0]
                    elif val_0 > val_1:
                        current_node = nodes[sel_1]
                    else:
                        current_node = nodes[np.random.choice([sel_0,sel_1])]
                    break
             
            z = q_func(self.sample(current_node,mode = 'complex'))
            nn_flops += 2*self.cb_graph.M -1
            # if nn > 2000:
            #     print('pause')
            if self.fading_estimation == 'exact':
                alpha_hats = self.channel.alphas
            elif self.fading_estimation == 'estimate':
                alpha_hats = self.channel.alphas + .25/2 * randcn(self.channel.L)
            elif self.fading_estimation == 'unitary':
                alpha_hats = np.ones(self.channel.L)
            
            athetai = np.array([avec(angle,self.cb_graph.M) @ np.conj(current_node.f) for angle in np.array([self.cb_graph.nodes[midx].steered_angle for midx in self.cb_graph.level_midxs[-1]])])
            nn_flops += 2 * self.cb_graph.M * N
            
            mus = np.sum([alpha_hats[ll] *  athetai for ll in np.arange(self.channel.L)],axis = 0)
            nn_flops += self.channel.L
            
            if self.channel.sigma_v <= .05: sigma_v_temp = .05
            else: sigma_v_temp = dcp(self.channel.sigma_v)
            pdfs =  1/(np.pi * sigma_v_temp**2) * np.exp(-np.abs(z-mus)**2/(sigma_v_temp**2))
            
            nn_flops += 5*N

            sum_vec = np.sum(posteriors*pdfs) + 1e-30
            # sum_vec = np.sum(posteriors*pdfs)
            posteriors = posteriors*pdfs / (sum_vec) + 1e-30
            nn_flops += 4*N-1

            posteriors = 1/np.sum(posteriors) * posteriors #only for normalizing, not explicitly written in algorithm

        
            #Estimated best at this iteration
            est_best_midx = self.cb_graph.level_midxs[-1][np.argmax(posteriors)]
            
            self.log_data['path'].append(est_best_midx)
            self.log_data['flops'].append(nn_flops)
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(nodes[est_best_midx]))
            self.log_data['samples'] = [nn]
            
            if (self.mode == 'VL' and np.max(posteriors) > 1- self.delta):  return
            elif (self.mode == 'FL' and nn == self.time_horizon): return 
            else:
                nn +=1
                self.channel.fluctuation(nn,angle_limits = self.cb_graph.min_max_angles)
                self.set_best()
            
            
        
            

class ABT(AlgorithmTemplate):
    """
    Adaptive Beam Tracking (ABT) from [1], uses initial alignment from HPM algorithm in [2]
    
    [1] Ronquillo, Nancy, and Tara Javidi. "Active beam tracking under stochastic mobility." ICC 2021-IEEE International Conference on Communications. IEEE, 2021.
    [2] Chiu, Sung-En, Nancy Ronquillo, and Tara Javidi. "Active learning and CSI acquisition for mmWave initial alignment." IEEE Journal on Selected Areas in Communications 37.11 (2019): 2474-2489.
    
    Notes
    -----
    
    FixedV does not work.
    
    """
    def __init__(self,params):
        """
        """
        super().__init__(params)
        self.delta = params['delta']
        self.fading_estimation = params['fading_estimation']
        
        
    def run_alg(self, time_horizon):
        """
        Executes the HPM algorithm for the simulation and logs various data points.

        This method runs the core HPM algorithm, capturing key data points at various stages of the process. 
        The logged data is stored in a dictionary and returned at the end of the method execution.
        
        Parameters
        ----------
        time_horizon : int
            Number of timesteps to track the moving entity
            
        Returns
        -------
        dict
            A dictionary containing logged data from the simulation run.
            The keys in the dictionary represent different data points, and the values represent the recorded data for those points.

        Raises
        ------
        SimulationError
            If the simulation encounters an error during execution.

        Example
        -------
            log_data = run_alg()
            print(log_data["relative_spectral_efficiency"])
        """
        log_data = {
            'relative_spectral_efficiency': [],
            'flops': []
        }
        
        def pidkl(node):
            """
            Returns posterior probability aggregate of the specified node.

            This function calculates the sum of the posterior probabilities 
            corresponding to the nodes with the narrowest beamforming patterns 
            at a given hierarchy level in the codebook graph.

            Parameters
            ----------
            node (Node)
                A node in the codebook graph.

            Returns
            -------
            float
                The aggregated posterior probability for the specified node.
            """
            #aggregated_indices = np.arange(ii * 2**(self.H-hh-1), (ii+1)*2**(self.H-hh-1))
            return np.sum(posteriors[np.arange(node.i * 2**(H-node.h-1), (node.i+1)*2**(H-node.h-1))])
        
        def q_func(y):
            """
            Generic quantization function.

            This function performs quantization on the input value. 
            The implementation can be customized based on specific quantization needs.

            Parameters
            ----------
            y (float)
                The input value to be quantized.

            Returns
            -------
            float
                The quantized value.
            """
            return y
        
        def unif_mean_normal_pdf(x,sigma,a,b): 
        	''' In my setting, a and b are the coverage limits of the beam'''
            #7 flops to compute per each element of x, assume fast integral LUT
        	return  (erf((x-a)/np.sqrt(2)/sigma) - erf((x-b)/np.sqrt(2)/sigma))/2/(b-a)
        
        nodes = self.cb_graph.nodes #Shorthand for nodes
        H = self.cb_graph.H # Number of levels of the beamforming codebook
        N = len(self.cb_graph.level_midxs[-1]) # Number of steered angle positions for the beamforming codebook
        M = self.cb_graph.M
        posteriors = np.ones(N)/N  #Establish priors
        pim1 = np.ones(N)/N
        VL = True
        nn = 0
        current_node = nodes[0]
        for nn in np.arange(time_horizon):
            nn_flops = 0.0
            
            #Make initial zoom-in node selection at h = 0
            z0,z1 = 0,1
            l_star = 0
            if pidkl(nodes[z0]) > pidkl(nodes[z1]):
                current_node = nodes[z0]
            elif pidkl(nodes[z0]) < pidkl(nodes[z1]):
                current_node = nodes[z1]
            else:
                current_node = nodes[np.random.choice([z0,z1])]
                
            nn_flops += 2 * 2**(H -nodes[z0].h-1) -1
            for hh in np.arange(1,H):
                z0,z1 = current_node.zoom_in_midxs
                if pidkl(current_node) > 0.5 or hh == 0:
                    l_star = dcp(hh)
                    if pidkl(nodes[z0]) > pidkl(nodes[z1]):
                        current_node = nodes[z0]
                    elif pidkl(nodes[z0]) < pidkl(nodes[z1]):
                        current_node = nodes[z1]
                    else:
                        current_node = nodes[np.random.choice([z0,z1])]
                    nn_flops += 2 * 2**(H -nodes[z0].h-1) -1
                else:
                    #options in (15) in Algorithm 1 of Chiu
                    sel_0 = current_node.zoom_out_midx
                    val_0 = np.abs(pidkl(nodes[sel_0]) - 0.5)
                    nn_flops += 2**(H-nodes[sel_0].h-1) - 1
                    sel_1 = self.cb_graph.level_midxs[l_star + 1][current_node.i]
                    val_1 = np.abs(pidkl(nodes[sel_1]) - 0.5)
                    nn_flops += 2**(H-nodes[sel_1].h-1) - 1 
                    if val_0 < val_1:
                        current_node = nodes[sel_0]
                    elif val_0 > val_1:
                        current_node = nodes[sel_1]
                    else:
                        current_node = nodes[np.random.choice([sel_0,sel_1])]
                    break
             
            z = q_func(self.sample(current_node,mode = 'complex'))
            nn_flops += 2*self.cb_graph.M -1
            
            if self.fading_estimation == 'exact':
                alpha_hats = self.channel.alphas
            elif self.fading_estimation == 'estimate':
                alpha_hats = self.channel.alphas + .25/2 * randcn(self.channel.L)
            elif self.fading_estimation == 'unitary':
                self.alpha_hats = np.ones(self.channel.L)
            
            athetai = np.array([avec(angle,self.cb_graph.M) @ np.conj(current_node.f) for angle in np.array([self.cb_graph.nodes[midx].steered_angle for midx in self.cb_graph.level_midxs[-1]])])
            nn_flops += 2 * self.cb_graph.M * N
            
            mus = np.sum([alpha_hats[ll] *  athetai for ll in np.arange(self.channel.L)],axis = 0)
            nn_flops += self.channel.L
            
            if self.channel.sigma_v <= .05: sigma_v_temp = .05
            else: sigma_v_temp = dcp(self.channel.sigma_v)
            pdfs =  1/(np.pi * sigma_v_temp**2) * np.exp(-np.abs(z-mus)**2/(sigma_v_temp**2))
            
            nn_flops += 5*N

            sum_vec = np.sum(pim1*pdfs) + 1e-30
            posteriors = pim1*pdfs / (sum_vec) + 1e-30
            nn_flops += 4*N -1

            posteriors = 1/np.sum(posteriors) * posteriors #only for normalizing, not explicitly written in algorithm
            est_best_midx = self.cb_graph.level_midxs[-1][np.argmax(posteriors)]
            
            #Update posteriors for next timestep based on the motion, perfectly known to agent.
            if self.channel.mode == 'WGNA':
                
                #12 flops to computer sigma_nnu2
                sigma_nnu2 = self.channel.tau**4/4 * self.channel.sigma_u**2 * (4*nn**3/3 - 4*nn**2 + 11*nn/3 -1) + self.channel.tau**2*(nn-1)**2*self.channel.sigma_u**2
                
                if np.sign(sigma_nnu2) == -1: sigma_nnu2 = 1e-32
                sigma_nnu = np.sqrt(sigma_nnu2)
                
                nn_flops += 13.0
                
                pdf_u = []
                for midx in self.cb_graph.level_midxs[current_node.h]:
                    pdf_u_beam = unif_mean_normal_pdf(nodes[midx].steered_angle, sigma_nnu, nodes[midx].steered_angle - self.cb_graph.beamwidths[current_node.h]/2, nodes[midx].steered_angle + self.cb_graph.beamwidths[current_node.h]/2)
                    pdf_u.append(pdf_u_beam)
                    nn_flops += 7.0
                pdf_u = np.repeat(pdf_u,int(2**(H-current_node.h-1))) + 1e-30
                pdf_u = 1/np.sum(pdf_u) * pdf_u
                pim1 = posteriors * pdf_u + 1e-30
                
            elif self.channel.mode == 'FixedV':
                # v = self.channel.sigma_u * len(self.cb_graph.level_midxs[-1]) / np.pi  #Per the expression following (14) in [1]
                v = self.channel.sigma_u / self.cb_graph.beamwidths[-1] #Per the expression following (14) in [1] where delta * pi is 1/N * pi, or the division of the swath of angles (in other words, the beamwidth)
                if v < 1:
                    pim1 = posteriors * (1-v) + v * np.roll(posteriors,int(np.sign(v))) + 1e-30
                else: 
                    pim1 = np.roll(posteriors,int(v)) + 1e-30 
                
            elif self.channel.mode == 'GaussianJumps':
                pim1 = []
                for phi in np.array([self.cb_graph.nodes[midx].steered_angle for midx in self.cb_graph.level_midxs[-1]]):
                    a_trunc = phi - np.pi/self.cb_graph.M/2
                    b_trunc = phi + np.pi/self.cb_graph.M/2
                    a, b = (a_trunc - phi) / self.channel.sigma_u, (b_trunc - phi) / self.channel.sigma_u
                    x = np.linspace(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1], self.cb_graph.M)
                    pdf_phi = truncnorm.pdf(x = x,a = a,b = b,loc = phi, scale = self.channel.sigma_u) + 1e-30
                    pdf_phi = pdf_phi/np.sum(pdf_phi)
                    pim1.append( posteriors @ pdf_phi + 1e-30)
                    
            pim1 = pim1/np.sum(pim1)
            #posteriors = np.array(pim1)/np.sum(pim1)
            
            #Estimated best at this iteration
            
            self.log_data['path'].append(est_best_midx)
            self.log_data['flops'].append(nn_flops)
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(nodes[est_best_midx]))
            
            self.channel.fluctuation(nn,self.cb_graph.min_max_angles)
            self.set_best()
            
        return log_data
    
    
    
    
## Track and Stop Algorithms
            
class TASD(AlgorithmTemplate):
    """
    Description
    -----------
    Track and Stop D (TASD) [1] algorithm performed on midxs specified from the cb_graph object passed on initialization (see documentaiton for AlgorithmTemplate)
    
    [1] Garivier, Aurélien, and Emilie Kaufmann. "Nonasymptotic sequential tests for overlapping hypotheses applied to near-optimal arm identification in bandit models." Sequential Analysis 40.1 (2021): 61-96.

    Original Julia code at
    https://github.com/EmilieKaufmann/BAI_epsBAI_code/blob/master/EpsilonBAIalgos.jl
    
    
    Attributes
    ----------
    delta : float
        Confidence parameter for upper confidence bound calculations.
    epsilon : float
        Tolerance paramter in which rewards are epsilon-optimal if mu + epsilon>= mu_max
        
    Methods
    -------
    __init__(self, params):
        Initializes the HOSUB algorithm with the provided parameters.
    run_base_alg(self):
        Runs the TASD algorithm to find the best beamforming vector.
    update_node(self, node, r):
        Updates the node's empirical mean reward and upper confidence bound.
    """
    
    def __init__(self,params):
        """
        Initializes the TASD algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph': object
                The codebook graph associated with the simulation.
            - 'channel': object
                The communication channel used in the simulation.
            - 'delta': float
                Confidence parameter for upper confidence bound calculations.
              'epsilon' : float
                Tolerance paramter in which rewards are epsilon-optimal if mu + epsilon>= mu_max
              'transmit_power_dbm' : float (optional)
                Transmit power in dBW, defaults to 0 (1 Watt in linear units)
        """
        super().__init__(params)
        self.delta = params['delta']
        self.epsilon = params['epsilon']
        if 'transmit_power_dbm' in params: self.transmit_power_dbm = params['transmit_power_dbm']
        else: self.transmit_power_dbm = 0
        if 'mode' in params: self.mode = params['mode']
        else: self.mode = 'stationary'
        
        #Initialize node attributes and method for bandit algorithm
        for node in self.cb_graph.nodes.values():
            node.mean_reward = self.sample(node,with_noise = False)
            node.empirical_mean_reward = 0.0
            node.num_pulls = 0.0
        
        self.set_best()
        
    def run_base_alg(self, current_midxs = None,time_horizon = 100000,update_logs = True, verbose = False):
        """
        Description
        -----------
        Runs the Track and Stop algorithm from [1] with Chernoff stopping + D-Tracking
        
        Parameters
        ----------
        current_midxs : array or list of ints
            master indices from codebook object corresponding to nodes which are played in 
            contention in a MAB game.  If left unspecified, then by default all of the master indices belonging to nodes
            with the narrowest beams are used.
            
        Subordinate Functions
        ---------------------
        rate(t, delta):
            Computes the rate function.
        d(mu1, mu2):
            Computes the divergence between two empirical values.
        lambdaX(x, mua, mub, epsilon, pre=1e-12):
            Computes the minimizer for lambda in (mu^- ; mu^+ - epsilon).
        gb(x, mua, mub, epsilon, pre=1e-12):
            Computes the minimum value of d(mua, lambda) + d(mub, lambda + epsilon).
        AdmissibleAux(mu, a, epsilon):
            Computes the admissible auxiliary values for a given arm.
        xbofy(y, mua, mub, epsilon, pre=1e-12):
            Finds x such that g_b(x) = y.
        dicoSolve(f, xMin, xMax, pre=1e-11):
            Finds m such that f(m) = 0 using binary search.
        auxEps(y, mu, a, epsilon, pre=1e-12):
            Returns F_mu(y) - 1.
        aOpt(mu, a, epsilon, pre=1e-12):
            Returns the optimal weights and values for the epsilon optimal arm.
        OptimalWeightsEpsilon(mu, epsilon, pre=1e-11):
            Returns T*(mu) and a matrix containing the candidate optimal weights.
        PGLRT(muhat, counts, epsilon, Aeps, K):
            Computes the parallel GLRT stopping rule and returns the best arm.
            
        Returns
        -------
        int
            The midx corresponding to the estimated best beamforming vector as indexed by the codebook graph.
        int
            The number of samples prior to terminating
        """
        if current_midxs is None:
            print("Warning: 'current_midxs' arg not specified, now set to midxs corresponding to nodes with narrowest beamforming vectors.  Exhaustive Sweep.")
            current_midxs = self.cb_graph.level_midxs[-1]
            
        ## Subordinate Functions
        def rate(t,delta): return np.log((np.log(t) + 1)/delta)
        
        def d(mu1, mu2):
            if mu2 < 0: 
                #print('Empirical value mu2-eps < 0, invalid value, need smaller epsilon')
                #mu2 = 1e-32
                return np.inf
            myd = 1/2 * np.log(mu2/mu1) + mu1/2/mu2 + (mu1 - mu2)**2 / 4 / mu2 / self.channel.sigma_v**2 - 1/2
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
                def func(lam): return d(mua,lam)  + x * d(mub,lam + epsilon) #From (5.3) (Might have been (5.3) in earlier version of paper)
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
            
            #This got stuck in an infinite loop whenever yMax==np.inf, tried to manually constrain the range of values.
            #Trying to compute fun(100000) always got hung up too.
            def fun(y): return auxEps(y,mu,a,epsilon,pre)
            if yMax==np.inf:
                # yMax=1
                # while fun(yMax)<0:
                #     yMax=yMax*2
                yMax = 10000
            # yMax = 2.0
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

        def PGLRT(muhat,counts,epsilon,Aeps,K):
            # compute the parallel GLRT stopping rule and return the Best arm 
            # counts have to be all positive 
            Aepsilon = [Aeps[i] for i in np.arange(len(Aeps))]
            L = len(Aepsilon)
            Zvalues = []
            for i in np.arange(L):
                a = Aepsilon[i]
                NA = counts[a]
                MuA = muhat[a]
                Zvalues.append(np.min([NA*gb(counts[b]/NA,MuA,muhat[b],epsilon) for b in np.arange(K) if b!=a]))
            # pick an argmin
            Ind = np.argmax(Zvalues)
            Best = Aepsilon[Ind]
            return np.max(Zvalues),Best
        
        # Main Algorithm Execution: Chernoff stopping + D-Tracking 
        nodes = self.cb_graph.nodes
        condition = True
        K=len(current_midxs)
        nn = 0
        #Sample each starting level beamforming vector once
        for midx in current_midxs:
            self.update_node(nodes[midx],self.sample(nodes[midx],transmit_power_dbm = self.transmit_power_dbm),current_midxs)
            
            nn += 1
        empirical_mean_rewards = [nodes[midx].empirical_mean_reward for midx in current_midxs]
        num_pulls = [nodes[midx].num_pulls for midx in current_midxs]

        est_best_midx = current_midxs[0]
        while True:
            #print(nn)
            # Empirical best arm
            Is = np.where(empirical_mean_rewards == np.max(empirical_mean_rewards))[0]
            I = Is[0] # In case the algorithm stops after initial sweep
            
            # Compute the stopping statistic
            Score,est_best_midx=PGLRT(empirical_mean_rewards,num_pulls,self.epsilon,Is,K)
            #if params['verbose']:
            #print(f'n = {nn}, Score {Score}, rate {rate(nn,self.delta)}\n')
    
            #Check to see if stopping criteria is met, if so, break loop
            if self.mode == 'non-stationary': nn_count = np.min([nn,nodes[0].W])
            else: nn_count = dcp(nn)
            if (Score > rate(nn_count,self.delta)):
                if nn == K and verbose: print('Info: Algorithm stopped after initial beam sweep.')
                break
            
            #Check to see if the algorithm is hung up, if so, break loop
            elif (nn >time_horizon):
                est_best_midx = current_midxs[np.argmax(empirical_mean_rewards)]
                print(f'Info: Stopped after {nn} samples, algorithm reached max time horizon.')
                return est_best_midx, nn
            
            #If the algorithm meets neither of the stopping criteria, continue sampling
            else:
                if (np.min(num_pulls) <= np.max([np.sqrt(nn) - K/2,0])):
                    # forced exploration
                    index_to_sample=np.argmin(num_pulls)
                else:
                    # continue and sample an arm
                    val,Weights=OptimalWeightsEpsilon(empirical_mean_rewards,self.epsilon,1e-11)
                    index_to_sample=np.argmax(np.array(Weights)-np.array(num_pulls)/nn)
            
            midx_to_sample = current_midxs[index_to_sample]
            
            # Draw the arm and update rewards
            self.update_node(nodes[midx_to_sample],self.sample(nodes[midx_to_sample],transmit_power_dbm = self.transmit_power_dbm),current_midxs)
            empirical_mean_rewards = [nodes[midx].empirical_mean_reward for midx in current_midxs]
            num_pulls = [nodes[midx].num_pulls for midx in current_midxs]
            nn+=1
            self.channel.fluctuation(nn,self.cb_graph.min_max_angles)
            self.set_best()
            
        est_best_midx = current_midxs[np.argmax(empirical_mean_rewards)]
        if update_logs:
            self.log_data['samples'].append(nn)
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(nodes[est_best_midx]))
            self.log_data['path'].append(est_best_midx)
        return est_best_midx,nn

    def update_node(self,node,r,current_midxs):
        """
        Description
        -----------
        Updates the node's empirical mean reward and upper confidence bound.

        This method updates the number of pulls, empirical mean reward, and upper
        confidence bound for the specified node based on the new sample reward.

        Parameters
        ----------
        node : object
            The node to be updated.
        r : float
            The new sample reward.

        Returns
        -------
        None
        """
        node.num_pulls += 1
        node.empirical_mean_reward = ((node.num_pulls-1) * node.empirical_mean_reward + r)/node.num_pulls

class NPHTS(TASD):
    """
    Description
    -----------
    Phased Track and Stop Beamforming approach from  [1]. This class implements the NPHTS algorithm, an extension of the TASD class, where TASD is run for each level
    specified by attribute p ('levels_to_play') in the hierarchical beamforming codebook.
    
    [1] Wei, Yi, Zixin Zhong, and Vincent YF Tan. "Fast beam alignment via pure exploration in multi-armed bandits." IEEE Transactions on Wireless Communications (2022).

    Original code at:
    https://github.com/YiWei0129/Fast-beam-alignment

    Parameters
    ----------
    params : dict
        A dictionary of parameters for initializing the NPHTS algorithm. 
        Must include 'levels_to_play', 'delta', and 'epsilon'.

        - levels_to_play: List[int]
            A list indicating the levels to be played in the algorithm.
        - delta: List[float]
            A list of delta values for each level.
        - epsilon: float
            The epsilon value for the algorithm.

    Raises
    ------
    AssertionError
        If the length of 'levels_to_play' does not match the length of 'delta'.
        If the last element of 'levels_to_play' is not 1.
        
    Methods
    -------
    __init__(self, params):
        Initializes the NPHTS algorithm with the provided parameters.

    run_alg(self):
        Runs the NPHTS algorithm to find the best beamforming vector.

    Notes
    -----
    Relies on the TASD class for the update_node method.
    
    Slightly modified from the implementation in [3] which has a fixed two-phase approach, whereas here we
    allow the user to select which levels to play by the key in params 'levels_to_play'.

    """
    def __init__(self,params):
        """
        Description
        -----------
        Initializes the NPHTS algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary of parameters for initializing the NPHTS algorithm. 
            Must include 'levels_to_play', 'delta', and 'epsilon'.

            - levels_to_play: List[int]
                A list indicating the levels to be played in the algorithm.
            - delta: List[float]
                A list of delta values for each level.
            - epsilon: float
                The epsilon value for the algorithm.
            - transmit_power_dbm : float (optional)
                Transmit power in dBW, defaults to 0 (1 Watt in linear units)
        Raises
        ------
        AssertionError
            If the length of 'levels_to_play' does not match the length of 'delta'.
            If the last element of 'levels_to_play' is not 1.
        """
        super().__init__(params)
        self.p = params['levels_to_play']
        self.deltas = params['delta']
        #self.epsilon = params['epsilon']
        
        assert len(self.p) == len(self.delta), "Error: The number of deltas provided must match the number of levels played, the number of 1s in 'levels_to_play'."
        assert self.p[-1] == 1, "Last element of 'levels_to_play' must be 1."
        
        if 'transmit_power_dbm' in params: self.transmit_power_dbm = params['transmit_power_dbm']
        else: self.transmit_power_dbm = 0
        
    def run_alg(self):
        """
        Description
        -----------
        Runs the NPHTS algorithm to find the best beamforming vector.
        
        This method implements the core logic of the NPHTS algorithm, leveraging 
        the parameters initialized in the __init__ method.
        
        Returns
        -------
        int
            The midx corresponding to the estimated best beamforming vector as indexed by the codebook graph.
        int
            The number of samples prior to terminating
        """
        nn = 0
        nodes = self.cb_graph.nodes
        current_midxs = self.cb_graph.base_midxs[0]
        for hh in np.arange(self.cb_graph.H):  #Breaks when there is only a singles node from the zoom in midxs
            self.delta = self.deltas[hh]
            if self.p[hh] == 0:
                current_midxs = np.hstack([nodes[midx].zoom_in_midxs for midx in current_midxs])
            else:
                #tas_instance_h = TASD({'cb_graph' : self.cb_graph, 'channel' : self.channel, 'delta' : self.deltas[hh], 'epsilon' : self.epsilon})
                recommendation_midx,num_samples = self.run_base_alg(current_midxs,update_logs = True)
                
                #TODO: Need to add logic to handle breaks due to hitting time horizon.
                
                # self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(nodes[recommendation_midx]))
                # self.log_data['path'].append(recommendation_midx)
                # self.log_data['samples'].append(num_samples)
                #Include the best sibling of recommended beam
                if recommendation_midx == nodes[recommendation_midx].post_sibling:
                    recommendation_midxs = [nodes[recommendation_midx].prior_sibling, recommendation_midx]
                elif recommendation_midx == nodes[recommendation_midx].prior_sibling:
                    recommendation_midxs = [nodes[recommendation_midx].post_sibling, recommendation_midx]
                else:
                    if nodes[nodes[recommendation_midx].prior_sibling].empirical_mean_reward > nodes[nodes[recommendation_midx].post_sibling].empirical_mean_reward:
                        recommendation_midxs = [nodes[recommendation_midx].prior_sibling, recommendation_midx]
                    elif nodes[nodes[recommendation_midx].prior_sibling].empirical_mean_reward < nodes[nodes[recommendation_midx].post_sibling].empirical_mean_reward:
                        recommendation_midxs = [nodes[recommendation_midx].post_sibling, recommendation_midx]
                        
                current_midxs = np.hstack([nodes[midx].zoom_in_midxs for midx in recommendation_midxs])
                nn += num_samples
        self.log_data['samples'] = [np.sum(self.log_data['samples'])]
        return
                
class MotionTS(TASD):
    """
    Description
    -----------
    Phased Track and Stop Beamforming approach from where only recent samples are considered.  In the case that an empirical mean chosen is less than any previous
    level's, zoom out.

    Parameters
    ----------
    params : dict
        A dictionary of parameters for initializing the NPHTS algorithm. 
        Must include 'levels_to_play', 'delta', and 'epsilon'.

        - levels_to_play: List[int]
            A list indicating the levels to be played in the algorithm.
        - delta: List[float]
            A list of delta values for each level.
        - epsilon: float
            The epsilon value for the algorithm.

    Raises
    ------
    AssertionError
        If the length of 'levels_to_play' does not match the length of 'delta'.
        If the last element of 'levels_to_play' is not 1.
        
    Methods
    -------
    __init__(self, params):
        Initializes the NPHTS algorithm with the provided parameters.

    run_alg(self):
        Runs the NPHTS algorithm to find the best beamforming vector.

    Notes
    -----
    Relies on the TASD class for the update_node method.
    
    Slightly modified from the implementation in [3] which has a fixed two-phase approach, whereas here we
    allow the user to select which levels to play by the key in params 'levels_to_play'.

    """
    def __init__(self,params):
        """
        Description
        -----------
        Initializes the algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary of parameters for initializing the NPHTS algorithm. 
            Must include 'levels_to_play', 'delta', and 'epsilon'.

            - levels_to_play: List[int]
                A list indicating the levels to be played in the algorithm.
            - delta: List[float]
                A list of delta values for each level.
            - sample_window_lengths : list[int]
                length of each sample window at level h
            - epsilon: float
                The epsilon value for the algorithm.
            - mode : str
            - transmit_power_dbm : float (optional)
                Transmit power in dBW, defaults to 0 (1 Watt in linear units)

        Raises
        ------
        AssertionError
            If the length of 'levels_to_play' does not match the length of 'delta'.
            If the last element of 'levels_to_play' is not 1.
        """
        super().__init__(params)
        #self.p = params['levels_to_play']
        self.deltas = params['delta']
        self.W = params['sample_window_lengths']
        self.epsilon = params['epsilon']
        self.mode = params['mode']
        if 'transmit_power_dbm' in params: self.transmit_power_dbm = params['transmit_power_dbm']
        else: self.transmit_power_dbm = 0
        if 'scale_reward' in params: self.scale_reward = params['scale_reward']
        else: self.scale_reward = False
        #assert len(self.p) == len(self.delta), "Error: The number of deltas provided must match the number of levels played, the number of 1s in 'levels_to_play'."
        #assert self.p[-1] == 1, "Last element of 'levels_to_play' must be 1."
        
        #Initialize node attributes and method for bandit algorithm
        for node in self.cb_graph.nodes.values():
            node.sample_history = []
            node.W = self.W[node.h]
            
    def run_alg(self,time_horizon):
        """
        Description
        -----------
        Runs the algorithm to find the best beamforming vector.
        
        This method implements the core logic of the NPHTS algorithm, leveraging 
        the parameters initialized in the __init__ method.
        
        Parameters
        ----------
        time_horizon : int
            Number of time steps to run the simulation
        
        Returns
        -------
        int
            The midx corresponding to the estimated best beamforming vector as indexed by the codebook graph.
        int
            The number of samples prior to terminating
        """
        nn = 0
        nodes = self.cb_graph.nodes
        self.comm_node = None
        current_midxs = self.cb_graph.base_midxs[0]
        for midx in current_midxs:
            self.update_node(nodes[midx],self.sample(nodes[midx],transmit_power_dbm=self.transmit_power_dbm),current_midxs)
            nn += 1
        previous_empirical_mean_rewards = [-np.inf] #impossible to zoom out at level h = 0
        while nn < time_horizon:
            self.delta = self.deltas[nodes[current_midxs[0]].h]
            
            if len(current_midxs) > 1:
                #print(nn)
                recommendation_midx,num_samples = self.run_base_alg(current_midxs,time_horizon = time_horizon,update_logs = False)
                nn += num_samples
                #Zoom out if any of the previous empirical mean rewards are greater than the one most recently obtained
                if np.any(nodes[recommendation_midx].empirical_mean_reward < np.array(previous_empirical_mean_rewards)):
                    
                    self.comm_node = nodes[self.comm_node.zoom_out_midx]
                    current_midxs = dcp(self.comm_node.zoom_in_midxs)
                    
                    previous_empirical_mean_rewards.pop(-1)
                    
                #Zoom in otherwise, and make the new set of current midxs the zoom-in indices of the chosen beam
                else:
                    self.comm_node = nodes[recommendation_midx]
                    current_midxs = dcp(self.comm_node.zoom_in_midxs)
                    previous_empirical_mean_rewards.append(self.comm_node.empirical_mean_reward)
                
                #Initialize new set of nodes in current_midxs, this performs a round-robin sampling
                for midx in current_midxs:
                    self.update_node(nodes[midx],self.sample(nodes[midx],transmit_power_dbm=self.transmit_power_dbm),current_midxs)
                    nn += 1
                
            #If we are fully zoomed in alrady, then we just sample the comm_node to update it's empirical mean.
            else:
                self.update_node(self.comm_node,self.sample(self.comm_node,transmit_power_dbm=self.transmit_power_dbm),current_midxs)
                nn += 1
                if np.any(self.comm_node.empirical_mean_reward < np.array(previous_empirical_mean_rewards)):
                    # 1. This just resets the MAB game.  comm_node unchanged
                    # current_midxs = nodes[nodes[recommendation_midx].zoom_out_midx].zoom_in_midxs
                    
                    # 2. This zooms out to the level prior to the comm_node, which is now in contention with it's neighbors.
                    self.comm_node = nodes[self.comm_node.zoom_out_midx]
                    current_midxs = dcp(self.comm_node.zoom_in_midxs)
                    previous_empirical_mean_rewards.pop(-1)
            
            
                
    def update_node(self,node,r,current_midxs):
        """
        Description
        -----------
        Updates the node's sample history, and deletes "old" samples as specified by the sampling window length "W"
        
        In the non-stationary setting, this "ages out" older samples of other nodes
        
        Parameters
        ----------
        node : object
            The node to be updated.
        r : float
            The new sample reward.
            
        Returns
        -------
        None
        """
        
        nodes = self.cb_graph.nodes
        
        #Update Node that was just sampled
        if self.scale_reward:
            node.sample_history.append(1 * r)
        else:
            node.sample_history.append(r)
        if len(node.sample_history) > node.W:
            node.sample_history.pop(0)
        
        #Age-out older samples on other nodes
        if self.mode == 'non-stationary':
            for other_node in self.cb_graph.nodes.values():
                if other_node.midx != node.midx:
                    other_node.sample_history.append(0.0)
                    if len(other_node.sample_history) > node.W:
                        other_node.sample_history.pop(0)
        
        for midx in current_midxs:
            nodes[midx].num_pulls = len(np.where(np.array(nodes[midx].sample_history) != 0)[0])
            if nodes[midx].num_pulls > 0:
                nodes[midx].empirical_mean_reward = 1/nodes[midx].num_pulls * np.sum(nodes[midx].sample_history)
                
        if self.comm_node == None: 
            self.log_data['relative_spectral_efficiency'].append(0.0)
            self.log_data['path'].append(np.nan)
        else: 
            rse = self.calculate_relative_spectral_efficiency(self.comm_node)
            if rse > 1:
                print('pause')
            self.log_data['relative_spectral_efficiency'].append(rse)
            self.log_data['path'].append(self.comm_node)
        self.log_data['samples'].append(1.0)
        
        self.channel.fluctuation()
        self.set_best()
if __name__ == '__main__':
    main()