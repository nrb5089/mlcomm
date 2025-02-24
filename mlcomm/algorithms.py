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
        
    def sample(self, node, transmit_power_dbm = 0, with_noise=True, mode='rss'): 
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
            if nodes[current_best_midx].h == self.cb_graph.H-1: 
                self.log_data['relative_spectral_efficiency'].append( self.calculate_relative_spectral_efficiency(nodes[current_best_midx]))
                return
            
            #Update neighborhood
            if nodes[current_best_midx].h == self.h0:
                current_neighborhood_midxs = [nodes[current_best_midx].prior_sibling,nodes[current_best_midx].post_sibling, nodes[current_best_midx].zoom_in_midxs[0], nodes[current_best_midx].zoom_in_midxs[1]]
            else:
                try:
                    current_neighborhood_midxs = [nodes[current_best_midx].zoom_out_midx, nodes[current_best_midx].zoom_in_midxs[0], nodes[current_best_midx].zoom_in_midxs[1]]
                except:
                    pass
            nodes[current_best_midx].num_lead_select += 1
            
            #Maximum degree of any node is 4, hence we use the integer check from line 12 in Algorithm 1 in [1].  Otherwise, sample the highest UCB in the neighborhood.
            if np.mod((nodes[current_best_midx].num_lead_select-1)/4,1) == 0:
                node_to_sample = nodes[current_best_midx]
            else:
               node_to_sample = nodes[current_neighborhood_midxs[np.argmax([nodes[midx].ucb for midx in current_neighborhood_midxs])]]
              
            #Sample node and update statistics
            self.update_node(node_to_sample,self.sample(node_to_sample))
            self.channel.fluctuation(nn, (self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
            self.set_best()
            
            nn += 1
            
        est_best_midx =  np.argmax([node.empirical_mean_reward for node in nodes.values()])
        self.log_data['relative_spectral_efficiency'].append( self.calculate_relative_spectral_efficiency(nodes[est_best_midx]))
        
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
        # if self.mode == 'stationary': time_horizon = 10000000 #Hard stop algorithm time ceiling
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
        if self.mode == 'stationary':
            self.log_data['path'].append(gamma_midx)
            self.log_data['samples'] = [nn]
            if self.comm_node == None:
                self.log_data['relative_spectral_efficiency'].append(0.0) 
            else:
                self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
            return gamma_midx,nn
            
            
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


class HBA(AlgorithmTemplate):
    """
    Description
    -----------
    Implements Hierarchical Beam Alignment (HBA) from [1]
    
    
    [1] Wu, Wen, et al. "Fast mmwave beam alignment via correlated bandit learning." IEEE Transactions on Wireless Communications 18.12 (2019): 5894-5908.
    
    
    Attributes
    ----------
    
    Methods
    -------
    
    Notes 
    -----
    
    """
    def __init__(self,params):
        super().__init__(params)
        
        self.zeta = params['zeta']
        self.gamma = params['gamma']
        self.rho1 = params['rho1']
        
        self.N = len(self.cb_graph.level_midxs[-1])                                    #Number of beam directions
        # self.hmax = int(np.log2(self.N)+1)                       #Maximum Tree depth
        self.hmax = self.cb_graph.H + 1
        self.Q = np.inf * np.ones([self.hmax,self.N])            #Q-values for algorithm
        
        
        self.Nt = np.zeros([self.hmax,self.N])                       
        self.Rt = np.zeros([self.hmax,self.N])
        self.Et = np.zeros([self.hmax,self.N])    
    


        
    def run_alg(self, time_horizon):
        
        def xa(xH,xL): return xL + (xH-xL)/2
        
        nodes = self.cb_graph.nodes
        self.Tcal = list([(0,0)])                               #Initialize tree to be constructed.
        xH,xL = (1,0)
        relative_spectral_efficiencies = []
        # t = 1
        # flops = []
        nn = 1
        while nn <= time_horizon:
            # print(t)
            # t_flops = 0.0
            h,j = (0,0)
            Pcal = list([(h,j)])
            xH,xL = (1,0)
            while (h,j) in self.Tcal:
                if self.Q[h+1,2*j] > self.Q[h+1,2*j+1]:    
                    (h,j) = (h+1,2*j)
                    xL = xa(xH,xL)
                    # t_flops += 3
                elif self.Q[h+1,2*j] < self.Q[h+1,2*j+1]:  
                    (h,j) = (h+1,2*j+1)
                    xH = xa(xH,xL)     
                    # t_flops += 3
                else:
                    (h,j) = (h+1, 2*j + np.random.choice([0,1]))
                
                Pcal.append((h,j))

                if h==self.hmax-1 or j == 2**h:
                    break
                    
            (Ht,Jt) = (h,j)
            if (Ht,Jt) not in self.Tcal:
                self.Tcal.append((Ht,Jt))

            sampling_idx = self.get_idx(h,j)
            node_to_sample = nodes[self.cb_graph.level_midxs[-1][sampling_idx]]
            # y = np.abs(self.sample(node_to_sample,mode = 'complex'))**2
            y = self.sample(node_to_sample,mode = 'rss')
            for (h,j) in Pcal:
                self.Nt[h,j] += 1                       #increment the number of times sampled
                self.Rt[h,j] = ((self.Nt[h,j] - 1)*self.Rt[h,j] + y)/self.Nt[h,j]
                # t_flops += 5.0
                
            for (h,j) in self.Tcal:
                if self.Nt[h,j] > 0:
                    if self.channel.sigma_v < .1: sigma = .1
                    else: sigma = self.channel.sigma_v
                    self.Et[h,j] = self.Rt[h,j] + np.sqrt(2* sigma**2 * np.log(nn)/self.Nt[h,j]) + self.rho1*self.gamma**h
                    # self.Et[h,j] = self.Rt[h,j] + np.sqrt(2* np.log(nn)/self.Nt[h,j]) + self.rho1*self.gamma**h
                    
                    # t_flops += 7.0
                else:
                    self.Et[h,j] = np.inf

            That = dcp(self.Tcal)
            for (h,j) in self.Tcal:
                if self.Nt[h,j]>0:
                    if h != self.hmax-1:
                        self.Q[h,j] = np.min([self.Et[h,j],np.max([self.Q[h+1,2*j],self.Q[h+1,2*j+1]])])
                        # t_flops += 3.0 
                    else:
                        self.Q[h,j] = self.Et[h,j]
                That.remove((h,j))
            
            # flops.append(t_flops)
            # print(xH-xL)
            if xH - xL <= self.zeta/self.N or nn == time_horizon: #Stopping criteria, still not clear what's going on 
                max_el = np.max(self.Et)
#                    
                self.mh,self.mj = np.where(self.Et == max_el) #Find current arm providing max rewards
                self.mh = self.mh[0]
                self.mj = self.mj[0]
                est_best_idx = self.get_idx(self.mh,self.mj)
                est_best_node = nodes[self.cb_graph.level_midxs[-1][est_best_idx]]
                self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(est_best_node))
                self.log_data['samples'].append(nn)
                return
            
            self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
            self.set_best()
            nn += 1

    def get_idx(self,h,j):
        def xa(xH,xL): return xL + (xH-xL)/2
        C = (j/2**h, (j+1)/2**h)
        mid_point =xa(C[0],C[1])
        idx = int(mid_point*self.N)
        return idx


    # def __init__(self,params):
    #     """
    #     Description
    #     ------------
        

    #     Parameters
    #     ----------
    #     params : dict
    #         'zeta'
    #         Hyperparameter governing stopping time, for a fixed resolution codebook, this should be set to be just finer than the resolution of the beamwidth.
    #         'rho1' 
    #         First hyperparameter dictating confidence bound width
    #         'gamma'
    #         Second hyperparameter dictating confidence bound width, scaled by level h
            
    #     Returns
    #     -------
    #     None.

    #     """
    #     super().__init__(params)
    #     self.zeta = params['zeta']
        
    #     nodes = self.cb_graph.nodes
    #     self.regions = []
    #     for hh in range(self.cb_graph.H):
    #         regions_h = []
    #         for jj in range(int(2**(hh+1))):
    #             region_center_pointing_angle = self.cb_graph.min_max_angles[0] + (self.cb_graph.min_max_angles[1] - self.cb_graph.min_max_angles[0]) * (1/(2**(hh+2)) + jj*1/(2**(hh+1)))
    #             closest_midx = self.cb_graph.level_midxs[-1][np.argmin([np.abs(region_center_pointing_angle - nodes[midx].steered_angle) for midx in self.cb_graph.level_midxs[-1]])]
    #             regions_h.append(HBA.Region({'hh': hh, 'jj': jj, 'corr_midx' : closest_midx, 'region_center_pointing_angle' : region_center_pointing_angle, 'max_depth' : self.cb_graph.H, 'rho1' : params['rho1'], 'gamma' : params['gamma']}))
    #         self.regions.append(regions_h)
        
    # def run_alg(self,time_horizon):
    #     """
    #     Description
    #     -----------
        
    #     """
    #     def xa(xL,xH): return xL + (xH-xL)/2
    #     def get_closest_midx(xL,xH):
    #         """
    #         Finds the midx corresponding to the beam with the pointing angle closest to the midpoint of xL and xH
    #         """
    #         required_pointing_angle = self.cb_graph.min_max_angles[0] + (self.cb_graph.min_max_angles[1]-self.cb_graph.min_max_angles[0]) * xa(xL,xH)
    #         closest_midx = self.cb_graph.level_midxs[-1][np.argmin([np.abs(required_pointing_angle - nodes[midx].steered_angle) for midx in self.cb_graph.level_midxs[-1]])]
    #         return closest_midx
            
    #     nodes = self.cb_graph.nodes
    #     nn = 0
    #     T = [(-1,0)]
    #     xL,xH = 0,1
    #     N = len(self.cb_graph.level_midxs[-1])
                
    #     while nn < time_horizon:
    #         hh, jj = -1,0
    #         P = []
    #         while (hh,jj) in T:
    #             if self.regions[hh+1][int(2*jj)].Q > self.regions[hh+1][int(2*jj+1)].Q:
    #                 (hh,jj) = (hh+1, int(2*jj))
    #                 xL = xa(xL,xH)
    #             elif self.regions[hh+1][int(2*jj)].Q < self.regions[hh+1][int(2*jj+1)].Q:
    #                 (hh,jj) = (hh+1,int(2*jj+1))
    #                 xH = xa(xL,xH)
    #             else:
    #                 (hh,jj) = (hh + 1, np.random.choice([int(2*jj),int(2*jj+1)]))
    #             P.append((hh,jj))
                
    #             if hh == self.cb_graph.H-1 or jj == 2**hh:
    #                 break
                
    #         Hnn,Jnn = dcp((hh,jj))
    #         T.append((Hnn,Jnn))
    #         r = self.sample(nodes[self.regions[Hnn][Jnn].corr_midx])
    #         nn +=1
            
    #         # Update rewards along the path
    #         for (hh,jj) in P:
    #             self.regions[hh][jj].update_R(r)
                
    #         # Update statistics through the tree
    #         for (hh,jj) in T:
    #             if hh == -1: pass
    #             else: self.regions[hh][jj].update_E(self.channel.sigma_v,nn)
            
    #         # Update Q values working backwards through the graph (bottom to top)
    #         That = dcp(T)
    #         h_max = np.max([elem[0] for elem in That])
    #         for hh in np.arange(0,h_max+1)[-1::-1]:
    #             leafs = [elem for elem in That if elem[0] == hh]
    #             for leaf in leafs: 
    #                 if leaf[0] == self.cb_graph.H-1:
    #                     self.regions[leaf[0]][leaf[1]].Q = dcp(self.regions[leaf[0]][leaf[1]].E)
    #                 else:
    #                     self.regions[leaf[0]][leaf[1]].Q = np.min([self.regions[leaf[0]][leaf[1]].E,np.max([self.regions[leaf[0]+1][int(2*leaf[1])].Q,self.regions[leaf[0]+1][int(2*leaf[1]+1)].Q])])
    #                 That.remove(leaf)
                
    #         if xH-xL < self.zeta/N and P[-1][0] == self.cb_graph.H-1: 
    #             est_best_midx = get_closest_midx(xL, xH)
                
    #             self.log_data['path'].append(est_best_midx)
    #             self.set_best()
    #             self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(nodes[est_best_midx]))
    #             self.log_data['samples'] = [nn]
    #             return 
    
    # class Region:
    #     """
    #     Description
    #     -----------
    #     Basic helper object to store bandit statistics for each region
        
    #     Attributes
    #     ----------
    #     corr_midx : int
    #         midx corresponding to beamforming vector in beamforming codebook object node.
    #     """
    #     def __init__(self,params):
    #         self.hh = params['hh']
    #         self.jj = params['jj']
    #         self.max_depth = params['max_depth']
    #         self.corr_midx = params['corr_midx']
    #         self.fac = params['rho1'] * params['gamma']**(self.hh+1)
    #         self.region_center_pointing_angle = params['region_center_pointing_angle']
    #         self.Q = np.inf
    #         self.E = np.inf
    #         self.R = 0.0
    #         self.num_picks = 0.0
            
    #         if self.hh + 1 == params['max_depth']: self.partition_regions = []
    #         else: self.partition_regions = [(self.hh + 1, 2*self.jj), (self.hh + 1, 2*self.jj + 1)]
        
    #     def update_R(self,r):
    #         self.num_picks += 1
    #         self.R = ((self.num_picks -1) * self.R + r)/self.num_picks
            
    #     def update_E(self,sigma,nn):
    #         if self.num_picks > 0:
    #             # self.E = self.R + np.sqrt(2*sigma**2*nn/self.num_picks) + self.fac
    #             self.E = self.R + np.sqrt(.01*sigma**2*nn/self.num_picks) + self.fac
    #         # if self.hh +1 == self.max_depth:
    #         #     self.Q = dcp(self.E)
    #         # else:
    #         # self.Q = np.min([self.E, np.max([Q0,Q1])])
                                 
class OffsetMAB(AlgorithmTemplate):
    """
    Description
    -----------
    Implements the Algorithm 1 OR ALgorithm 2 from [1]  using the confidence terms from the DBZ paper in stationary and non-stationary environments. Only uses the narrowest beamforming vectors, but the "arms" are actually the offset from the arm being used for communication. In updating the rewards, the node assigned the sample history was rolled from another.  This mechanic accounts for the motion, as described in [1].
    
    There are some adaptation differences with aligning the time steps with the use of a beamforming vector.  The algorithm in [1] 
    executes a beam sweep of several beamforming vectors at a single time step, which does not match my time scale.  
    
    [1] Zhang, Jianjun, et al. "Beam alignment and tracking for millimeter wave communications via bandit learning." IEEE Transactions on Communications 68.9 (2020): 5519-5533.
    
    
    Parameters
    ----------
    params : dict
        dict with params
        
    Attributes
    ----------
    cb_graph : object
        The codebook graph associated with the simulation.
    epsilon : float
        'epsilon-greed' policy parameter
    alpha : float
        discount paramter for non-stationary rewards
    policy : str
        Specifies 'UCB' or 'epsilon-greedy' policy to use for beamforming vector selection
    mode : str
        Specifies 'stationary' or 'non-stationary' setting, either or is a valid entry
    c : float 
        Secondary confidence parameter
        
    Notes
    -----
    
    'epsilon' is unused in policy 'UCB' and 'c' is unused in policy 'epsilon-greedy'
    
    Example
    -------
    bandit = OffsetMAB({'cb_graph' : cb_graph, 'channel' : channel, 'epsilon' : .1, 'alpha' : 1e-4,'mode' : 'non-stationary','policy' : 'UCB', 'c' : 1})
    """
    def __init__(self,params):
        """
        """
        super().__init__(params)
        self.epsilon = params['epsilon']
        self.mode = params['mode']
        self.policy = params['policy']
        assert self.mode == 'stationary' or self.mode == 'non-stationary', 'Parameter Selection Error: Valid entries for parameter "mode" are "stationary" (default) and "non-stationary". '
        assert self.policy == 'UCB' or self.policy == 'epsilon-greedy', 'Parameter Selection Error: Valid entries for paramter "policy" are "UCB" and "epsilon-greedy". '
        self.eps = params['epsilon']
        if self.mode == 'non-stationary': self.alpha = params['alpha']
        elif self.mode == 'stationary': self.alpha = 0.0
        self.c = params['c']
        if 'transmit_power_dbm' in params: self.transmit_power_dbm = params['transmit_power_dbm']
        else: self.transmit_power_dbm = 0
    
        #Initialize ALgorithm Mechanics
        self.us = [-3,-2,-1,0,1,2,3]
        self.bs = [2,4,6]
        
    def run_alg(self,time_horizon):
        cb_graph = self.cb_graph
        nodes = self.cb_graph.nodes
        N = time_horizon
        
        self.comm_node = None
        
        action_space = [(u,b) for u in self.us for b in self.bs]
        est_avg_rewards = np.zeros(len(action_space))
        
        #Find best beamforming vector overall to start
        nn = 1
        rsss = []
        for midx in cb_graph.level_midxs[-1]:
            rsss.append(self.perform_sample_update_channel(nn,nodes[midx]))
            nn += 1
        rsss = np.array(rsss)
        est_best_midx = cb_graph.level_midxs[-1][np.argmax(rsss)]
        num_samples_space = np.ones(len(action_space))

        current_midxs = self.update_current_midxs(est_best_midx)
        current_midxs_indices_map = [[nodes[midx].i for midx in calFub] for calFub in current_midxs]

        #Update the reward acording to the maximum value obtained for a beamforming vector within each calFub (subset of beamforming vectors)
        #We do not discount the initial beam sweep, but in reality, it might have to be.
        for index in range(len(action_space)): est_avg_rewards[index] = np.max(rsss[current_midxs_indices_map[index]])

        est_best_midxs = [dcp(est_best_midx)]
        #Algorithm loop starts here
        while nn < N:
            #When you "play an arm", you are actually sampling with each beam from calFub, so depending on the cardinality of calFub, you are actually using multiple time steps
            if self.policy == 'epsilon-greedy': 
                explore = np.random.uniform(0,1) > 1-self.eps
                if explore:
                    chosen_action_index = np.random.choice(range(len(action_space)))
                else:
                    chosen_action_index = np.argmax(est_avg_rewards)
                
            elif self.policy == 'UCB': 
                ucbs = [est_avg_rewards[index] + np.sqrt(self.c * np.log(nn)/num_samples_space[index]) for index in range(len(action_space))]
                chosen_action_index = np.argmax(ucbs)
                
            #Sweep with beams from chosen arm and update rewards for all arms accroding to time out
            Rs = []
            for midx in current_midxs[chosen_action_index]:
                Rs.append(self.perform_sample_update_channel(nn,nodes[midx]))
                nn += 1
            
            #Update reward for chosen arm
            est_best_midx = current_midxs[chosen_action_index][np.argmax(Rs)]
            self.comm_node = dcp(nodes[est_best_midx])
            est_best_midxs.append(est_best_midx)
            R = np.max(Rs)
            num_samples_space[chosen_action_index] += 1
            est_avg_rewards[chosen_action_index] = dcp(est_avg_rewards[chosen_action_index]) + (R - dcp(est_avg_rewards[chosen_action_index]))/num_samples_space[chosen_action_index] #induces recursion if no dcp
            
            #Update rewards for discount
            for index in range(len(est_avg_rewards)):
                if index == chosen_action_index:
                    est_avg_rewards[index] = self.alpha * R + (1-self.alpha) * est_avg_rewards[index] 
                else:
                    est_avg_rewards[index] = (1-self.alpha) * est_avg_rewards[index] 
            
            current_midxs = self.update_current_midxs(est_best_midx)
        
        return
    
    def perform_sample_update_channel(self,nn,node_to_sample):
        """
        Description
        -----------
        Wrapper function to perform operations necessary during sampling.
        
        Notes
        -----
        Requires 2* self.cb_graph.M +1 flops for the inner product and absolute value squared.
        """
        # self.update_node(node_to_sample,self.sample(node_to_sample,transmit_power_dbm=self.transmit_power_dbm))
        r = self.sample(node_to_sample,transmit_power_dbm=self.transmit_power_dbm)
        
        if self.comm_node == None:
            self.log_data['relative_spectral_efficiency'].append(0.0)
            self.log_data['path'].append(np.nan)
        else:
            # self.set_best()
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
            self.log_data['path'].append(self.comm_node.midx)
        self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
        self.set_best()
        return r
    
    def update_current_midxs(self,est_best_midx):
        """
        Description
        -----------
        The 'current_midxs' variables is intended to track the midxs associated with the subset codebook, denoted as calF_{u,b}, in [1].
    
        Parameters
        ----------
        est_best_midx : int
            midx corresponding to the beamforming vector with the current high average rewards
    
        Returns
        -------
        current_midxs : list of ints
            list of midxs based on the indexing of (u,b) in [1]
    
        """
        #Populate each calFub (caligraphic F) according to (10) in [1], which consists of a set of subsets of beamforming vectors.
        current_midxs = []
        for u in self.us:
            for b in self.bs:
                calFub = [] 
                for bidx in range(b):
                    if est_best_midx + u + bidx < np.min(self.cb_graph.level_midxs[-1]):
                        calFub.append(est_best_midx + u + bidx + len(self.cb_graph.level_midxs[-1]))
                    elif est_best_midx + u + bidx > np.max(self.cb_graph.level_midxs[-1]):
                        calFub.append(est_best_midx + u + bidx - len(self.cb_graph.level_midxs[-1]))
                    else: 
                        calFub.append(est_best_midx + u + bidx)
                current_midxs.append(calFub)
        return current_midxs
    
    
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
            
            #mus = np.sum([alpha_hats[ll] *  athetai for ll in np.arange(self.channel.L)],axis = 0)
            mus = np.sum([alpha_hats[ll] *  athetai for ll in np.arange(1)],axis = 0)
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
                self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
                self.set_best()
            
            
        
            

class ABT(AlgorithmTemplate):
    """
    Adaptive Beam Tracking (ABT) from [1], uses initial alignment from HPM algorithm in [2]
    
    [1] Ronquillo, Nancy, and Tara Javidi. "Active beam tracking under stochastic mobility." ICC 2021-IEEE International Conference on Communications. IEEE, 2021.
    [2] Chiu, Sung-En, Nancy Ronquillo, and Tara Javidi. "Active learning and CSI acquisition for mmWave initial alignment." IEEE Journal on Selected Areas in Communications 37.11 (2019): 2474-2489.
    
    Notes
    -----
    
    
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
            
            # mus = np.sum([alpha_hats[ll] *  athetai for ll in np.arange(self.channel.L)],axis = 0)
            mus = np.sum([alpha_hats[ll] *  athetai for ll in np.arange(1)],axis = 0)
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
            
            self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
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
            self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
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
        
        self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
        self.set_best()
        
        
class EKF(AlgorithmTemplate):
    """
    Description
    -----------
    Uses the Extended Kalman Filter (EKF) from [1] to select beamforming vectors based on the nearest pointing angle.
    
     [1] V. Va, H. Vikalo and R. W. Heath, "Beam tracking for mobile millimeter wave communication systems," 2016 IEEE Global Conference on Signal and Information Processing (GlobalSIP), Washington, DC, USA, 2016, pp. 743-747, doi: 10.1109/GlobalSIP.2016.7905941. 


    Attributes
    ----------
    
    M : int
        number of antenna elements
    rho : float
        fading coefficient for channel model
    comm_node : object
        Node from codebook object currently being used to communicate
        
    
    Methods
    -------
    
    """
    def __init__(self, params):
        """
        Initializes a simulation using the EKF framework in [1] algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph': object
                The codebook graph associated with the simulation.
            - 'channel': object
                The communication channel used in the simulation.

        """
        super().__init__(params)
        self.M = self.cb_graph.M
        self.rho = self.channel.rho
        self.comm_node = None #initialize communication node as none to start
        
    def run_alg(self,time_horizon):
        nodes = self.cb_graph.nodes
        
        #Take initial samples and update time index and pad RSE with zero for intial alignment period.
        # initial_complex_samples = np.array([self.sample(nodes[midx],mode = 'complex') for midx in self.cb_graph.level_midxs[-1]])
        nn = 1
        initial_complex_samples = []
        for midx in self.cb_graph.level_midxs[-1]: 
            initial_complex_samples.append(self.sample(nodes[midx],mode = 'complex'))
            self.log_and_update_channel(nn)
            nn +=1
            
        
        #Choose beamforming vector that yields the highest RSS, choose this as comm_node and steered_angle as initial state estimate
        #Unsure how to initialize alpha, I just used the complex value received, but it might be easier just to pass the value from channel.
        est_best_idx = np.argmax(np.abs(initial_complex_samples)**2)
        est_best_midx = self.cb_graph.level_midxs[-1][est_best_idx]
        self.comm_node = nodes[est_best_midx]
        alpha_est = initial_complex_samples[est_best_idx]
        
        #Initialize KF state and covariance
        self.x_hat_k_k = np.array([self.comm_node.steered_angle,np.real(alpha_est),np.imag(alpha_est)])
        self.P_k_k = dcp(self.channel.Qu)
        
        #Main Algorithm Loop
        while nn < time_horizon:
            self.ekf_recursion()
            self.log_and_update_channel(nn)
            nn += 1
        return
    
    
    def ekf_recursion(self):
        """
        Standard Extended Kalman Filter recursion steps.

        """
        def wrap_angle(x): 
            """
            Wraps an angle to be within a specified range.
            
            This function adjusts the first element of the input array `x` so that it
            falls within the specified range [angle_min, angle_max], which we set as the
            codebook limits.
            
            Parameters
            ----------
            x : np.ndarray
                The input array where the first element represents the angle to be wrapped.
            Returns
            -------
            x : np.ndarray
                The input array with the first element adjusted to be within the specified range.
            """
            angle_max = self.cb_graph.min_max_angles[1]
            angle_min = self.cb_graph.min_max_angles[0]
            swath_width = np.abs(angle_max-angle_min)
            if np.isnan(x[0]):
                print('Angle Estimate is nan')
            x[0] = np.mod(x[0],np.pi)
            if x[0] > angle_max: x[0] = x[0] - swath_width
            elif x[0] < angle_min: x[0] = x[0] + swath_width
            return x
        
        
        inv = np.linalg.inv
        
        x_hat_k_km1 = dcp(self.x_hat_k_k)
        P_k_km1 = self.P_k_k + self.channel.Qu #This is what they put in the paper
        H = self.delh(x_hat_k_km1)
        Hh = H.T
        K = P_k_km1 @ Hh @ inv(H@P_k_km1@Hh + self.channel.Qv)
        y = self.sample(self.comm_node, mode = 'complex')
        y = np.array([np.real(y),np.imag(y)])
        h = self.h(x_hat_k_km1)
        if np.any(np.isnan(h)):
            print('h is nan')
        self.x_hat_k_k = x_hat_k_km1 + K @ (y - self.h(x_hat_k_km1))
        self.P_k_k = (np.eye(3)-K@H) @ P_k_km1
        
        # self.fix_phase(verbose = False)
        self.x_hat_k_k = wrap_angle(self.x_hat_k_k)
        self.update_beam_steering(verbose = False)

        return 
    
    def h(self,x,real = True):
        """
        Observation model estimate
        
        """
        M = self.cb_graph.M
        pi = np.pi
        Phi_D = np.cos(x[0]) - np.cos(self.comm_node.steered_angle) + 1e-14
        def e(x): return np.exp(1j * pi * x)
        
        # TODO: Verify the magnitude is correct, this was M^2 earlier, but we also had AoD and AoA
        h = (x[1] + 1j*x[2])/M * (1-e(-M*Phi_D))/(1-e(-Phi_D))
        if real: h = np.hstack([np.real(h),np.imag(h)])
        return h

    def delh(self,x,real = True,hermitian = False):
        """
        Linearization for observation model estimate
        """
        M = self.channel.M
        pi = np.pi
        Phi_D = np.cos(x[0]) - np.cos(self.comm_node.steered_angle) + 1e-14
        a_hat = x[1] + 1j*x[2]
        def e(x): return np.exp(1j * pi * x)
        partialaod= a_hat/M * np.sin(x[0])*((-1j*M*pi*e(-M*Phi_D))*(1-e(-Phi_D))-(1-e(-M*Phi_D))*(-1j*pi*e(-Phi_D)))/(1-e(-Phi_D))**2
        partialar = 1 /M * (1-e(-M*Phi_D))/(1-e(-Phi_D))
        partialai = 1j/M * (1-e(-M*Phi_D))/(1-e(-Phi_D))
        delh = np.array([partialaod,partialar,partialai])
        if hermitian: delh = np.conj(delh)
        if real: delh = np.vstack([np.real(delh),np.imag(delh)])
        if hermitian: delh = np.transpose(delh)
        return delh
    
    def update_beam_steering(self,verbose = False):
        """
        If the angle a half beamwidth away in either direction, choose the closest beamforming vector.
        """
        #Check Beam Switch
        thresh = self.cb_graph.beamwidths[-1]/2
        nodes = self.cb_graph.nodes
        if self.x_hat_k_k[0] - self.comm_node.steered_angle > thresh: 
            self.comm_node = nodes[self.comm_node.post_sibling]
            if verbose: print('switch aod +')
        elif self.x_hat_k_k[0] - self.comm_node.steered_angle < -thresh: 
            self.comm_node = nodes[self.comm_node.prior_sibling]
            if verbose: print('switch aod -')
        return 
    
    def log_and_update_channel(self,nn):
        """
        Description
        -----------
        Wrapper function to perform operations for updating the logs and channel fluctuations
        
        """
        
        if self.comm_node == None:
            self.log_data['relative_spectral_efficiency'].append(0.0)
            self.log_data['path'].append(np.nan)
        else:
            # self.set_best()
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
            self.log_data['path'].append(self.comm_node.midx)
            self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
            self.set_best()

        
class PF(AlgorithmTemplate):
    """
    Description
    -----------
    Uses the Particle Filter (EKF) from [1] to select beamforming vectors based on the nearest pointing angle, and brodens the beam based on the number of antenna elements required.
    
    We obtain our initial state estimate by using an exhaustive search with the most narrow beams.
    
    Unlike the original implementation in [1], which turns elements on and off to narrow or broaden the beam, we choose beamforming vectors from the ternary beamforming codebook based on the number of elements in (15) in [1], that possesses the similar beamwidth.  For example, consider a ternary hiearchical codebook with beamwidths at each level: [24, 8, 2.67, 0.89] (in degrees), at time step n, the method in [1] determines 128 elments, which corresponds to roughly to the .89 degree beamwidth in our array of beamwidths.   After observation at time step n+1, the algorithm determines 32 elements, which 2/32 is approximately a 3.58 degree beamwidth.  Because it is wider than 2.67, we broaden the beam to the level with 8 degrees with the beamforming vector whose pattern points in the direction of the current angle estimate and broaden it.
    
    
     [1] H. Chung, J. Kang, H. Kim, Y. M. Park and S. Kim, "Adaptive Beamwidth Control for mmWave Beam Tracking," in IEEE Communications Letters, vol. 25, no. 1, pp. 137-141, Jan. 2021, doi: 10.1109/LCOMM.2020.3022877.

    Attributes
    ----------
    
    S : int
        number of particles
    comm_node : object
        Node from codebook object currently being used to communicate
        
    
    Methods
    -------
    
    """
    def __init__(self, params):
        """
        Initializes a simulation using the EKF framework in [1] algorithm with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph': object
                The codebook graph associated with the simulation.
            - 'channel': object
                The communication channel used in the simulation.
            - 'num_particles' : int
                Number of particles used in the particle filter

        """
        super().__init__(params)
        self.S = params['num_particles']
        self.comm_node = None #initialize communication node as none to start
        
        
    def run_alg(self,time_horizon):
        nodes = self.cb_graph.nodes
        
        #Take initial samples and update time index and pad RSE with zero for intial alignment period.
        # initial_complex_samples = np.array([self.sample(nodes[midx],mode = 'complex') for midx in self.cb_graph.level_midxs[-1]])
        nn = 1
        initial_complex_samples = []
        for midx in self.cb_graph.level_midxs[-1]: 
            initial_complex_samples.append(self.sample(nodes[midx],mode = 'complex'))
            self.log_and_update_channel(nn)
            nn +=1
            
        
        #Choose beamforming vector that yields the highest RSS, choose this as comm_node and steered_angle as initial state estimate
        #Unsure how to initialize alpha, I just used the complex value received, but it might be easier just to pass the value from channel.
        est_best_idx = np.argmax(np.abs(initial_complex_samples)**2)
        est_best_midx = self.cb_graph.level_midxs[-1][est_best_idx]
        self.comm_node = nodes[est_best_midx]
        alpha_est = initial_complex_samples[est_best_idx]
        
        #Initial Estimate
        self.xk_hat = np.array([self.comm_node.steered_angle,0.0,np.real(alpha_est),np.imag(alpha_est)])
        
        #Initialize particles with estimate
        self.particles = {}
        initial_weight = 1.0 / self.S
        for ss in np.arange(self.S):
            self.particles[ss] = PF.Particle(ss,initial_weight,dcp(self.xk_hat))
            
        while nn < time_horizon:
            
            
            #Generate Particles as predictions
            self.get_predictions()
            # for ss in np.arange(self.S):
            #     u = np.random.multivariate_normal(np.zeros(3),self.channel.Qu)
            #     self.particles[ss].state = self.channel.F@self.particles[ss].state + self.channel.G@u
                
            #Get Observation
            # zs = self.zk()
            z = self.sample(nodes[midx],mode = 'complex')
            
            #Update weights
            self.update_weights(z)
            
            #Compute estimate and adjust beamformer
            self.xk_hat = self.get_estimate()
            if self.comm_node.steered_angle - self.xk_hat[0] > self.comm_node.beamwidth/2:
                self.comm_node = nodes[self.comm_node.prior_sibling]
            elif self.xk_hat[0] - self.comm_node.steered_angle > self.comm_node.beamwidth/2:
                self.comm_node = nodes[self.comm_node.post_sibling]
                
            #Calculate number of active beamforming elements, and adjust beamwidth correspondingly
            # self.Mk = np.max([np.min([np.floor(2.8/np.pi/self.zeta_k()/np.sin(self.xk_hat[2])),self.params['M_0']]),2])
            Mk = np.max([np.min([np.floor(2.8/np.pi/self.zeta_k()/np.sin(self.comm_node.steered_angle)),self.cb_graph.M]),2])
            notional_beamwidth = 2/Mk
            if notional_beamwidth > self.cb_graph.beamwidths[self.comm_node.h] and self.comm_node.h > 0:
                self.comm_node = nodes[self.comm_node.zoom_out_midx]
            elif self.comm_node.h < self.cb_graph.H-1:
                if notional_beamwidth <= self.cb_graph.beamwidths[self.comm_node.h+1]:
                    self.comm_node = nodes[self.comm_node.zoom_in_midxs[1]]
                
            
            
            #Resample
            self.resample()
            
            #State Fluctuation
            # self.update_state()
            self.log_and_update_channel(nn)
            nn += 1
            
    def get_predictions(self):
        
        def wrap_angle(x): 
            """
            Wraps an angle to be within a specified range.
            
            This function adjusts the first element of the input array `x` so that it
            falls within the specified range [angle_min, angle_max], which we set as the
            codebook limits.
            
            Parameters
            ----------
            x : np.ndarray
                The input array where the first element represents the angle to be wrapped.
            Returns
            -------
            x : np.ndarray
                The input array with the first element adjusted to be within the specified range.
            """
            angle_max = self.cb_graph.min_max_angles[1]
            angle_min = self.cb_graph.min_max_angles[0]
            swath_width = np.abs(angle_max-angle_min)
            if np.isnan(x[0]):
                print('Angle Estimate is nan')
            x[0] = np.mod(x[0],np.pi)
            if x[0] > angle_max: x[0] = x[0] - swath_width
            elif x[0] < angle_min: x[0] = x[0] + swath_width
            return x
        
        for ss in np.arange(self.S):
            u = np.random.multivariate_normal(np.zeros(3),self.channel.Qu)
            self.particles[ss].state = self.channel.F@self.particles[ss].state + self.channel.G@u
            self.particles[ss].state = wrap_angle(self.particles[ss].state)
            
    def update_weights(self,z):
        pdfs = []
        old_weights = dcp([self.particles[ss].weight for ss in range(self.S)])
        mus = []
        for ss in np.arange(self.S):
            x = self.particles[ss].state
            # mu_x_s = np.conj(a(self.xk_hat[2],self.Mk)) @ self.hk(self.particles[ss].state)
            mu_x_s = np.conj(self.comm_node.f) @ ((x[2] + 1j * x[3]) * avec(x[0],self.cb_graph.M))
            mus.append(mu_x_s)
            pdf_d = 1.0/ (np.pi * self.channel.sigma_v**2) * np.exp(-np.sum(np.abs(z-mu_x_s)**2)/self.channel.sigma_v**2)
            pdfs.append(pdf_d + 1e-16)
        mus = np.array(mus)
        # weights = [np.array(pdfs[ss])/np.sum(pdfs) for ss in np.arange(self.params['S'])]
        weights_unnorm = np.array([pdf * old_weight for pdf,old_weight in zip(pdfs,old_weights)])
        weights = weights_unnorm/np.sum(weights_unnorm)
        
        #Assign/Normalize Weights
        for ss in np.arange(self.S): self.particles[ss].weight = weights[ss]
        
    def get_estimate(self): 
        
        x_hat_ = np.zeros(len(self.particles[0].state))
        for ss in np.arange(self.S):
            x_hat_ += self.particles[ss].weight * self.particles[ss].state
        # x_hat_[2] = np.mod(x_hat_[2],np.pi)
        return x_hat_ 
    
    def zeta_k(self):
        zeta = 0.0
        for ss in np.arange(self.S):
            # zeta += self.particles[ss].weight * (self.particles[ss].state[2] - self.xk_hat[2])**2
            zeta += self.particles[ss].weight * (self.particles[ss].state[0] - self.comm_node.steered_angle)**2
        return np.sqrt(zeta)
    
    def resample(self):
        """
        Resampling routine from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        """
        weights = [self.particles[ss].weight for ss in range(self.S)]
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum,np.random.uniform(0,1,self.S))
        
        new_states = [dcp(self.particles[ss].state) for ss in indexes]
        for ss in range(self.S): 
            self.particles[ss].state = dcp(new_states[ss])
            self.particles[ss].weight = 1/self.S
    
    def log_and_update_channel(self,nn):
        """
        Description
        -----------
        Wrapper function to perform operations for updating the logs and channel fluctuations
        
        """
        
        if self.comm_node == None:
            self.log_data['relative_spectral_efficiency'].append(0.0)
            self.log_data['path'].append(np.nan)
        else:
            # self.set_best()
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
            self.log_data['path'].append(self.comm_node.midx)
            self.channel.fluctuation(nn,(self.cb_graph.min_max_angles[0],self.cb_graph.min_max_angles[1]))
            self.set_best()
            
    class Particle:
        def __init__(self,index,initial_weight,initial_state):
            # self.params = params
            self.index = index
            self.weight = initial_weight
            self.state = initial_state



if __name__ == '__main__':
    main()