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
import pickle
import os
import warnings
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.special import erf
import scipy.io
from time import time
from util import * 

class BasicChannel:
    """
    A base class to represent a communication channel.  Transmit power assumed to be unitary.

    Attributes
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TrinaryPointedHierarchicalCodebook.  See description of class types in mlcomm.codebooks.
    snr : float
        Signal-to-noise ratio.
    sigma_v : float
        Noise standard deviation, calculated from SNR.
    ht : numpy array of complex float
        Channel reponse placeholder
    seed : int
        Seed for random number generation.
        
    Methods
    -------
    array_reponse
        Generates the noisy channel response with respect to specified snr.
    """

    def __init__(self, params):
        """
        Initializes the Channel with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph' : object 
                Type instance of BinaryHierarchicalCodebook or TrinaryPointedHierarchicalCodebook.  See description of class types in mlcomm.codebooks.
            - 'snr': float
                Signal-to-noise ratio.
            - 'seed': int
                Seed for random number generation.
        """
        #self.cb_graph = params['cb_graph']
        self.snr = params['snr']
        self.sigma_v = np.sqrt(10**(-self.snr/10))
        self.ht = [0.0 + 0.0j]#placeholder channel response
        self.seed = params['seed']
        #print('Initializing random number generator seed within BasicChannel')
        #np.random.seed(seed = self.seed)

    def array_response(self,transmit_power_dbm = 0,with_noise = True):
        """
        
        Notes
        -----
        The arg parameter 'transmit_power_dbm' is used only for debugging in this particular class object.
        
        It is also a placeholder so that algorithms may use either channel model.
        
        """
        tx_power_mwatts = 10**(transmit_power_dbm/10)
        if with_noise:
            return np.sqrt(tx_power_mwatts) * self.ht + self.sigma_v * randcn(len(self.ht))
        else: 
            return np.sqrt(tx_power_mwatts) * self.ht

class Channel:
    """
    A base class to represent a communication channel.  scenario dependent parameters dictate transmit power.  Designate noise variance by the attribute
    sigma_v.

    Attributes
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TrinaryPointedHierarchicalCodebook.  See description of class types in mlcomm.codebooks.
    sigma_v : float
        Noise standard deviation.
    ht : numpy array of complex float
        Channel reponse placeholder
    seed : int
        Seed for random number generation.
        
    Methods
    -------
    array_reponse
        Generates the noisy channel response with respect to specified snr.
    """

    def __init__(self, params):
        """
        Initializes the Channel with the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'cb_graph' : object 
                Type instance of BinaryHierarchicalCodebook or TrinaryPointedHierarchicalCodebook.  See description of class types in mlcomm.codebooks.
            - 'noise_variance': float
                Signal-to-noise ratio.
            - 'seed': int
                Seed for random number generation.
        """
        #self.cb_graph = params['cb_graph']
        self.sigma_v = np.sqrt(params['noise_variance'])
        self.ht = [0.0 + 0.0j]#placeholder channel response
        self.seed = params['seed']
        # print(f'Initializing random number generator seed within Channel: {self.seed}.')
        # np.random.seed(seed = self.seed)
        
    def array_response(self,transmit_power_dbm = 0, with_noise = True):
        tx_power_mwatts = 10**(transmit_power_dbm/10)
        out = np.sqrt(tx_power_mwatts) * self.ht
        if with_noise:
            out = out + self.sigma_v * randcn(len(self.ht))
        return out
    
class AWGN(BasicChannel):
    """
    Description
    -----------
    Additive White Gaussian Noise Channel
    
    Attributes
    ----------
    M : int
        Number of elements in the array.
    angle : float
        Angle of arrival/departure in radians, converted from degrees.  Main path.
    ht : numpy ndarray of complex float
        Channel response
    """
    
    def __init__(self,params):
        """
        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'num_elements': int
                Number of elements in the array.
            - 'angle_degs': float
                Angle of arrival/departure in degrees.
        """
        super().__init__(params)
        self.M = params['num_elements']
        self.angle = params['angle_degs'] * np.pi/180
        self.alphas = [1.0]
        self.L = 1
        self.ht = avec(self.angle,self.M)
    
    def fluctuation(self,*args,**kwargs):
        #Do nothing, yay!
        return
    
class RicianAR1(BasicChannel):
    """
    Description
    -----------
    Class object governing and containing parameters for a RicianAR1 channel model for a uniform linear array (ULA) along the x-axis.
    Parameter scenarios are set as in [1] [2].
    
    [1] Chiu, Sung-En, Nancy Ronquillo, and Tara Javidi. "Active learning and CSI acquisition for mmWave initial alignment." IEEE Journal on Selected Areas in Communications 37.11 (2019): 2474-2489.
    [2] Blinn, Nathan, Jana Boerger, and Matthieu Bloch. "mmWave Beam Steering with Hierarchical Optimal Sampling for Unimodal Bandits." ICC 2021-IEEE International Conference on Communications. IEEE, 2021.
    
    Attributes
    ----------
    M : int
        Number of elements in the array.
    angle : float
        Angle of arrival/departure in radians, converted from degrees.  Main path.
    mu : float
        First fading parameter.
    Kr : float
        Second fading parameter.
    g : float
        Correlation parameter.
    L : int
        Number of signal paths.
    snr : float
        Signal-to-noise ratio.
    angles: numpy ndarray of floats
        angle of arrival/departure of all L paths.
    alphas : numpy ndarray of complex float
        Dynamically updated fading coefficients for each path.
    ht : numpy ndarray of complex float
        Channel response with fading
        
    Methods
    -------
    channel_fluctuation(self):
        Updates the channel state to simulate fluctuations in dynamic fading.
        
        
    Notes
    -----
    Multi-path effects tend to take place around the maain path, we choose this value
    to be .35 radians (~20 degrees).  More detail available in 
    
    - Rappaport, Theodore S., et al. "Millimeter wave mobile communications for 5G cellular: It will work!." IEEE access 1 (2013): 335-349.
    - Akdeniz, Mustafa Riza, et al. "Millimeter wave channel modeling and cellular capacity evaluation." IEEE journal on selected areas in communications 32.6 (2014): 1164-1179.
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'num_elements': int
                Number of elements in the array.
            - 'angle_degs': float
                Angle of arrival/departure in degrees.
            - 'fading_1': float
                First fading parameter.
            - 'fading_2': float
                Second fading parameter.
            - 'correlation': float
                Correlation parameter.
            - 'num_paths': int
                Number of signal paths.
            - 'snr': float
                Signal-to-noise ratio.
            - 'seed': int
                Seed for random number generation.
            Recommended: 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451
        """
        super().__init__(params)
        self.M = params['num_elements']
        self.angle = params['angle_degs'] * np.pi/180
        self.mu = params['fading_1']            
        self.Kr = params['fading_2']
        self.g = params['correlation']
        self.L = params['num_paths']
        self.snr = params['snr']
        self.seed = params['seed']

        self.angles = np.concatenate([[self.angle], np.random.uniform(self.angle-.35,self.angle+.35,self.L-1)]) 
        self.alphas = np.concatenate([[1.],.1*np.ones(self.L-1)])* (np.sqrt(self.Kr/(1+self.Kr))*self.mu + np.sqrt(1/(1+self.Kr))*randcn(self.L))      #Initialize alpha (fading param)
        self.ht = np.sum([self.alphas[ii] * avec(self.angles[ii],self.M) for ii in np.arange(self.L)],axis = 0)
        
        
    def fluctuation(self,nn,*args):
        self.alphas = np.sqrt(self.Kr/(1+self.Kr))*self.mu \
                + (self.alphas - np.sqrt(self.Kr/(1+self.Kr))*self.mu) * np.sqrt(1-self.g)\
                            + randcn(self.L) * np.sqrt(self.g/(1+self.Kr))
        self.ht = np.sum([self.alphas[ii] * avec(self.angles[ii],self.M) for ii in np.arange(self.L)],axis = 0)
        
class NYU_preset(Channel):
    """
    Description
    ------------
    Preset channel responses with spatial consistency used in DBZ simulations. Generated from NYU Sim 4.0:
        
    https://wireless.engineering.nyu.edu/nyusim/
    
    Each scenario Rural Macro ('RMa'), Urban Macro ('UMa'), and Urban Micro ('UMi') has two profiles, 'Linear' and 'Hexagon', each with 100 different sets of 600 time "snapshots" or time steps by our nomenclature.
    
    *IMPORTANT*: The folder 'preset_nyu_sim' must be in the same directory as channels.py.  This >8GB folder with all preset channel conditions is available from the author at nblinn6@gatech.edu 
    
    Alternatively, the parameters for how we generated this are available in our work and may be generated by the user.
    
    Parameters
    ----------
    params : dict
        A dictionary of parameters used to initialize the channel.
        
        'profile' : str
            The profile to be used in the simulation. Valid options are 'Linear' and 'Hexagon'.
            
        'scenario' : str
            The scenario to be used in the simulation. Valid options are 'RMa', 'UMa', and 'UMi'.
        
        'num_elements' : int
            The number of elements in the MIMO array.
        
        'set_number' : int
            The specific set number within the profile and scenario to be used. Must be between 1 and 100.
    
    Attributes
    ----------
    M : int
        Number of elements in the MIMO array.
    
    set_number : int
        The selected set number within the chosen profile and scenario.
    
    profile : str
        The selected profile ('Linear' or 'Hexagon').
    
    scenario : str
        The selected scenario ('RMa', 'UMa', or 'UMi').
    
    Hs : list of np.ndarray
        A list of normalized channel matrices for each time snapshot.
    
    ht : np.ndarray
        The channel matrix at the current time step.
    
    Methods
    -------
    channel_fluctuation(nn):
        Updates the channel matrix to the nth time snapshot.
        If nn is out of bounds, prints an error message.
    """
    def __init__(self, params):
        super().__init__(params)
        self.M = params['num_elements']
        self.set_number = params['set_number']
        self.profile = params['profile']
        self.scenario = params['scenario']
        
        assert params['scenario'] == 'RMa' or params['scenario'] == 'UMa' or params['scenario'] == 'UMi', "Incorrect Enumerated Choice for 'scenario': Please choose 'RMa', 'UMa', 'UMi'"
        assert params['profile'] == 'Linear' or params['profile'] == 'Hexagon', "Incorrect Enumerated Choice for 'profile': Please choose 'Linear' or 'Hexagon'."
        assert params['set_number'] <= 100, "Please choose a 'set_number' between 1 and 100"
        struct_key = 'CIR_MIMO_EVO'  
        abspath = os.path.dirname(os.path.abspath(__file__))
        file_path = abspath + f"/preset_nyu_sim/Output_{self.scenario}_{self.profile}_{self.set_number}/CIR_MIMO_EVO.mat"
        backup_file_path = abspath + f"/../../../preset_nyu_sim/Output_{self.scenario}_{self.profile}_{self.set_number}/CIR_MIMO_EVO.mat"
        if os.path.exists(file_path):
            print('Loading channel presets.')
        elif os.path.exists(backup_file_path):
            file_path = backup_file_path
            print('Channel found in upper directory.')
        else:
            print('Channel presets not found.')
            return
        
        attr_names = []
        for ii in np.arange(600):
            attr_names.append(f'Snapshot{ii+1}')
        
        mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        data = mat[struct_key]

        Hs_unnorm = []
        aods = []
        for attr_name in attr_names:
            Hs_unnorm.append(getattr(getattr(data,attr_name),'H_ensemble'))
            aods.append(getattr(getattr(data,attr_name),'AODs'))
        
        self.Hs = []
        for H in Hs_unnorm:
            fac =  np.linalg.norm(H)/ np.sqrt(self.M)
            self.Hs.append(H/fac)
        self.ht = self.Hs[0]
        
    def channel_fluctuation(self, nn,*args):
        """
        Updates the channel matrix to the nth time snapshot.

        Parameters
        ----------
        nn : int
            The index of the time snapshot to switch to.
        
        Returns
        -------
        None
        
        Raises
        ------
        IndexError
            If nn is out of the valid range of snapshots.
        """
        try:
            self.ht = self.Hs[nn]
        except IndexError:
            print(f'Error: Time step too large, {nn} is not valid')
            return

class NYU1(Channel):
    """
    Description
    -----------
    Channel model based on Table 1 in [1] for a uniform linear array (ULA) along the x-axis. For our purposes, we consider the outage probability to be zero. 
    We maintain the same path angles along with the slow time fading stochastic value governed by sigma_dB. 
    When initialized, a value is selected and held. We also assume zero Doppler.

    [1] Akdeniz, Mustafa Riza, et al. "Millimeter wave channel modeling and cellular capacity evaluation." IEEE journal on selected areas in communications 32.6 (2014): 1164-1179.

    Attributes
    ----------
        M : int
            Number of elements in the array.
        angle : float
            Main path angle in degrees.
        environment : str
            Line-of-sight condition ('LOS' or 'NLOS').
        scenario : str
            Scenario ('Rural' or 'Urban').
        center_frequency_ghz : float
            Center frequency in GHz (valid choices are 28 and 72).
        propagation_distance : float
            Propagation distance in meters.
        sigma_v : float
            Noise standard deviation.
        wavelength : float
            Wavelength corresponding to the center frequency.
        alpha_dB : float
            Path loss coefficient in dB.
        beta : float
            Path loss exponent.
        sigma_dB : float
            Standard deviation of shadow fading in dB.
        lam : float
            Mean number of clusters (Poisson distributed).
        r_tau : float
            Decay factor for cluster powers.
        zeta : float
            Shadow fading factor for clusters.
        cluster_path_rms : float
            RMS of cluster path angles.
        num_paths_per_cluster : int
            Number of paths per cluster.
        slow_time_fade_factor : float
            Slow time fading factor.
        K : int
            Number of clusters.
        gamma_p : list
            Cluster powers.
        gamma : np.ndarray
            Normalized cluster powers.
        cluster_angles : np.ndarray
            Cluster angles.
        cluster_path_angles : list
            Path angles within each cluster.
        PL_dB : float
            Path loss in dB.
        ht : np.ndarray
            Channel impulse response.
        gs : np.ndarray
            Scaling factors for the clusters.

    Methods
    --------
        __init__(self, params)
            Initializes the channel model with the given parameters.
            
    Notes
    -----
    We reorganize the paths such that the strongest path is always the first indexed path.
    
    Our model has some minor additions from the paper, we incorporate the notion of 'Rural' and 'Urban' modes to distinguish between
    scenarios where there is either one dominant path with at most one cluster or the specified number of paths/clusters as denoted in [1].
    
    We also make the assumption that the RF front end of the receiver has some type of automatic gain control (AGC) in which the received signal 
    is normalized.
    """
    
    def __init__(self, params):
        """
        Initializes the channel model with the given parameters.

        Parameters
        ----------
            params : dict
                Dictionary containing the following keys:
                    'num_elements' : int
                        Number of elements in the array.
                    'angle_degs' : float
                        Main path angle in degrees.
                    'environment' : str
                        Line-of-sight condition ('LOS' or 'NLOS').
                    'scenario' : str
                        Environmental scenario ('Rural' or 'Urban').
                    'center_frequency_ghz' : float
                        Center frequency in GHz (valid choices are 28 and 72).
                    'propagation_distance' : float
                        Propagation distance in meters.
                    'noise_variance' : float
                        Noise variance of the channel.

        """
        super().__init__(params)
        self.M = params['num_elements']
        self.angle = params['angle_degs'] #Choose main path explicitly.
        assert params['environment'] == 'LOS' or params['environment'] == 'NLOS', "Incorrect Enumerated Choice: Please choose 'LOS' or 'NLOS'." 
        self.environment = params['environment'] #Valid choices are 'LOS' and 'NLOS'
        assert params['scenario'] == 'Rural' or params['scenario'] == 'Urban', "Incorrect Enumerated Choice: Please choose 'Rural' or 'Urban'"
        self.scenario = params['scenario'] #Valid choice are 'Rural' and 'Urban'
        self.center_frequency_ghz = params['center_frequency_ghz']#Valid choices are 28 and 72
        self.propagation_distance = params['propagation_distance'] #meters
        
        self.wavelength = 3e8/(self.center_frequency_ghz * 1e9)
        
        if self.center_frequency_ghz == 28:
            if self.environment == 'LOS':
                self.alpha_dB = 61.4
                self.beta = 2.0
                self.sigma_dB = 5.8
            elif self.environment == 'NLOS':
                self.alpha_dB = 72.0
                self.beta = 2.92
                self.sigma_dB = 8.7
            self.lam = 1.8
            self.r_tau = 2.8
            self.zeta = 4.0
            self.cluster_path_rms = 10.2 * np.pi/180
        elif self.center_frequency_ghz == 72:
            if self.environment == 'LOS':
                self.alpha_dB = 69.8
                self.beta = 2
                self.sigma_dB = 5.8
            elif self.environment == 'NLOS':
                self.alpha_dB = 86.6 #or 82.7
                self.beta = 2.45 # or 2.69
                self.sigma_dB = 8.0 # or 7.7
            self.lam = 1.9
            self.r_tau = 3.0
            self.zeta =  4.0
            self.cluster_path_rms = 10.5 * np.pi/180
        
        if self.scenario == 'Rural':
            self.num_paths_per_cluster = 1
            self.cluster_path_rms = 0.0
        elif self.scenario == 'Urban':
            self.num_paths_per_cluster = 20
            self.cluster_path_rms = 10.5 * np.pi/180
            
        self.slow_time_fade_factor = self.sigma_dB * np.random.randn()
        # self.slow_time_fade_factor = 0.0
        
        self.K = np.max([np.random.poisson(self.lam),1]) #Number of Clusters
        self.gamma_p = []
        for k in np.arange(self.K):
            self.gamma_p.append(np.random.uniform(0,1)**(self.r_tau-1)*10**(0.1*self.zeta*np.random.randn(1)[0]))
        self.gamma = []
        for val in self.gamma_p:
            self.gamma.append(val/np.sum(self.gamma_p))
        self.gamma = np.array(self.gamma)
        self.cluster_angles = [self.angle]
        if self.K > 1: self.cluster_angles.extend(np.random.uniform(0,2*np.pi,self.K-1))
        self.cluster_angles = np.array(self.cluster_angles)
        self.cluster_path_angles = []
        L = self.num_paths_per_cluster
        for kk in np.arange(self.K):
            for angle in self.cluster_angles:
                self.cluster_path_angles.append(angle + self.cluster_path_rms * np.random.randn(L)) # L = 20 paths according to paragraph preceeding (9)
            
        self.PL_dB = self.alpha_dB + 10*self.beta * np.log10(self.propagation_distance) + self.slow_time_fade_factor
        self.ht = np.zeros(self.M) + 0.0 * 1j
        self.gs = np.sort(1/np.sqrt(L) * np.sqrt(self.gamma * 10**(-0.1 * self.PL_dB)) * np.random.randn(self.K)[0])
        for kk in np.arange(self.K):
            #g = 1/np.sqrt(L) * np.sqrt(self.gamma[kk] * 10**(-0.1 * self.PL_dB)) * np.random.randn(1)[0]
            g = self.gs[kk]
            for angle in self.cluster_path_angles[kk]:
                self.ht = self.ht +  g * avec(angle,self.M)
        
        #Automatic gain control on the front end
        self.ht = self.ht/np.linalg.norm(self.ht)
    
    def fluctuation(self,*args):
        #Do nothing, yay!
        return
    
class NYU2(Channel):
    """
    
    Description
    ------------
    
    Channel model based on the publications from NYU Wireless, many functions are derived directly from the NYU source code (https://wireless.engineering.nyu.edu/nyusim/)
    
    [1] Introduces the model and how to construct it, [2] introduces NYU Sim, and [3] adds the spatial consistency effects on top of [1] into the utility in [2] with additional data/effects from [4,5].
    
    Generates a grid for correlated spatial variables (encompassing both LOS and NLOS environments) that is of size 400 x 400 meters.  The array is placed at the center
    of the 400 meters and is assumed at (x,y) = (0,0).
    
    [1] Samimi, Mathew K., and Theodore S. Rappaport. "3-D millimeter-wave statistical channel model for 5G wireless system design." IEEE Transactions on Microwave Theory and Techniques 64.7 (2016): 2207-2225.
    
    [2] Sun, Shu, George R. MacCartney, and Theodore S. Rappaport. "A novel millimeter-wave channel simulator and applications for 5G wireless communications." 2017 IEEE international conference on communications (ICC). IEEE, 2017.
    
    [3] Ju, Shihao, and Theodore S. Rappaport. "Simulating motion-incorporating spatial consistency into NYUSIM channel model." 2018 IEEE 88th vehicular technology conference (VTC-Fall). IEEE, 2018.
    
    [4] S. Ju, O. Kanhere, Y. Xing and T. S. Rappaport, “A Millimeter-Wave Channel Simulator NYUSIM with Spatial Consistency and Human Blockage” in IEEE 2019 Global Communications Conference, pp. 1–7, Dec. 2019.
    
    [5] S. Sun et al., "Investigation of Prediction Accuracy, Sensitivity, and Parameter Stability of Large-Scale Propagation Path Loss Models for 5G Wireless Communications," in IEEE Transactions on Vehicular Technology, vol. 65, no. 5, pp. 2843-2860, May 2016.
    
    
    The channel has granularity of squares 1x1 meter for characterizing channel fading effects in a 400mx400m grid.  When the Mobile Entity (ME) translates from one grid square to another, 
    
    Many functions are directly translated from the NYU Sim MATLAB Source Code, including ``get_los_map``, ``get_sf_map``, ``get_delay_info``.
    
    Attributes
    -----------
    
    
    Methods
    --------
    
        set_environment
            Sets the object instance environment variables, i.e., environment 'LOS', 'NLOS', with specified scenario according to the ME position triple passed to the method as an arg.
            
        fluctuation
            Adjusts the channel based on ME motion, if the ME changes environment (i.e., 'NLOS' to 'LOS'), the channel reinitializes path variables.
    
    Notes
    ------
    This implementation does not consider polarization, human blockage, or indoor-outdoor penetration loss factors
    """
    
    
    
    def __init__(self, params):
        """
        Initializes the channel model with the given parameters.

        Parameters
        ----------
            params : dict
                Dictionary containing the following keys:
                    'num_elements' : int
                        Number of elements in the array.
                    'scenario' : str
                        Environmental scenario ('RMa', 'UMa', 'UMi').
                    'center_frequency_ghz' : float
                        Center frequency in GHz (valid choices are 28 and 72).
                    'initial_me_position' : triple of floats (x,y,z)
                        Initial position of the mobile entity. Ignored in NLOS environment, chosen from distribution d~U(60,200) when NLOS
                    'initial_me_velocity' : triple of floats (x_dot, y_dot, z_dot)
                        Initial velocity of mobile entity.
                    'sigma_u_m_s2' : float
                        Variance governing Gaussian distributed acceleration for x and y 
                    'noise_variance' : float
                        Noise variance of the channel.
                    'map_generation_mode' : str
                        Valid choices are "new" and "load".  If "load".  In "new" mode, saves the sf_maps and los_map as with the specified "map_index".  In "load" mode, loads the sf_maps and los_map with corresponding "map_index".  Saves/loads from specified directory by dict value "maps_dir"
                    'map_index' : int
                        Indicates the index used to load/save 
                    'maps_dir' : str
                        Path specifying where to save/load sf_maps and los_map
        """
        super().__init__(params)
        
        self.M = params['num_elements']
        assert params['scenario'] == 'RMa' or params['scenario'] == 'UMa' or params['scenario'] == 'UMi', "Incorrect Enumerated Choice: Please choose 'RMa', 'UMa', 'UMi'"
        self.scenario = params['scenario'] #Valid choice are 'Rural' and 'Urban'
        assert params['center_frequency_ghz'] >= 28 and params['center_frequency_ghz'] <= 72, "Incorrect value for range of 'center_frqeuency_ghz', please enter value between 28 and 72 (may equal 28 or 72)."
        self.center_frequency_ghz = params['center_frequency_ghz']#Valid choices are 28 and 72
        
        self.initial_me_position = params['initial_me_position'] 
        self.initial_me_velocity = params['initial_me_velocity']
        self.d_co = params['correlation_distance']
        self.sigma_u = params['sigma_u_m_s2'] 
        
        self.h_BS = 20
        self.h_MS = self.initial_me_position[2]
        
        self.me_position = dcp(self.initial_me_position)
        self.me_velocity = dcp(self.initial_me_velocity)
        
        # Environment setting (pressure, humidity, foliage loss), precalculated using the internal NYU Sim function 'mpm93_forNYU' with the following parameters
        p,u,temp,RR,Fol,dFol,folAtt = 1013.2, 50, 20, 150, 'No', 0, 0.4
        atmos_idx = int(self.center_frequency_ghz - 28)

        # Array of atmospheric data, index 0 corresponds to 28 GHz, and the last element corresponds to 72 GHz. The values are indexed by a granularity of 1 GHz
        atmospheric_data = [
            0.0275696495821050, 0.0285223128615362, 0.0294907100235393, 0.0304738786913288, 
            0.0314711615494553, 0.0324821051200185, 0.0335063921991314, 0.0345437995000089, 
            0.0355941718454488, 0.0366574070527800, 0.0377334480258621, 0.0388222802061213, 
            0.0399239336901567, 0.0410384902849999, 0.0421660968285762, 0.0433069876179524, 
            0.0444615213996605, 0.0456302434466445, 0.0468139941094057, 0.0480141112548014, 
            0.0492328468819002, 0.0504743797624106, 0.0517481179557135, 0.0530795488960932, 
            0.0545344415627232, 0.0562706349842568, 0.0593623445682586, 0.0607712292391748, 
            0.0630861812700089, 0.0655605279727314, 0.0672754290475197, 0.0680387167444610, 
            0.0686000087996802, 0.0686144442408844, 0.0672487882074241, 0.0636936157778784, 
            0.0593854153534244, 0.0560543358076998, 0.0539190114727561, 0.0526854783490967, 
            0.0519516460966182, 0.0514481942077645, 0.0510459061873077, 0.0506917824772509, 
            0.0503652975384719]
        
        #Atmospheric loss factor in db/m
        self.atmospheric_loss_factor_db_m  = atmospheric_data[atmos_idx] 
        

        #Build grid for spatial fading and los/nlos condition
        half_grid_side_length = 200
        self.pos_grid = []
        index = 0
        for yy in np.arange(-half_grid_side_length,half_grid_side_length):
            for xx in np.arange(-half_grid_side_length,half_grid_side_length):
                self.pos_grid.append(NYU2.SpatiallyCoherentPosition({'index': index, 'x' : xx, 'y': yy, 'z' : self.h_MS}))
                index += 1
                
        # Generate LOS/NLOS map first, based on position, the set of parameters are used for that environment. 
        # The NYU Sim code generates the SF map first, but has a dependency on LOS/NLOS condition, see 'sigma' attribute.
        #I'm supposing their implementation was just to simplify the deployment and use where the UE moves only locally.  
        #See the description in the user manual on page 36.
        
        if params['map_generation_mode'] == 'new':
            t0 = time()
            print('Creating LOS map...')
            los_map = self.get_los_map(int(2*half_grid_side_length))
            pickle.dump(los_map, open(params["maps_dir"] + f'/los_map_{params["map_index"]}.pkl','wb'))
            print(f"Time required to generate LOS map: {time()-t0}")
            
            t0 = time()
            print('Creating SF maps...')
            #In order to build the SF map, I need to know sigma (fading paramter) which is dependent on LOS/NLOS.
            sf_los_map, sf_nlos_map = self.get_sf_map(2*half_grid_side_length)
            pickle.dump(sf_los_map, open(params["maps_dir"] + f'/sf_los_map_{params["map_index"]}.pkl','wb'))
            pickle.dump(sf_nlos_map, open(params["maps_dir"] + f'/sf_nlos_map_{params["map_index"]}.pkl','wb'))
            print(f"Time required to generate SF maps: {time()-t0}")
            
        elif params['map_generation_mode'] == 'load':
            los_map = pickle.load(open(params["maps_dir"] + f'/los_map_{params["map_index"]}.pkl','rb'))
            #print('Debugging with a LOS map of all ones!') 
            #los_map = np.ones([400,400]) #Debugging purposes only.
            sf_los_map = pickle.load(open(params["maps_dir"] + f'/sf_los_map_{params["map_index"]}.pkl','rb'))
            sf_nlos_map = pickle.load(open(params["maps_dir"] + f'/sf_nlos_map_{params["map_index"]}.pkl','rb'))
        
        self.los_map = los_map
        self.sf_los_map = sf_los_map
        self.sf_nlos_map = sf_nlos_map
        #Initiazlize grid of values indicating 'LOS' and 'NLOS' along with the value on the SF map value
        for pos in self.pos_grid:
            pos.los = los_map[pos.y+half_grid_side_length,pos.x+half_grid_side_length]
            if pos.los == 1:
                pos.sf = sf_los_map[pos.y,pos.x]
                pos.los = 'LOS'
            elif pos.los == 0:
                pos.sf = sf_nlos_map[pos.y,pos.x]
                pos.los = 'NLOS'
        
        #Initialize the environment
        self.prev_environment = 'not set' #Just generic str so that it fails the logic test in fluctuation method and initializes
        self.fluctuation(mode = 'initialization')
        

            
    def set_environment(self):
        """
        Description
        ------------
        Based on the current location of the ME, stored as attribute triple, 'me_position', 
        the position is cross-referenced with being LOS or NLOS and then paramters corresponding 
        to the particular environment are assigned to change the state of the channel.
        
        The position is changed over time by calling the 'update_me' method, which applies a 
        White Gaussian Accleration transition.
        
        
        Parameters
        ----------

        Returns
        -------
        None.

        """
        pos_idx = self.get_grid_index(self.me_position[0],self.me_position[1],self.me_position[2]) #Index of spatially coherent grid based on ME initial position
        
        # Parameter setting based on the fields in Table III and IV in [1]
        if self.pos_grid[pos_idx].los == 'LOS':
            self.environment = 'LOS'
            
            # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
            if self.center_frequency_ghz == 28:
                self.n_bar, self.sigma = 2.1, 3.6
            elif self.center_frequency_ghz == 72:
                self.n_bar, self.sigma = 2.0, 5.2 
            else:
                print("Warning: Value not specified as exactly 28 or 73 for 'center_frequency_ghz', Table III does not specify explicit values, taking intermediary for n_bar and sigma.")
                self.n_bar, self.sigma = 2.05, 4.4  
                
            self.mu_AOD, self.mu_AOA = 1.9, 1.8
            self.X_max = 0.2
            self.mu_tau_ns = 123
            self.Gamma_ns, self.sigma_Z = 25.9, 1
            self.gamma_ns, self.sigma_U = 16.9, 6
            self.mu_AOD_degs, self.sigma_AOD_degs = -12.6, 5.9
            self.mu_AOA_degs, self.sigma_AOA_degs = 10.8, 5.3
            self.sigma_theta_AOD_degs, self.sigma_phi_AOD_degs = 8.5, 2.5
            self.sigma_theta_AOA_degs, self.sigma_phi_AOA_degs = 10.5, 11.5
            
        elif self.pos_grid[pos_idx].los == 'NLOS':
            self.environment = 'NLOS'
            self.propagation_distance = np.random.uniform(60,200)
            if self.center_frequency_ghz == 28:
                self.n_bar, self.sigma = 3.4, 9.7  # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
                self.mu_AOD, self.mu_AOA = 1.6, 1.6
                self.X_max = 0.5
                self.mu_tau_ns = 83
                self.Gamma_ns, self.sigma_Z = 49.4, 3
                self.gamma_ns, self.sigma_U = 16.9, 6
                self.mu_AOD_degs, self.sigma_AOD_degs = -4.9, 4.5
                self.mu_AOA_degs, self.sigma_AOA_degs = 3.6, 4.8
                self.sigma_theta_AOD_degs, self.sigma_phi_AOD_degs = 9.0, 2.5
                self.sigma_theta_AOA_degs, self.sigma_phi_AOA_degs = 10.1, 10.5
            elif self.center_frequency_ghz == 73:
                self.n_bar, self.sigma = 3.3, 7.6  # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
                self.mu_AOD, self.mu_AOA = 1.5, 2.5
                self.X_max = 0.5
                self.mu_tau_ns = 83
                self.Gamma_ns, self.sigma_Z = 56.0, 3
                self.gamma_ns, self.sigma_U = 15.3, 6
                self.mu_AOD_degs, self.sigma_AOD_degs = -4.9, 4.5
                self.mu_AOA_degs, self.sigma_AOA_degs = 3.6, 4.8
                self.sigma_theta_AOD_degs, self.sigma_phi_AOD_degs = 7.0, 3.5
                self.sigma_theta_AOA_degs, self.sigma_phi_AOA_degs = 6.0, 3.5
            else:
                print("Warning: Value not specified as exactly 28 or 73 for 'center_frequency_ghz', Table III does not specify explicit values, taking intermediary for n_bar and sigma.")
                self.n_bar, self.sigma = 3.35, 8.65  # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
                self.mu_AOD, self.mu_AOA = 1.5, 2.1
                self.X_max = 0.5
                self.mu_tau_ns = 83
                self.Gamma_ns, self.sigma_Z = 51.0, 3
                self.gamma_ns, self.sigma_U = 15.5, 6
                self.mu_AOD_degs, self.sigma_AOD_degs = -4.9, 4.5
                self.mu_AOA_degs, self.sigma_AOA_degs = 3.6, 4.8
                self.sigma_theta_AOD_degs, self.sigma_phi_AOD_degs = 11.0, 3.0
                self.sigma_theta_AOA_degs, self.sigma_phi_AOA_degs = 7.5, 6.0
                
                
    def fluctuation(self, nn=0, min_max_angles = (0,3.14159), tau = 1, mode = 'normal'):
        """
        Description
        -----------
        
        Channel fluctuation, the ME undergoes motion and the environment is checked to be changed (i.e., from LOS to NLOS).  If so we reinitialize the angles in the respective way.  The first angle in the 'LOS' environment is always the ME relative to the BS.
        
        Parameters
        ----------
        mode : str
            valid choices are 'normal' and 'intialization', defaults to 'normal'.  Set to 'initialization' in the __init__ method to get initial conditions for ME position.
        """
        assert mode == 'normal' or mode == 'initialization', "Please enter a valid kwarg selection for mode, valid choices are 'normal' and 'initialization'"
        
        if mode == 'normal':
            self.update_me(tau = tau)
            self.set_environment()
        elif mode == 'initialization': 
            self.set_environment()
        
        #Get ME position relative to grid index and spherical coordinate version
        pos_idx = self.get_grid_index(self.me_position[0],self.me_position[1],self.me_position[2])  #Index of spatially coherent grid based on ME initial position
        me_d,me_az,me_el = self.get_d_az_el(self.me_position[0], self.me_position[1], self.me_position[2])
        
        #If environment is the same as previous, update it accordingly
        if self.environment == self.prev_environment:
            self.path_loss_dB = 20* np.log10(4*np.pi*self.center_frequency_ghz * 1e9/3e8) + 10 * self.n_bar * np.log10(self.pos_grid[pos_idx].d) + self.atmospheric_loss_factor_db_m * self.pos_grid[pos_idx].d + self.pos_grid[pos_idx].sf
            
            vel_x,vel_y,vel_z = self.me_velocity[0],self.me_velocity[1],self.me_velocity[2]
            
            #These are from the User_manual_NYUSIM_v4  (3.25) for LOS and (3.26) for NLOS
            xBern = np.random.randint(1,3,self.total_num_paths) #Vector of 1s and 2s of length self.total_num_paths
            path_idx_start = 0
            
            #In a LOS scenario, the first path has no additional delay and is hence the strongest path.  No reflections.
            if self.environment == 'LOS':
                S_az_aod = (vel_y * np.cos(self.az_aod[0]) - vel_x * np.sin(self.az_aod[0]))/me_d/np.sin(self.el_aod[0]) 
                S_el_aod = (vel_x * np.cos(self.az_aod[0])*np.cos(self.el_aod[0]) + vel_y*np.cos(self.el_aod[0])*np.sin(self.az_aod[0])-vel_z*np.sin(self.el_aod[0]))/me_d 
                S_az_aoa = (vel_y * np.cos(self.az_aoa[0]) - vel_x * np.sin(self.az_aoa[0]))/me_d/np.sin(self.el_aoa[0]) 
                S_el_aoa = (vel_x * np.cos(self.az_aoa[0])*np.cos(self.el_aoa[0]) + vel_y*np.cos(self.el_aoa[0])*np.sin(self.az_aoa[0])-vel_z*np.sin(self.el_aoa[0]))/me_d 
                
                self.az_aod[0] = np.mod(self.az_aod[0] + tau * S_az_aod,2*np.pi)
                self.el_aod[0] = np.mod(self.el_aod[0] + tau * S_el_aod,2*np.pi)
                self.az_aoa[0] = np.mod(self.az_aoa[0] + tau * S_az_aoa,2*np.pi)
                self.el_aoa[0] = np.mod(self.el_aoa[0] + tau * S_el_aoa,2*np.pi)
                
                path_idx_start = 1
                
            #Update Delay
            delays_old = dcp(self.t_ns)
            # self.t_ns[0] = me_d * 1e9 / 3e8 + self.tau_ns[0] + self.rho[0] 
            deltaDist = [(vel_x * np.sin(self.el_aoa[ii])*np.cos(self.az_aoa[ii]) + vel_y * np.sin(self.el_aoa[ii])*np.sin(self.az_aoa[ii]) + vel_z * np.cos(self.el_aoa[ii]) ) * tau for ii in range(self.total_num_paths)]
            for ii in range(self.total_num_paths): self.t_ns[ii] = self.t_ns[ii] + deltaDist[ii-1]/3e8 * 1e9  #In the NYU Sim code this is self.t_ns[ii] - deltaDist[ii-1]/3e8 * 1e9, not sure how they polarity of the sign is incorporated.
            self.tau_ns, self.rho = self.get_delay_info(self.t_ns, self.Msp)

            #Update Power
            #Update the time cluster powers (mW) 
            z_db = self.sigma_Z * np.random.randn()
            # Pp = np.array([np.exp(-self.tau_ns[nn]/self.Gamma_ns) * 10**(self.sigma_Z * np.random.randn()/10) for nn in range(self.N)]) # (21) in [1]
            Pp = np.array([np.exp(-self.tau_ns[nn]/self.Gamma_ns) * 10**(z_db/10) for nn in range(self.N)]) # (21) in [1]
            self.P = Pp/np.sum(Pp) / 10**(self.path_loss_dB/10) # (22) in [1]
            
            #Update the cluster sub-path powers (mW)
            u_db = self.sigma_U * np.random.randn()
            # Pip = [[np.exp(-self.rho[nn][mm]/self.gamma_ns) * 10**(self.sigma_U * np.random.randn()/10) for mm in range(self.Msp[nn])] for nn in range(self.N) ] # (24) in [1]
            Pip = [[np.exp(-self.rho[nn][mm]/self.gamma_ns) * 10**(u_db/10) for mm in range(self.Msp[nn])] for nn in range(self.N) ] # (24) in [1]
            self.Pi = [[Pip[nn][mm] / np.sum(Pip[nn]) * self.P[nn] for mm in range(self.Msp[nn])] for nn in range(self.N)] # (25) in [1]
            self.rho = np.hstack(self.rho)
            self.Pi = np.hstack(self.Pi)
            
            #Update Phase
            dt_delay = np.array(self.t_ns)-np.array(delays_old)
            self.sp_phase = np.mod(self.sp_phase + dt_delay * 2 * np.pi * self.center_frequency_ghz * 1e-3, 2*np.pi)
            self.sp_phase = np.hstack(self.sp_phase)
            
            #Lines 460 through 478 translated from NYU Sim code
            for i_path in range(path_idx_start,self.total_num_paths):
                tempBern = xBern[i_path]
                deltaRS = self.az_aoa[i_path] + (-1)**tempBern * self.az_aod[i_path] + tempBern*np.pi
                vel_RS_x = np.mod(deltaRS+(-1)**tempBern * vel_x, 2*np.pi)
                vel_RS_y = np.mod(deltaRS+(-1)**tempBern * vel_y, 2*np.pi)
                vel_RS_z = 0.0
                
                # 0.3 comes from 1e-9 * 3e8
                S_az_aod = (vel_RS_y * np.cos(self.az_aod[i_path]) - vel_RS_x * np.sin(self.az_aod[i_path]))/np.sin(self.el_aod[i_path])/(self.t_ns[i_path] * 0.3) 
                S_el_aod = (vel_RS_x * np.cos(self.az_aod[i_path])*np.cos(self.el_aod[0]) + vel_RS_y*np.cos(self.el_aod[i_path])*np.sin(self.az_aod[i_path])-vel_RS_z*np.sin(self.el_aod[i_path]))/(self.t_ns[i_path] * 0.3) 
                S_az_aoa = (vel_RS_y * np.cos(self.az_aoa[i_path]) - vel_RS_x * np.sin(self.az_aoa[i_path]))/np.sin(self.el_aoa[i_path])/(self.t_ns[i_path] * 0.3)  
                S_el_aoa = (vel_RS_x * np.cos(self.az_aoa[i_path])*np.cos(self.el_aoa[i_path]) + vel_RS_y*np.cos(self.el_aoa[i_path])*np.sin(self.az_aoa[i_path])-vel_RS_z*np.sin(self.el_aoa[i_path]))/(self.t_ns[i_path] * 0.3)  
                
                self.az_aod[i_path] = np.mod(self.az_aod[i_path] + tau * S_az_aod,2*np.pi)
                self.el_aod[i_path] = np.mod(self.el_aod[i_path] + tau * S_el_aod,2*np.pi)
                self.az_aoa[i_path] = np.mod(self.az_aoa[i_path] + tau * S_az_aoa,2*np.pi)
                self.el_aoa[i_path] = np.mod(self.el_aoa[i_path] + tau * S_el_aoa,2*np.pi)
                    
        else:
            #Generate Number of Time Clusters N and AOD/AOA Spatial Lobes (SLs)
            self.N = np.random.randint(1,7)  # (12) from [1]
            self.L_AOD = np.min([5, np.max([1,np.random.poisson(self.mu_AOD)])]) # (13) from [1]
            self.L_AOA = np.min([5, np.max([1,np.random.poisson(self.mu_AOA)])]) # (14) from [1]
            
            #Generate the number of cluster sub-paths in each time cluster
            self.Msp = [np.random.randint(1,31) for nn in range(int(self.N))] # (15) from [1]
            self.total_num_paths = np.sum(self.Msp)
            
            #Generate the path loss for current position, this has dependencies on LOS/NLOS captured by n_bar and the pos_grid[pos_idx].sf
            self.path_loss_dB = 20* np.log10(4*np.pi*self.center_frequency_ghz * 1e9/3e8) + 10 * self.n_bar * np.log10(me_d) + self.atmospheric_loss_factor_db_m * me_d + self.pos_grid[pos_idx].sf
            
            #Generate intracluster sub-path excess delays (ns)
            inv_BB_bb_ns = 1/400e6 * 1e9
            x_max_rand = np.random.uniform(0,self.X_max)
            # self.rho = [[(inv_BB_bb_ns * mm)**(1 + np.random.uniform(0,self.X_max)) for mm in range(self.Msp[nn])] for nn in range(self.N)]
            self.rho = [[(inv_BB_bb_ns * mm)**(1 + x_max_rand) for mm in range(self.Msp[nn])] for nn in range(self.N)]
            
            #Generate cluster excess delays (ns)
            tau_pp = np.random.exponential(self.mu_tau_ns,self.N) # (18) in [1]
            Delta_tau = np.sort(tau_pp) - np.min(tau_pp) # (19) in [1]
            self.tau_ns = [0]
            for nn in np.arange(self.N-1): self.tau_ns.append(self.tau_ns[nn] + self.rho[nn][-1] + Delta_tau[nn+1] + 25)
            
            #Generate the time cluster powers (mW) 
            z_db = self.sigma_Z * np.random.randn()
            # Pp = np.array([np.exp(-self.tau_ns[nn]/self.Gamma_ns) * 10**(self.sigma_Z * np.random.randn()/10) for nn in range(self.N)]) # (21) in [1]
            Pp = np.array([np.exp(-self.tau_ns[nn]/self.Gamma_ns) * 10**(z_db/10) for nn in range(self.N)]) # (21) in [1]
            self.P = Pp/np.sum(Pp) / 10**(self.path_loss_dB/10) # (22) in [1]
            
            #Generate the cluster sub-path powers (mW)
            u_db = self.sigma_U * np.random.randn()
            # Pip = [[np.exp(-self.rho[nn][mm]/self.gamma_ns) * 10**(self.sigma_U * np.random.randn()/10) for mm in range(self.Msp[nn])] for nn in range(self.N) ] # (24) in [1]
            Pip = [[np.exp(-self.rho[nn][mm]/self.gamma_ns) * 10**(u_db/10) for mm in range(self.Msp[nn])] for nn in range(self.N) ] # (24) in [1]
            self.Pi = [[Pip[nn][mm] / np.sum(Pip[nn]) * self.P[nn] for mm in range(self.Msp[nn])] for nn in range(self.N)] # (25) in [1]
            self.Pi = np.hstack(self.Pi)
            
            #Generate the sub-path phases
            self.sp_phase = [[np.random.uniform(0,2*np.pi) for mm in range(self.Msp[nn])] for nn in range(self.N)]
            self.sp_phase = np.hstack(self.sp_phase)
            
            #Recover absolute time delays of cluster sub-paths, if LOS, self.tau_ns[0] and self.rho[0][0] should be 0.  Units in ns
            self.t_ns = [[me_d * 1e9 / 3e8 + self.tau_ns[nn] + self.rho[nn][mm] for mm in range(self.Msp[nn])] for nn in range(self.N)]
            self.rho = np.hstack(self.rho)
            self.t_ns = np.hstack(self.t_ns)
            
            #Generate mean AOA and AOD azimuth and elevation angles
            # Angles relative to the BS
            # The first AOA/AOD corresponds to the spatial position of the ME
            if self.environment == 'LOS':
                me_az_degs,me_el_degs = 180/np.pi * me_az, 180/np.pi * me_el
                self.az_mean_aod_degs = [me_az_degs]
                self.el_mean_aod_degs = [me_el_degs ]
                self.az_mean_aoa_degs = [me_az_degs + 180]
                self.el_mean_aoa_degs = [me_el_degs + 90]
            elif self.environment == 'NLOS': 
                self.az_mean_aod_degs = [np.random.uniform(0,360)]
                self.el_mean_aod_degs = [self.mu_AOD_degs + self.sigma_AOD_degs * np.random.randn()]
                self.az_mean_aoa_degs= [np.random.uniform(0,360)]
                self.el_mean_aoa_degs = [self.mu_AOA_degs + self.sigma_AOA_degs * np.random.randn()]
            
            for ll in range(1,self.L_AOD): self.az_mean_aod_degs.append(np.random.uniform(360*ll/self.L_AOD,360 * (ll+1)/self.L_AOD))
            for ll in range(1,self.L_AOD): self.el_mean_aod_degs.append(self.mu_AOD_degs + self.sigma_AOD_degs * np.random.randn())
            for ll in range(1,self.L_AOA): self.az_mean_aoa_degs.append(np.random.uniform(360*ll/self.L_AOA,360 * (ll+1)/self.L_AOA))
            for ll in range(1,self.L_AOA): self.el_mean_aoa_degs.append(self.mu_AOA_degs + self.sigma_AOA_degs * np.random.randn())
            
            self.az_aod,self.el_aod,self.az_aoa,self.el_aoa = [],[],[],[]
            self.cluster_lobe_mapping = []
            for nn in range(self.N):
                for mm in range(self.Msp[nn]):
                    if nn == 0 and mm == 0 and self.environment == 'LOS': ii_az,jj_az,ii_el,jj_el = 0,0,0,0
                    else:
                        ii_az,ii_el = np.random.randint(self.L_AOD,size = 2)
                        jj_az,jj_el = np.random.randint(self.L_AOA,size = 2)
                    self.az_aod.append(np.mod(np.pi/180*(self.az_mean_aod_degs[ii_az] + self.sigma_theta_AOD_degs * np.random.randn()),2*np.pi))
                    self.el_aod.append(np.mod(np.pi/180*(self.el_mean_aod_degs[ii_el] + self.sigma_phi_AOD_degs * np.random.randn()),2*np.pi))
                    self.az_aoa.append(np.mod(np.pi/180*(self.az_mean_aoa_degs[jj_az] + self.sigma_theta_AOA_degs * np.random.randn()),2*np.pi))
                    self.el_aoa.append(np.mod(np.pi/180*(self.el_mean_aoa_degs[jj_el] + self.sigma_phi_AOA_degs * np.random.laplace()),2*np.pi))
                    self.cluster_lobe_mapping.append([ii_az,ii_el,jj_az,jj_el])
                    
        #Build new channel response based on updates
        self.build_h()
        
        #Set memory for next environment check during next fluctuation
        self.prev_environment = dcp(self.environment)


    def update_me(self,tau = 1):
        """
        Description
        ------------
        Simulates ME motion with a White Gaussian Noise Acceleration model
        
        Parameters
        -----------
        tau : float
            Time in seconds during transition
            
        """
        u = self.sigma_u * np.random.randn(2)
        
        new_x_pos = self.me_position[0] + self.me_velocity[0] * tau + tau**2/2 * u[0]
        new_x_vel = self.me_velocity[0] + tau * u[0]
        new_y_pos = self.me_position[1] + self.me_velocity[1] * tau + tau**2/2 * u[1]
        new_y_vel = self.me_velocity[1] + tau * u[1]
        
        #No variation in z cordinates for now
        new_z_pos = self.me_position[2]
        new_z_vel = self.me_velocity[2]
        
        self.me_position = (new_x_pos,new_y_pos,new_z_pos)
        self.me_velocity = (new_x_vel,new_y_vel,new_z_vel)
        
    def get_los_map(self, area):
        """
        Generate the map of spatially correlated LOS/NLOS condition.
    
        Parameters
        -----------
        area : int
            The size of the area (grid size).
    
        Returns
        --------
        los_map : ndarray 
            The map of spatially correlated LOS/NLOS condition.
        """
    
        # Generate grid
        d_px = 1
        N = int(np.floor(area / d_px + 1))
        delta = self.d_co / d_px
        
        x, y = np.meshgrid(np.linspace(-area/2, area/2, N), np.linspace(-area/2, area/2, N))
        # x, y = np.meshgrid(np.arange(-2*N,2*N+1), np.arange(-N,N+1))
        
        Pr_LOS = np.zeros((N, N))
    
        d_2D = np.sqrt(x**2 + y**2) + np.finfo(float).eps
        d1 = 22
        d2 = 100
    
        # Probability of LOS/NLOS condition (depends on model and scenario)
        for i in range(N):
            for j in range(N):
                if self.scenario == 'UMi':
                    Pr_LOS[i, j] = (min(d1 / d_2D[i, j], 1) * (1 - np.exp(-d_2D[i, j] / d2)) + 
                                    np.exp(-d_2D[i, j] / d2))**2
                elif self.scenario == 'UMa':
                    if self.h_MS < 13 or self.h_BS <= 18:
                        C = 0
                    elif 13 < self.h_MS < 23:
                        C = ((self.h_MS - 13) / 10)**1.5 * 1.25e-6 * d_2D[i, j]**3 * np.exp(-d_2D[i, j] / 150)
                    else:
                        raise ValueError('Height of base station or mobile terminal is out of range.')
                    Pr_LOS[i, j] = ((min(d1 / d_2D[i, j], 1) * (1 - np.exp(-d_2D[i, j] / d2)) + 
                                     np.exp(-d_2D[i, j] / d2)) * (1 + C))**2
                elif self.scenario == 'RMa':
                    if d_2D[i, j] <= 10:
                        Pr_LOS[i, j] = 1
                    elif d_2D[i, j] > 10:
                        Pr_LOS[i, j] = np.exp(-(d_2D[i, j] - 10) / 1000)
    
        # The filter coefficient is considered as 0 when the distance is beyond 4*d_co.
        M = int(np.floor(8 * delta + 1))
        if M % 2 == 0:
            M -= 1
        h = np.zeros((M, M))
        init_map = np.random.randn(N + M - 1, N + M - 1)
    
        # Generate the filter
        for i in range(M):
            for j in range(M):
                h[i, j] = np.exp(-np.sqrt(((M + 1) / 2 - (i + 1))**2 + ((M + 1) / 2 - (j + 1))**2) / self.d_co)
    
        # Apply the filter and generate the correlated map
        corr_map_pad = convolve2d(init_map, h, mode='same')
        corr_q = corr_map_pad[(M+1)//2:(M+1)//2+N, (M+1)//2:(M+1)//2+N]
        corr_k = 0.5 * (1 + erf(corr_q / np.sqrt(2)))
    
        los_map = (corr_k < Pr_LOS).astype(int)
    
        return los_map


    def get_sf_map(self, area):
        """
        Obtain the map of spatially correlated shadow fading.
    
        Parameters
        -----------
        area : int
            The size of the area (grid size).
    
        Returns
        -------
        sf_map : ndarray 
            The map of spatially correlated shadow fading.
        """
    
        # Generate grid
        N = area + 1
        d_px = 1
        delta = self.d_co / d_px
    
        # The filter coefficient is considered as 0 when the distance is beyond 4*d_co.
        M = int(np.floor(8 * delta + 1))
        h = np.zeros((M, M))
        init_map = np.random.randn(N + M - 1, N + M - 1)
    
        # Generate the filter
        for i in range(M):
            for j in range(M):
                h[i, j] = np.exp(-np.sqrt(((M + 1) / 2 - (i + 1))**2 + ((M + 1) / 2 - (j + 1))**2) / self.d_co)
    
        # The mean of the shadow fading
        mu = 0
    
        # Filter the initial map
        corr_map_pad = convolve2d(init_map, h, mode='same')
    
        # Crop the map back to the original size
        start_idx = int((M + 1) / 2)
        corr_map = corr_map_pad[start_idx:start_idx + N, start_idx:start_idx + N]
    
        # Do normalization
        # Calculate the actual mean
        mu0 = np.mean(corr_map)
        # Calculate the actual variance
        sigma0 = np.sqrt(np.var(corr_map))
        
        # Scale the Correlated Map 
        if self.center_frequency_ghz == 28:
            sigma = 3.6
        elif self.center_frequency_ghz == 72:
            sigma = 5.2 
        else:
            print("Warning: Value not specified as exactly 28 or 73 for 'center_frequency_ghz', Table III does not specify explicit values, taking intermediary for n_bar and sigma.")
            sigma = 4.4  
            
        sf_los_map = corr_map * (sigma / sigma0) + (mu - mu0)
            
        if self.center_frequency_ghz == 28:
            sigma = 9.7  # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
        elif self.center_frequency_ghz == 73:
            sigma = 7.6  # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
        else:
            print("Warning: Value not specified as exactly 28 or 73 for 'center_frequency_ghz', Table III does not specify explicit values, taking intermediary for n_bar and sigma.")
            sigma = 8.65  # Path Loss Exponent (PLE) and Standard Deviation for Path Loss
                
        sf_nlos_map = corr_map * (sigma / sigma0) + (mu - mu0)
        
        return sf_los_map, sf_nlos_map


    def get_grid_index(self,x,y,z):
        """
        Description
        -----------
        Returns index of grid position based on position coordinates (x,y,z).  Used in cross-referencing position
        with SF map and LOS map.
        
        Parameters
        ----------
        x : float
            Position in x-coordinate
        y : float
            Position in y-coordinate
        z : float
            Position in z-coordinate
        """
        return np.argmax([np.sqrt((pos.x-x)**2 + (pos.y-y)**2 + (pos.z-z)**2) for pos in self.pos_grid])
    
    def get_d_az_el(self,x,y,z):
        """
        Description
        ------------
        Translates x,y,z position into spherical coordinates relative to the origin

        Parameters
        ----------
        x : float
            Position in x-coordinate
        y : float
            Position in y-coordinate
        z : float
            Position in z-coordinate

        Returns
        -------
        d : float
            Distance between origin and x,y,z
        az : float
            Azimuth angle
        el : float
            Elevation/Zenith angle

        """
        # Radial distance
        d = np.sqrt(x**2 + y**2 + z**2)
        
        # Azimuthal angle (phi), in the xy-plane
        az = np.arctan2(y, x)
        
        # Polar angle (theta), from the z-axis
        el = np.arccos(z / d) if d != 0 else 0
        return d,az,el


    def get_delay_info(self, DelayList, nSP):
        # DelayList is LOS + NLOS, so we calculate excess delay
        nTC = len(nSP)
        tau = DelayList - np.min(DelayList)
    
        # Compute the cumulative sum of nSP to get indices
        tmp = [np.sum(nSP[:i+1]) for i in range(nTC)]
    
        # Get the first component indices
        firstComponentIdx = np.array([0] + tmp[:-1]) + 1
        tau_n = tau[firstComponentIdx - 1]  # Adjust for Python's 0-based indexing
    
        # Prepare the structure for rho_mn
        # rho_mn = {}
    
        # Get edge indices
        edgeIdx = np.append(firstComponentIdx, len(DelayList) + 1)
    
        # Compute rho_mn for each component
        rho_mn = []
        for cIdx in range(nTC):
            sp = tau[edgeIdx[cIdx] - 1:edgeIdx[cIdx + 1] - 1]  # Adjust for Python indexing
            spDelay = sp - np.min(sp)
            # rho_mn[f'c{cIdx + 1}'] = spDelay
            rho_mn.append(spDelay)
        # rho_mn = np.hstack(rho_mn)
        return tau_n, rho_mn

    def build_h(self):
        self.ht = 1j * np.zeros(self.M)
        
        #Received ULA Channel Response from BS transmission
        for ii in range(self.total_num_paths):
            self.ht += np.sqrt(self.Pi[ii]) * np.exp(1j * self.sp_phase[ii]) *  avec(self.az_aod[ii], self.M)
            
    class SpatiallyCoherentPosition:
        """
        Description
        -----------
        Used with the NYU2 channel class to organize the grid of spatially correlated values.
        
        Parameters
        ----------
        params - dict
        
        Notes
        -----
        Careful with units, some parameters are derived in dB, but calculations are in mW.
        """
        def __init__(self,params):
            self.index = params['index']
            self.x = params['x'] #x Position relative to BS
            self.y = params['y'] #y Position relative to BS
            self.z = params['z'] #z Position relative to BS
            self.d = np.sqrt(self.x**2 + self.y**2 + self.z**2) #Distance from BS (at origin) and grid point
            
            # self.az = np.atan2(self.y, self.x)
            self.az = np.arctan2(self.y,self.x)
            
            self.el = np.arccos(self.z / self.d) if self.d != 0 else 0
                    
            self.los = None
        
class DynamicMotion(BasicChannel):
    """
    DynamicMotion is a class to represent various parameters and computations for a communication channel with a uniform linear array (ULA) along the x-axis.
    
    Attributes
    ----------
    M : int
        Number of elements in the array.
    sigma_u : float
        Standard deviation of the kinematic motion.
    initial_angle : float
        Initial angle in radians, converted from degrees.
    L : int
        Number of signal paths.
    channel_mode : str
        Mode of the channel ('WGNA', 'FixedV', 'GaussianJumps').
    seed : int
        Seed for random number generation.
    angles : np.ndarray
        Array of angles for the signal paths.
    alphas : np.ndarray
        Array of complex gains for the signal paths.
    x_k : np.ndarray
        State vector containing initial angle, angular velocity, and real/imaginary parts of the first path's gain.
    ht : np.ndarray
        Array representing the sum of the channel responses for all paths.
    F : np.ndarray
        State transition matrix.
    Qu : np.ndarray
        Kinematic motion covariance matrix.
    G : np.ndarray
        Mode-dependent matrix used in the state transition.
    Qv : np.ndarray
        Observation covariance matrix.
    rho : float
        Fading parameter.
    tau : float
        Time step for state transitions.

    Methods
    -------
    fluctuation(self):
        Updates the channel state to simulate fluctuations in movement and fading.
        
    Notes
    -----
    Multi-path effects tend to take place around the maain path, we choose this value
    to be .35 radians (~20 degrees).  More detail available in [2]
    - [1] Akdeniz, Mustafa Riza, et al. "Millimeter wave channel modeling and cellular capacity evaluation." IEEE journal on selected areas in communications 32.6 (2014): 1164-1179.
    - [2] Rappaport, Theodore S., et al. "Millimeter wave mobile communications for 5G cellular: It will work!." IEEE access 1 (2013): 335-349.

    - The attribute sigma_u is used as the initial angular velocity for the fixed velocity case and the WGNA case.
    """
    def __init__(self,params):
        """
        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - 'num_elements': int
                Number of elements in the array.
            - 'sigma_u_degs': float
                Standard deviation of the kinematic angular motion in degrees.
            - 'initial_angle_degs': float
                Initial angle in degrees.
            - 'num_paths': int
                Number of signal paths.
            - 'mode': str
                Mode of the channel ('WGNA', 'FixedV', 'GaussianJumps').
              'scenario' : str
                Indicates LOS or NLOS scenario, where the dominant beam is 10 dB higher or 3 dB higher than multipath elements, respectively.
            - 'seed': int
                Seed for random number generation.
            - 'fading': float
                Fading parameter.
            - 'time_step': float
                Time step for state transitions.
        """
        super().__init__(params)
        self.M = params['num_elements']
        self.sigma_u = params['sigma_u_degs'] * np.pi/180
        self.initial_angle = params['initial_angle_degs'] * np.pi/180
        self.rho = params['fading']
        self.tau = params['time_step']
        self.L = params['num_paths']
        if 'scenario' in params: self.scenario = params['scenario']
        else: self.scenario = 'LOS'
        self.mode = params['mode']
        self.seed = params['seed']
        

        
        #State Transition Matrix
        self.F = np.array([ [1, self.tau, 0,        0       ],
                            [0, 1,            0,        0       ],
                            [0, 0,            self.rho, 0       ],
                            [0, 0,            0,        self.rho]])
        
        #Kinematic Motion Covariance
        self.Qu = np.array([  [self.sigma_u**2, 0,                 0                 ],
                        [0,               (1-self.rho**2)/2, 0                 ],
                        [0,               0,                 (1-self.rho**2)/2]])
        
        if self.mode == 'WGNA':
            self.G = np.array([ [self.tau**2/2, 0, 0],
                                [self.tau,      0, 0],
                                [0,             1, 0],
                                [0,             0, 1]])
            self.initial_angular_velocity = dcp(self.sigma_u)
            
        elif self.mode == 'FixedV':
            self.G = np.array([ [0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            self.initial_angular_velocity = dcp(self.sigma_u)
            
        elif self.mode == 'GaussianJumps':
            self.G = np.array([ [self.tau, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            self.initial_angular_velocity = 0.0
            
            
        self.paths = [DynamicMotion.Path({'initial_angle' : self.initial_angle, 'initial_angular_velocity' : self.initial_angular_velocity, 'scenario' : self.scenario, 'is_dominant_path' : True, 'mode' : self.mode})]
        for ll in range(self.L-1):
            self.paths.append(DynamicMotion.Path({'initial_angular_velocity' : self.initial_angular_velocity, 'scenario' : self.scenario, 'is_dominant_path' : False, 'mode' : self.mode}))
        self.angles = [path.state[0] for path in self.paths]
        self.alphas = [path.state[2] + 1j * path.state[3] for path in self.paths]
        # self.angles = np.concatenate([[self.initial_angle],np.random.uniform(self.initial_angle - .35,self.initial_angle + .35,self.L-1)])
        # if self.scenario == 'LOS':
        #     self.alphas = randcn(self.L) * np.concatenate([[1.0],.1 * np.ones(self.L-1)])
        # else:
        #     self.alphas = randcn(self.L) * np.concatenate([[1.0],.5 * np.ones(self.L-1)])
        
        self.log_data = {'angles' : [self.angles],
                         'alphas' : [self.alphas]}
        
        # self.ht = np.sum([self.alphas[ii] * avec(self.angles[ii],self.M) for ii in np.arange(self.L)],axis = 0)
        self.ht = np.sum([(path.state[2] + 1j * path.state[3]) * avec(path.state[0],self.M) for path in self.paths],axis = 0)
        
        #Observation Covariance
        self.Qv = self.sigma_v**2/2 * np.eye(2)
    
    def fluctuation(self, nn=0, angle_limits = (0,3.141592653589793),*args):
        """
        Description
        -----------
        Time fluctuation of channel due to fading and motion.
        
        Parameters
        ----------
        angle_limits : tuple of floats
            tuple indicating min and max angles of simulation

        """
        def wrap_angle(x,angle_min,angle_max): 
            """
            Wraps an angle to be within a specified range.
            
            This function adjusts the first element of the input array `x` so that it
            falls within the specified range [angle_min, angle_max]. If the angle exceeds
            the maximum or minimum bounds, it is wrapped around accordingly. Additionally,
            it handles the case where the angle estimate is NaN.
            
            Parameters
            ----------
            x : np.ndarray
                The input array where the first element represents the angle to be wrapped.
            angle_min : float
                The minimum allowable angle.
            angle_max : float
                The maximum allowable angle.
            
            Returns
            -------
            x : np.ndarray
                The input array with the first element adjusted to be within the specified range.
            """
            swath_width = np.abs(angle_max-angle_min)
            if np.isnan(x[0]):
                print('Angle Estimate is nan')
            x[0] = np.mod(x[0],np.pi)
            if x[0] > angle_max: x[0] = x[0] - swath_width
            elif x[0] < angle_min: x[0] = x[0] + swath_width
            return x
        
        #Loop through each path and update it accordingly
        for path in self.paths:
            u = np.random.multivariate_normal(np.zeros(3),self.Qu)
            path.state = self.F@path.state + self.G@u
            if path.is_dominant_path: path.state = wrap_angle(path.state, angle_limits[0],angle_limits[1])
        self.angles = [path.state[0] for path in self.paths]
        self.alphas = [path.state[2] + 1j * path.state[3] for path in self.paths]
        self.ht = np.sum([(path.state[2] + 1j * path.state[3]) * avec(path.state[0],self.M) for path in self.paths],axis = 0)
        # self.x_k = self.F@self.x_k + self.G@u
        # self.x_k = wrap_angle(self.x_k,angle_limits[0],angle_limits[1])
        # self.ht = (self.x_k[2] + 1j * self.x_k[3]) * avec(self.x_k[0],self.M)
        # if self.scenario == 'LOS':
        #     self.alphas = np.concatenate([[(self.x_k[2] + 1j * self.x_k[3])],randcn(self.L-1) * .1 * np.ones(self.L-1)])
        # else:
        #     self.alphas = np.concatenate([[(self.x_k[2] + 1j * self.x_k[3])],randcn(self.L-1) * .5 * np.ones(self.L-1)])
        # if self.sigma_u > 0:
        #     self.angles = np.concatenate([[self.x_k[0]],np.random.uniform(self.x_k[0] - .35,self.x_k[0] + .35,self.L-1)])
        # self.ht = np.sum([self.alphas[ii] * avec(self.angles[ii],self.M) for ii in np.arange(self.L)],axis = 0)
        
        self.log_data['angles'].append(self.angles)
        self.log_data['alphas'].append(self.alphas)

    class Path:
        def __init__(self,params):
            self.is_dominant_path = params['is_dominant_path']
            if params['is_dominant_path']:
                self.initial_angle = params['initial_angle']
                if params['scenario'] == 'LOS': self.initial_alpha = randcn(1)
                else: self.initial_alpha = np.sqrt(.1) * randcn(1)
            else:
                self.initial_angle = np.random.uniform(0,2*np.pi)
                self.initial_alpha = np.sqrt(.1) * randcn(1)
            
            self.initial_angular_velocity = params['initial_angular_velocity']
        
            self.state = np.array([self.initial_angle, self.initial_angular_velocity, np.real(self.initial_alpha), np.imag(self.initial_alpha)])
        
            