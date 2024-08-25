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
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.special import erf
import scipy.io
from .util import * 

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
        
    def array_response(self,transmit_power_dbw = 1,with_noise = True):
        """
        
        Notes
        -----
        The arg parameter 'transmit_power_dbw' is used only for debugging in this particular class object.
        
        It is also a placeholder so that algorithms may use either channel model.
        
        """
        tx_power_watts = 10**(transmit_power_dbw/10)
        if with_noise:
            return np.sqrt(tx_power_watts) * self.ht + self.sigma_v * randcn(len(self.ht))
        else: 
            return np.sqrt(tx_power_watts) * self.ht

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
        print('Initializing random number generator seed within Channel')
        np.random.seed(seed = self.seed)
        
    def array_response(self,transmit_power_dbw, with_noise = True):
        tx_power_watts = 10**(transmit_power_dbw/10)
        return np.sqrt(tx_power_watts) * self.ht + self.sigma_v * randcn(len(self.ht))

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
    
    def fluctuation(self,*args):
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
        self.mode = params['mode']
        self.seed = params['seed']
        
        self.angles = np.concatenate([[self.initial_angle],np.random.uniform(self.initial_angle - .35,self.initial_angle + .35,self.L-1)])
        self.alphas = randcn(self.L) * np.concatenate([[1.0],.1 * np.ones(self.L-1)])
        
        self.ht = np.sum([self.alphas[ii] * avec(self.angles[ii],self.M) for ii in np.arange(self.L)],axis = 0)
        
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
            self.x_k = np.array([self.initial_angle,self.sigma_u,np.real(self.alphas[0]),np.imag(self.alphas[0])])
            
        elif self.mode == 'FixedV':
            self.G = np.array([ [0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            self.x_k = np.array([self.initial_angle,self.sigma_u,np.real(self.alphas[0]),np.imag(self.alphas[0])])
            
        elif self.mode == 'GaussianJumps':
            self.G = np.array([ [self.tau, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            self.x_k = np.array([self.initial_angle,0.0,np.real(self.alphas[0]),np.imag(self.alphas[0])])
            
        #Observation Covariance
        self.Qv = self.sigma_v**2/2 * np.eye(2)
    
    def fluctuation(self, nn, angle_limits = (0,3.141592653589793),*args):
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
        
        u = np.random.multivariate_normal(np.zeros(3),self.Qu)
        self.x_k = self.F@self.x_k + self.G@u
        self.x_k = wrap_angle(self.x_k,angle_limits[0],angle_limits[1])
        self.ht = (self.x_k[2] + 1j * self.x_k[3]) * avec(self.x_k[0],self.M)
        self.alphas = np.concatenate([[(self.x_k[2] + 1j * self.x_k[3])],randcn(self.L-1) * .1 * np.ones(self.L-1)])
        self.angles = np.concatenate([[self.x_k[0]],np.random.uniform(self.x_k[0] - .35,self.x_k[0] + .35,self.L-1)])
        self.ht = np.sum([self.alphas[ii] * avec(self.angles[ii],self.M) for ii in np.arange(self.L)],axis = 0)
        


