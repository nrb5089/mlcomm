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
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as dcp
from numpy.linalg import inv
import pickle
from mlcomm.util import *


class BinaryHierarchicalCodebook:
    """
    Description
    ------------
    Organizes the master codebook using a binary tree graph.  Each node corresponds
    to a beamforming vector, and has a master index (midx).
    
    Along with the master index, each level has an index 'h' along with a left-to-right index 'i' 
    from 0 to 2**(h + 1)-1, for instance, node 3 has (h,i) = (1,1), and node 11 has (h,i) = (3,5).
    
    Each node is a 'Node' class which holds various attributes along with indices.
    
    The beamforming vectors of the codebook are calculated based on the min and max angles specified 
    in the params dict (described in Parameters).  The sets of phase shifts are designed to be applied to a 
    Uniform Linear Array (ULA) along the x-axis.  The main beam of the beamforming patterns  that correspond 
    to the beamforming vectors are non-overlapping and evenly spaced to cover the specfied swath.
    

    Attributes
    -----------
    H : int
        Number of levels in the hierarchical codebook.
        
    M : int
        Number of physical antenna elements.
    
    B : int
        Number of RF chains, the number of analog-to-digital or 
        digital-to-analog converters.
        
    S : int
        Number of data streams to support simultaneous users.
    
    beamwidths : numpy ndarray of floats
        beamwidth indicating the coverage area for beams, the h element is the h level's beamwidth.
    
    steered_angles : list of numpy ndarray of floats
        The h element in the list is a numpy ndarray of length 2^(h+1) with elements indicating the steered beam direction for beam (h,i).
        
    nodes : list of Node type class
        Each element of the list is an instance of Node class for a particular beamforming vector.
        
    level_midxs : list of lists
        Each element in the list is a list of ints, the h list is the master indices at level h.
    
    base_midxs: list of lists
        equivalent to level_midxs for this class.
    
    g : float
        gain value of the codebook
    
    NH : int
        Total number of beamforming vectors considered in all MAB games along one path.
        
    Notes
    ------
    Additional attributes are passed to each instance of class Node in nodes.
    
    Recommend set 'depth' = np.ceil(np.log2(num_elements)) or set 'num_elements' <= 2**depth.
    
    Quality of beamforming vector patterns degrades with fewer RF chains, set by num_rf_chains.
    
    Example Use
    ------------
    For use in Multi-Armed Bandit algorithms in which each beam represents an
    arm of a bandit, one may assign bandit statistics to each Node class object
    with mean rewards, number of times pulled, empirical mean, etc.
    
    Codebook featured in works such as 
    - Chiu, Sung-En, Nancy Ronquillo, and Tara Javidi. "Active learning and CSI acquisition for mmWave initial alignment." IEEE Journal on Selected Areas in Communications 37.11 (2019): 2474-2489.
    - Blinn, Nathan, Jana Boerger, and Matthieu Bloch. "mmWave Beam Steering with Hierarchical Optimal Sampling for Unimodal Bandits." ICC 2021-IEEE International Conference on Communications. IEEE, 2021.
    """
    def __init__(self,params):
        """
        Parameters
        -----------
        params : A dict containing the following keys
        
                'depth' : int 
                    Number of levels in the graph.
                    
                'num_elements' : int 
                    Physical number of antenna elements in ULA.
                    
                'num_rf_chains' : int 
                    Number of analog to digital or digital to analog converters.
                    
                'num_data_streams' : int 
                    Number of data streams corresponding to 1-to-1 with users.
                    
                'min_max_angles_degs': tuple of floats
                    A tuple containing two angles (in degrees), representing minimum and maximum angles.  Max range of (0,180).
                
        Returns
        --------
        None
        """
        # avec = avec_ula
        self.H = params['depth']
        self.M = params['num_elements']
        self.B = params['num_rf_chains']
        self.S = params['num_data_streams']
        self.min_max_angles = np.array(params['min_max_angles_degs']) * np.pi/180
        assert self.min_max_angles[1] > self.min_max_angles[0], "Error: Max angle must be greater than Min angle"
        
        #Determine beamwdith and pointing angles of each beamforming pattern
        self.beamwidths = np.abs(self.min_max_angles[1]-self.min_max_angles[0]) / 2**np.arange(1,self.H+1)
        self.steered_angles = [np.arange(self.min_max_angles[0] + beamwidth/2,self.min_max_angles[1],beamwidth) for beamwidth in self.beamwidths]
        
        #Candidate vectors is the overdetermined dictionary described in Alkhateeb-2014
        candidate_vectors = np.vstack([avec(angle, self.M) for angle in np.linspace(self.min_max_angles[0],self.min_max_angles[1],2*self.M)]).T
        
        
        #Initialize Nodes
        midx = 0
        self.level_midxs = []
        self.nodes = {}
        for hh,beamwidth in enumerate(self.beamwidths):
            midxs_level = []
            for ii,steered_angle in enumerate(self.steered_angles[hh]):
                self.nodes[midx] = Node({'master_index' : midx,
                                         'level' :  hh,
                                          'level_index' : ii,
                                          'steered_angle' : steered_angle,
                                          'cvg_region_start' : steered_angle - beamwidth/2,
                                          'cvg_region_stop' : steered_angle + beamwidth/2})
                                          # 'aggregated_indices' : np.arange(ii * 2**(self.H-hh-1), (ii+1)*2**(self.H-hh-1)) })
                
                aggregated_indices = np.arange(ii * 2**(self.H-hh-1), (ii+1)*2**(self.H-hh-1)) 
                
                # optimal_beamforming_vector = np.sum([avec(steered_angle,self.M) for steered_angle in self.steered_angles[-1][self.nodes[-1].aggregated_indices]],axis = 0)
                optimal_beamforming_vector = np.sum([avec(steered_angle,self.M) for steered_angle in self.steered_angles[-1][aggregated_indices]],axis = 0)
                F_bb, F_rf, F = build_hybrid_vector(optimal_beamforming_vector,candidate_vectors,self.B,self.S)
                
                self.nodes[midx].f_bb, self.nodes[midx].f_rf,self.nodes[midx].f = F_bb,F_rf,F
                
                #Define edges of the node with it's parent, child, and sibling nodes (all nearest neighbors)
                if ii == 0: self.nodes[midx].prior_sibling = dcp(midx)
                else: self.nodes[midx].prior_sibling = midx - 1
                
                if ii == len(self.steered_angles[hh])-1: self.nodes[midx].post_sibling = dcp(midx)
                else: self.nodes[midx].post_sibling = midx + 1
                
                if hh == 0: self.nodes[midx].zoom_out_midx = dcp(midx)
                else: self.nodes[midx].zoom_out_midx = int(np.floor((midx-2)/2))
                
                if hh == len(self.beamwidths)-1: self.nodes[midx].zoom_in_midxs = [midx]
                else: self.nodes[midx].zoom_in_midxs = [int(2*midx + 2),int(2*midx + 3)]
                

                    
                midxs_level.append(midx)
                midx += 1
                
            self.level_midxs.append(midxs_level)
            self.base_midxs = dcp(self.level_midxs)
        
        self.g = 2
        self.NH = len(self.base_midxs[0]) + 2 * (self.H-1)
        
    def get_midx(self,h,i): return self.level_midxs[h][i]

class TernaryPointedHierarchicalCodebook:
    """
    Description
    ------------
    Each node is a 'Node' class which holds various attributes along with indices
    
    The beamforming vectors of the codebook are calculated based on the min and max angles specified 
    in the params dict (described in Parameters).  The sets of phase shifts are designed to be applied to a 
    Uniform Linear Array (ULA) along the x-axis.  The main beam of the beamforming patterns  that correspond 
    to the beamforming vectors are non-overlapping and evenly spaced to cover the specfied swath.
    
    Attributes
    -----------
    num_h0 : int
        Number of initial beamforming vectors in the top-level base set.
        
    H : int
        Number of levels in the hierarchical codebook.
        
    M : int
        Number of physical antenna elements.
    
    B : int
        Number of RF chains, the number of analog-to-digital or 
        digital-to-analog converters.
        
    S : int
        Number of data streams to support simultaneous users.
    
    beamwidths : numpy ndarray of floats
        beamwidth indicating the coverage area for beams, the h element is the h level's beamwidth.
    
    steered_angles : list of numpy ndarray of floats
        The h element in the list is a numpy ndarray of length 2^(h+1) with elements indicating the steered beam direction for beam (h,i).
        
    nodes : list of Node type class
        Each element of the list is an instance of Node class for a particular beamforming vector.
        
    level_midxs : list of lists
        Each element in the list is a list of ints, the h list is the master indices at level h.
    
    base_midxs: list of lists
        the set of size self.num_h0 which minimally covers the range of angles with the broadest beamforming patterns.

    g : float
        gain value of the codebook
    
    NH : int
        Total number of beamforming vectors considered in all MAB games along one path.
        
    Notes
    ------
    Additional attributes are passed to each instance of class Node in nodes.
    
    Recommend set 'depth' = np.ceil(np.log2(num_elements)) or set 'num_elements' <= 2**depth.
    
    Quality of beamforming vector patterns degrades with fewer RF chains, set by num_rf_chains.
    
    Example Use
    ------------
    For use in Multi-Armed Bandit algorithms in which each beam represents an
    arm of a bandit, one may assign bandit statistics to each Node class object
    with mean rewards, number of times pulled, empirical mean, etc.
    """
    def __init__(self,params):
        """
        Parameters
        -----------
        params : A dict containing the following keys
                'num_initial_non_overlapping' : int
                    Number of beamforming vectors at level h=0 that are non-overlapping
                    
                'depth' : int 
                    Number of levels in the graph.
                    
                'num_elements' : int 
                    Physical number of antenna elements in ULA
                    
                'num_rf_chains' : int 
                    Number of analog to digital or digital to analog converters
                    
                'num_data_streams' : int 
                    Number of data streams corresponding to 1-to-1 with users
                    
                'min_max_angles_degs': tuple of floats
                    A tuple containing two angles (in degrees), representing minimum and maximum angles.  Max range of (0,180)
                
        Returns
        --------
        None
        """
        
        self.num_h0 = params['num_initial_non_overlapping']
        self.H = params['depth']
        self.M = params['num_elements']
        self.B = params['num_rf_chains']
        self.S = params['num_data_streams']
        self.min_max_angles = np.array(params['min_max_angles_degs']) * np.pi/180
        assert self.min_max_angles[1] > self.min_max_angles[0], "Error: Max angle must be greater than Min angle"
        
        #Determine beamwdith and pointing angles of each beamforming pattern
        self.beamwidths = np.abs(self.min_max_angles[1]-self.min_max_angles[0]) / self.num_h0 / 3**np.arange(self.H)
        self.steered_angles = np.arange(self.min_max_angles[0] + self.beamwidths[-1]/2,self.min_max_angles[1],self.beamwidths[-1])
        quantities_to_aggregate = 3**np.arange(self.H)[-1::-1]  #number of beamforming vectors aggregated at each level.
        
        #Aggregates beams that are outside the specified coverage area.  Candidate vectors for HAD codebook need to 
        #include these
        #TODO: Add option to not do this.
        over_coverage_steered_angles = []
        for hh in np.arange(self.H-1):
            over_coverage_steered_angles.append(np.concatenate([self.steered_angles[0] - (np.arange(np.floor(quantities_to_aggregate[hh]/2))[-1::-1] +1) * self.beamwidths[-1],
                                                     self.steered_angles, 
                                                     self.steered_angles[-1] + (np.arange(np.floor(quantities_to_aggregate[hh]/2))[-1::-1] +1) * self.beamwidths[-1]]))
        over_coverage_steered_angles.append(self.steered_angles)
        
        #Candidate vectors is the overdetermined dictionary described in Alkhateeb-2014, since the widest beamforming vectors aggregated directions outside the coverage
        #region, we set the cardinality of the overdetermined matrix to be twice that number of beamforming vectors aggregated.
        candidate_vectors = np.vstack([avec(angle, self.M) for angle in np.linspace(over_coverage_steered_angles[0][0]-self.beamwidths[-1]/2,over_coverage_steered_angles[0][-1]+self.beamwidths[-1]/2,2*(quantities_to_aggregate[0] + self.num_h0*3**(self.H-1)))]).T
        
        
        #Initialize Nodes
        midx = 0
        self.level_midxs = []
        self.base_midxs = []
        self.nodes = {}
        for hh,beamwidth in enumerate(self.beamwidths):
            midxs_level = []
            for ii,steered_angle in enumerate(self.steered_angles):
                self.nodes[midx] = Node({'master_index' : midx,
                                          'level' :  hh,
                                          'level_index' : ii,
                                          'steered_angle' : steered_angle,
                                          'cvg_region_start' : steered_angle - beamwidth/2,
                                          'cvg_region_stop' : steered_angle + beamwidth/2})
               
                
                optimal_beamforming_vector = np.sum([avec(steered_angle,self.M) for steered_angle in over_coverage_steered_angles[hh][ii:ii+quantities_to_aggregate[hh]]],axis = 0)
                F_bb, F_rf, F = build_hybrid_vector(optimal_beamforming_vector,candidate_vectors,self.B,self.S)
                self.nodes[midx].f_bb, self.nodes[midx].f_rf,self.nodes[midx].f = F_bb,F_rf,F
                
                midxs_level.append(midx)
                midx += 1
                
            self.level_midxs.append(midxs_level)
            self.base_midxs.append(np.arange(3**hh * self.num_h0) * quantities_to_aggregate[hh] + np.floor(quantities_to_aggregate[hh]/2) + len(self.steered_angles) * hh)
            
        #Define edges of the node with it's parent, child, and sibling nodes (all nearest neighbors), for the
        # Pointed codebook, we define the siblings as those nearest neighboring at the same level that do not overlap.
        for node in self.nodes.values():
            hh,ii = node.h,node.i
            
            node.prior_sibling = self.level_midxs[hh][ii-quantities_to_aggregate[hh]]
            
            node.post_sibling = self.level_midxs[hh][int(np.mod(ii + quantities_to_aggregate[hh],len(self.steered_angles)))]
            
            if hh == 0: node.zoom_out_midx = dcp(node.midx)
            else: node.zoom_out_midx = self.level_midxs[hh-1][ii]
        
        for node in self.nodes.values():
            hh,ii = node.h,node.i
            if hh == len(self.beamwidths)-1: node.zoom_in_midxs = [node.midx]
            else: node.zoom_in_midxs = [self.nodes[self.level_midxs[hh+1][ii]].prior_sibling,self.level_midxs[hh+1][ii],self.nodes[self.level_midxs[hh+1][ii]].post_sibling]
        
        self.g = 3
        self.NH = len(self.base_midxs[0]) + 3 * (self.H-1)
        
        
    def get_midx(self,h,i): return self.level_midxs[h][i]

    
class Node:
    """
    A class to represent a node in a hierarchical structure.

    Attributes
    ----------
    midx : int
        The master index of the node.
    
    h : int
        The hierarchical level of the node.
    
    i : int
        The index of the node at its level.
    
    indices : tuple
        A tuple containing the hierarchical level and index of the node.
    
    steered_angle : float
        The steered angle of the node.
    
    cvg_region_start : float
        The start of the coverage region for the node.
    
    cvg_region_stop : float
        The stop of the coverage region for the node.
    
    beamwidth : float
        The beamwidth of the node, calculated as the absolute difference between
        the coverage region start and stop.

    zoom_out_midx : int
        Provided by initializing codebook class object. For the TernaryPointedHierarchicalCodebook, this is a master 
        index of a beamforming vector with the same steering angle at level h-1.  This is the parent node in a 
        BinaryHierarchicalCodebook
        
    zoom_in_midxs : list of ints
        Provided by initializing codebook class object. Master indices of narrower beamforming vectors at h+1 that 
        the Node aggregates.  They are non-overlapping.
    """
    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
            A dictionary containing the following keys
                
            - 'master_index': int
                The master index of the node.
                
            - 'level': int
                The hierarchical level of the node.
                
            - 'level_index': int
                The index of the node at its level.
                
            - 'steered_angle': float
                The steered angle of the node.
                
            - 'cvg_region_start': float
                The start of the coverage region for the node.
                
            - 'cvg_region_stop': float
                The stop of the coverage region for the node.
        """
        self.midx = params['master_index']
        self.h = int(params['level'])
        self.i = int(params['level_index'])
        self.indices = (self.h, self.i)
        self.steered_angle = params['steered_angle']
        self.cvg_region_start = params['cvg_region_start']
        self.cvg_region_stop = params['cvg_region_stop']
        self.beamwidth = np.abs(params['cvg_region_stop'] - params['cvg_region_start'])

        
def save_codebook(cb_graph,filename,savepath = './'):
    """
    Description
    -----------
    Save cb_graph for later use. Often faster than intial creation.

    Parameters
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
        
    filename : string
        Name of save pickle file.
        
    savepath : string, optional
        The default is './'.

    Returns
    -------
    None.
    """
    pickle.dump(cb_graph,open(savepath + filename + '.pkl','wb'))
    
def load_codebook(filename,loadpath):
    """
    Description
    -----------
    Load cb_graph for use.
    
    Parameters
    ----------
    filename : string
        Name of save pickle file.
        
    loadpath : string, optional
        The default is './'.

    Returns
    -------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
    """
    
    return pickle.load(open(loadpath + filename + '.pkl','rb'))
    
def build_hybrid_vector(F_opt,A_can,num_rf_chains,num_data_streams=1):
    """
    Description
    -----------
    Build hybrid analog-digital beamforming vector for optimal set of weights
    contained in vector F_opt.  The final hybrid beamforming vector is expressed
    as the matrix product, F = F_bb@F_rf.  If the number of RF chains is equal
    to the number of elements, F is equivalent to F_opt.
    
    There is an extra set of lines of code that ensures that different columns
    of the candidate vectors matrix are selected as to not make the resulting codeword 
    matrix singular.
    
    Parameters
    -----------
    F_opt : numpy-ndarray of complex float
        array of complex floats of length M, corresponding to M number of elements
        for the ideal beamforming vector.
        
    A_can : numpy-ndarray of complex float
        matrix whose columns are the possible beamforming vectors which may be 
        aggregated to form new ones.  In most cases, this is just the set of narrowest
        beamforming vectors.
    
    num_rf_chains : int
        number of analog-to-digital or digital-to-analog converters in receiver
        or transmitter architecture.  Many applications assume a 1-to-1 ratio
        of possible simultaneous users to the number of RF chains, hence we fix
        the number of number of data streams to 1 (next parameter).
        
    num_data_streams : int
        number of data streams (simultaneous users) of the channel.
        
    Returns
    --------
    F_bb : numpy-ndarray of complex float
        Matrix of complex floats of dimension num_data_streams X num_rf_chains
        which is the baseband precoding (decoding) digital matrix.
    
    F_rf : numpy-ndarray of complex float
        Matrix of complex floats of dimension num_rf_chains X M (number of elements)
        which is the RF analog beamforming set of weights.
    
    F : numpy-ndarray of complex float
        Hybrid beamforming vector, which is the product of F_bb@F_rf
    """
    F_rf = []
    F_res = dcp(F_opt)
    for ii in range(num_rf_chains):
        Psi = np.conj(A_can.T)@F_res
        Psi_mat = np.real(np.outer(Psi, np.conj(Psi.T)))
        k = np.argmax(np.max(Psi_mat,axis = 0))

        F_rf.append(A_can[:,k])
        F_rf_app = np.vstack(F_rf).T
        F_bb = inv(np.conj(F_rf_app.T)@F_rf_app)@np.conj(F_rf_app.T)@F_opt
        if num_data_streams ==1:
            F_res = (F_opt-F_rf_app@F_bb)/np.linalg.norm(F_opt-F_rf_app@F_bb)
        else:
            F_res = (F_opt-F_rf_app@F_bb)/np.linalg.norm(F_opt-F_rf_app@F_bb, 'fro')
    F_bb = np.sqrt(num_data_streams) * F_bb / np.linalg.norm(F_rf_app@F_bb)
    
    return F_bb, F_rf, F_bb@F_rf

    
def quantize(X,M,Nq=3):
    '''
    Description
    -----------
    Quantizes with respect to nearest complex numbered value.  Relevant to 
    coarse analog phase shifters that may have finite granularity to phase values.
    
    Parameters
    ----------
    X : numpy ndarray
        matrix of complex floats which we loop through and quantize each element.
        
    M : int
        Number of antenna elements in array, important due to element normalization.
        Not used in codebook constructions.
    
    Nq : int
        Number of quantization levels, i.e. 2**Nq.
    
    Returns
    -------
    Xq : numpy ndarray
        quantized matrix of complex floats which we looped through and quantized each element.
    '''
    phz_bins = 1/np.sqrt(M) * np.exp(1j * 2* np.pi *np.arange(int(2**Nq)) /int(2**Nq))
    Xq = []
    for row in X:
        Xq_row = []
        for element in row:
            k = np.argmin(np.abs(phz_bins - element))
            Xq_row.append(phz_bins[k])
        Xq.append(np.array(Xq_row))
        
    return np.vstack(Xq)

