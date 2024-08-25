===================================
Algorithm Prototypes and Templates
===================================

All algorithms are based on the parent class ``AlgorithmTemplate``

.. code-block:: python
    
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
                             'flops' : []
                             }

We see the above has some useful attributes declared for referencing or adding to later on.  The method ``self.set_best()``, in particular,

.. code-block:: python

    def set_best(self):
        """
        Description
        ------------
        Sets the attribute best_midx, which is the midx belonging to the node with the highest mean reward.
        """
        self.best_midx = np.argmax([self.sample(node,with_noise=False) for node in self.cb_graph.nodes.values()])
        self.best_node = self.cb_graph.nodes[self.best_midx]

which fetches the best ``midx`` corresponding to the beamforming vector with the highest mean reward under these channel conditions and beamforming codebook.  We actually apply the beamforming vector and take a measurement using

.. code-block:: python

    def sample(self, node, transmit_power_dbw = 1, with_noise=True, mode='rss'): 
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
        transmit_power_dbw : float
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
            return np.abs(np.conj(node.f).T @ self.channel.array_response(transmit_power_dbw = transmit_power_dbw,with_noise=with_noise))**2
        elif mode == 'complex':
            return np.conj(node.f).T @ self.channel.array_response(with_noise=with_noise)

We also are frequently interested in evaluating the performance of an algorithm using the relative spectral efficiency.  We provide a method to handle this, where the quantity calculated is relative to the beamforming vector fetched by ``self.get_best()``

.. code-block:: python

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
        
Your custom algorithms can more quickly integrate into the ``mlcomm`` framework by creating child classes for your algorithm:

.. code-block:: python

    import mlcomm as mlc
    from mlcomm.algorithms import AlgorithmTemplate
    
    MyAlgorithm(AlgorithmTemplate)
        def __init__(self,params):
            super().__init__(params)
            self.param1 = params['param1']
            
            for node in self.cb_graph.values():
                node.mean_reward = 0.0
                node.num_pulls = 0.0
                
                #...
            #...
            #Rest of __init__ function
            #...
        
        def run_alg(self,*args,**kwargs):
            nodes = cb_graph.nodes
            
            #...
            #Algorithm execution goes here
            #...
            
            node2sample = self.pick_node_to_sample()
            r = self.sample(node2sample)
            self.update_node(r,node2sample)
            self.channel.fluctuation(nn,self.cb_graph.min_max_angles)
            
        def update_node(self,reward_observed,node):
            node.num_pulls += 1.0
            node.mean_reward = ((node.num_pulls-1) * node.mean_reward + reward_observed) / node.num_pulls
            #...
        
        def pick_node_to_sample(self):
            #Returns codebook object beloging to node/vertex 
            #...
            return node2sample
            
        def helper_method1(self):
            self.param1 = 10 * self.param1
           #...