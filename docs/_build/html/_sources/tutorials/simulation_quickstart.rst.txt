======================
Simulation Quickstart
======================

In order to run simulations with ``mlcomm``, three components are required:

1. A codebook
2. A channel
3. An algorithm

An assortment of each are stored in the respective modules ``codebooks``, ``channels``, ``algorithms``.  First, create your codebook, this can often be a bottleneck if your rebuilding it every algorithm iteration.  We recommend saving it off.

.. code-block:: python

    import mlcomm as mlc
    from mlcomm import codebooks as cb
    cb_graph = cb.BinaryHierarchicalCodebook({'depth':6, 'num_elements' : 64, 'num_rf_chains' : 32, 'num_data_streams' : 1, 'min_max_angles_degs' : (30,150)})
    cb.save_codebook(cb_graph, filename='my_codebook',savepath = './')

The codebook graph ``cb_graph`` is the key object that stores everything about your codebook, and will also help track observations and other statistics in Multi-Armed Bandit (MAB) algorithms, for example.  Next, create your channel.

.. code-block:: python

    import numpy as np
    from mlcomm import channels
    
    NUM_PATHS = 5
    SNR = 20 #in dB

    aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1]) * 180/np.pi

    mychannel = channels.RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : 0})
    
This generates one instance of the channel, in evaluating algorithms, you may want to specify several seed instances along with different angles or Signal-to-Noise Ratio (SNR).  We now instantiate an algorithm instance of ``HOSUB``.

.. code-block:: python

    from mlcomm import algorithms
    bandit = HOSUB({'cb_graph' : cb_graph, 'channel' : mychannel, 'time_horizon' : 150, 'starting_level' : 2, 'c' : 1, 'delta' : .01})
    
To run the algorithm, call the ``HOSUB`` class method ``bandit.run_alg()``.  All algorithms have an equivalent method to do this, and are based on the parent class ``AlgorithmTemplate``.  All algorithms have a attribute dictionary ``log_data`` with key-values that describe the algorithm performance.

.. code-block:: python

    bandit.run_alg()

After the algorithm runs, we can report out the results with the function below that takes the algorithm instance ``bandit`` as an argument.

.. code-block:: python

    def report_bai_result(bandit):
    """
    Description
    -----------
    Prints several outputs of the resultant simulation.

    Parameters
    ----------
    bandit : object
        Object corresponding to best arm identification algorithm post simulation.

    """
    log_data = bandit.log_data
    print(f'Estimated Best Node midx: {log_data["path"][-1]} after {np.sum(log_data["samples"])} samples')
    print(f'Actual Best Node midx: {bandit.best_midx}')
    print(f'Resultant Relative Spectral Efficiency: {log_data["relative_spectral_efficiency"][-1]}')
    print('\n')

The full code is shown below, assuming you've saved your codebook as recommended, this should run without issue!

.. code-block:: python
    
    import numpy as np
    import mlcomm as mlc
    
    NUM_PATHS = 5
    SNR = 20 #in dB
    
    def hosub_multi_run():
        for seed in np.arange(100):
            if seed == 0: print(f'Initialized RNG in main loop. Seed = {seed}')
            np.random.seed(seed = seed)
            cb_graph = mlc.codebooks.load_codebook(filename='mycodebook', loadpath='./')
            aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0],cb_graph.min_max_angles[1]) * 180/np.pi
            
            #Channel Option
            mychannel = mlc.channels.RicianAR1({'num_elements' : cb_graph.M, 'angle_degs' : aoa_aod_degs, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : NUM_PATHS, 'snr' : SNR, 'seed' : seed})
            
            bandit = mlc.algorithms.HOSUB({'cb_graph' : cb_graph, 'channel' : mychannel, 'time_horizon' : 150, 'starting_level' : 2, 'c' : 1, 'delta' : .01})
            bandit.run_alg()
            report_bai_result(bandit)
            
    hosub_multi_run()
    
The output should look like 

.. code-block:: console

    ...
    
    Estimated Best Node midx: 71 after 150 samples
    Actual Best Node midx: 71
    Resultant Relative Spectral Efficiency: 1.0


    Estimated Best Node midx: 36 after 150 samples
    Actual Best Node midx: 36
    Resultant Relative Spectral Efficiency: 1.0


    Estimated Best Node midx: 118 after 150 samples
    Actual Best Node midx: 118
    Resultant Relative Spectral Efficiency: 1.0


    Estimated Best Node midx: 49 after 150 samples
    Actual Best Node midx: 49
    Resultant Relative Spectral Efficiency: 1.0


    Estimated Best Node midx: 25 after 150 samples
    Actual Best Node midx: 107
    Resultant Relative Spectral Efficiency: 0.8954281962709966


    Estimated Best Node midx: 76 after 150 samples
    Actual Best Node midx: 76
    Resultant Relative Spectral Efficiency: 1.0


    Estimated Best Node midx: 75 after 150 samples
    Actual Best Node midx: 75
    Resultant Relative Spectral Efficiency: 1.0
    
    ...
    
Additional algorithm prototypes for quickstart, including this one, are in the ``mlcomm/tests/algorithms_test.py`` module.