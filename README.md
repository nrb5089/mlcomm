# mlcomm package

## Package description

Package providing algorithms to test machine learnig algorithms in the context of wireless communications. Focuses primarily on Multi-Armed Bandit type algorithms for mmWave beamforming initial alignment and tracking.  We include comparison algorithms as well.

The package offers the following submodules targeted at Uniform Linear Array (ULA) beamforming channels.  

- mlcomm/codebooks
    - Binary Hierarchical Beamforming Codebook
    - Ternary Hierarchical Beamforming Codebook
    
- mlcomm/algorithms: multi-arm bandit-based algorithms including
    - Hierarchical Optimal Sampling for Unimodal bandits (HOSUB) [REF](https://doi.org/10.1109/ICC42927.2021.9500373)
    - Dynamic Beam Zooming (DBZ) [REF](http://arxiv.org/abs/2209.02896) 
    - Hierarchical Posterior Matching (HPM) [REF](https://doi.org/10.1109/JSAC.2019.2933967)
    - Active Beam Tracking (ABT) [REF](https://doi.org/10.1109/ICC42927.2021.9500601)
    - Two-Phase Heteroscedastic Track-and-Stop (2PHTS) [REF](https://doi.org/10.1109/TWC.2022.3217131)
    - Beam Alignment and Tracking Bandit Learning (BA-T) [REF](https://doi.org/10.1109/TCOMM.2020.2988256)
    - Extended Kalman Filter (EKF) [REF] (https://doi.org/10.1109/GlobalSIP.2016.7905941)
    - Particle Filter (PF) for Beamwidth Adjustments [REF](https://doi.org/10.1109/LCOMM.2020.3022877)
    - Track-and-Stop Tracking (TST) [REF] (In Review for IEEE ICC 2025)
        
- mlcomm/channels
    - Additive White Gaussian Noise (AWGN) 
    - Rician AR1 [REF](https://doi.org/10.1109/JSAC.2019.2933967)
    - Dynamic Motion [REF](http://arxiv.org/abs/2209.02896)
    - NYU Sim (preset) [REF](https://doi.org/10.1109/ICC.2017.7996792)
        - Please contact authors for generated channel response files.


## Experiment Reproduction

In order to reproduce the experiments from [1] and [2], please follow the instructions below, run the following via terminal or command line.  Simulations used Python version 3.12.4.

1. If you have not already cloned the directory, ```git clone https://github.com/nrb5089/mlcomm.git ```.
2. ```cd mlcomm/tests```, Running in this directory will ensure the codebook objects are placed correctly.
3. Run ```python codebooks_test.py``` to create the respective ternary and binary codebooks. This will create two ```.pkl``` files that are the codebook graph objects in the simulations.
    - These files are larger than Github's size limit of 25 MB.

### DBZ Experiments

1. Move your working directory to ```mlcomm/dbz_experiments```.
    - From ```mlcomm/tests```, type ```cd ../dbz_experiments```
2. For DBZ, performance over different severity of ME motion and SNR can be simulated using the script ```python dbz_ia_local.py``` and ```python dbz_dm_local.py```, for initial alignment and tracking, respectively.
    - Simulations attempt to utilize multiple processors on the host machine.
3. Data is logged in ```data/ia``` or ```data/dm```, respectively.

### TST Experiments

1. Move your working directory to ```mlcomm/tst_experiments```.
    - From ```mlcomm/tests```, type ```cd ../tst_experiments```
2. For TST, performance over different severity of ME motion and SNR can be simulated using the script ```python tst_ia_local.py``` and ```python tst_dm_local.py```, for initial alignment and tracking, respectively.
    - Simulations attempt to utilize multiple processors on the host machine.
3. Data is logged in ```data/ia``` or ```data/dm```, respectively.

### HPM/ABT Experiments

1. Move your working directory to ```mlcomm/hpm_abt_experiments```.
    - From ```mlcomm/tests```, type ```cd ../hpm_abt_experiments```
2. For TST, performance over different severity of ME motion and SNR can be simulated using the script ```python hpm_ia_local.py``` and ```python abt_dm_local.py```, for initial alignment and tracking, respectively.
    - Simulations attempt to utilize multiple processors on the host machine.
3. Data is logged in ```data/ia``` or ```data/dm```, respectively.



### Other Algorithms

The above steps may be used as well to run various algorithms for comparison in [1], use directories ```<alg>_experiments``` and run ```<alg>_dm_local.py```.


- [1] N. Blinn and M. Bloch, “Multi-armed bandit dynamic beam zoomingfor mmwave alignment and tracking,” 2024. [Online]. Available: https://arxiv.org/abs/2209.02896
- [2] N. Blinn and M. Bloch, "Track-and-Stop for Initial Alignment and Tracking," 2024, (In Review) 

## Documentation

Full documentation available at readthedocs [here](https://mlcomm.readthedocs.io/en/latest/)

To view full documentation locally for installation, module description, and usage.  

```
git clone https://github.com/nrb5089/mlcomm.git
cd docs
make html
```

Open ```_build\html\index.html``` in any browser.


