# mlcomm package

## Package description

Package providing algorithms to test machine learnig algorithms in the context of wireless communications. Current focus is mostly on Multi-Armed Bandit type algorithms.

The package offers the following submodules targetted at Uniform Linear Array (ULA) beamforming channels.

- mlcomm/codebooks
    - Binary Hierarchical Beamforming Codebook
    - Ternary Hierarchical Beamforming Codebook
    
- mlcomm/algorithms: multi-arm bandit-based algorithms including
    - Hierarchical Optimal Sampling for Unimodal bandits (HOSUB) [REF](https://doi.org/10.1109/ICC42927.2021.9500373)
    - Dynamic Beam Zooming (DBZ) [REF]   
    - Hierarchical Posterior Matching (HPM) [REF](https://doi.org/10.1109/JSAC.2019.2933967)
    - Active Beam Tracking (ABT) [REF](https://doi.org/10.1109/ICC42927.2021.9500601)
    - Two-Phase Heteroscedastic Track-and-Stop (2PHTS) [REF](https://doi.org/10.1109/TWC.2022.3217131)
        
- mlcomm/channels
    - Additive White Gaussian Noise (AWGN) 
    - Rician AR1 [REF](https://doi.org/10.1109/JSAC.2019.2933967)
    - Dynamic Motion [REF]
    - NYU Sim (preset) [REF](https://doi.org/10.1109/ICC.2017.7996792)
    

## Documentation

To view full documentation locally for installation, module description, and usage.  

```
git clone https://github.gatech.edu/mb364/mlcomm.git
cd docs
make html
```

Open ```_build\html\index.html``` in any browser.