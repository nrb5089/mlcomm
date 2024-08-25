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

import sys
sys.path.insert(0,'../mlcomm')
import numpy as np
import matplotlib.pyplot as plt
from mlcomm.channels import *
from mlcomm.util import *

def main():
    
    #Example Designs for Various channels
    #mychannel = RicianAR1({'num_elements' : 64, 'angle_degs' : 90, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : 5, 'snr' : 0, 'seed' : 0})
    #mychannel = NYU1({'num_elements' : 64, 'angle_degs' : 90, 'environment' : 'LOS', 'scenario' : 'Rural', 'center_frequency_ghz' : 28, 'propagation_distance' : 50, 'noise_variance' : 1e-10, 'seed' : 0})
    #mychannel = NYU2({'num_elements' : 64, 'angle_degs' : 90, 'environment' : 'LOS', 'scenario' : 'RMa', 'center_frequency_ghz' : 28, 'propagation_distance' : 50, 'correlation_distance' : 20, 'initial_me_position' : (50,30,2),'noise_variance' : 1e-10, 'spatial_coherence_on' : False,'seed' : 0})
    mychannel = NYU_preset({'num_elements' : 64, 'set_number' : 1, 'scenario' : 'RMa', 'profile' : 'Hexagon', 'noise_variance' : 1e-10, 'seed' : 0})
    dynamic_motion_demo()
    return 0

def dynamic_motion_demo():
    
    mychannel = DynamicMotion({'num_elements' : 64, 'sigma_u_degs' : .001, 'initial_angle_degs' : 90,  'fading' : .995, 'time_step': 1, 'num_paths' : 5, 'snr' : 0, 'mode' : 'WGNA', 'seed' : 0})
    
    angles = []
    for nn in np.arange(2000):
        angles.append(mychannel.angles[0] * 180/np.pi)
        mychannel.fluctuation()
    fig,ax = plt.subplots()
    ax.plot(angles)
    ax.set_xlabel('n')
    ax.set_ylabel('Angle (Degrees)')
    plt.show()
    
if __name__ == '__main__':
    main()
