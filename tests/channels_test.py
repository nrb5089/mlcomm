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
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import deepcopy as dcp
from channels import *
from util import * 


def main():
    init_figs()
    #Example Designs for Various channels
    #mychannel = RicianAR1({'num_elements' : 64, 'angle_degs' : 90, 'fading_1' : 1, 'fading_2' : 10, 'correlation' : 0.024451, 'num_paths' : 5, 'snr' : 0, 'seed' : 0})
    #mychannel = NYU1({'num_elements' : 64, 'angle_degs' : 90, 'environment' : 'LOS', 'scenario' : 'Rural', 'center_frequency_ghz' : 28, 'propagation_distance' : 50, 'noise_variance' : 1e-10, 'seed' : 0})
    # mychannel = NYU2({'num_elements' : 64, 'scenario' : 'RMa', 'center_frequency_ghz' : 28, 'initial_me_position' : (20,50,20), 'initial_me_velocity' : (-1,1,0), 'sigma_u_m_s2' : .05,'correlation_distance' : 30, 'noise_variance' : 1e-10, 'seed' : 1, 'map_generation_mode' : 'load', 'map_index' : 1, 'maps_dir' : './'})
    #view_los_map(mychannel)
    #test_fluctuations_nyu2()
    # view_nyu2_channel()
    dynamic_motion_demo()
    
def test_fluctuations_nyu2():
    mychannel = NYU2({'num_elements' : 64, 'scenario' : 'RMa', 'center_frequency_ghz' : 28, 'initial_me_position' : (20,50,20), 'initial_me_velocity' : (-1,1,0), 'sigma_u_m_s2' : .05,'correlation_distance' : 30, 'noise_variance' : 1e-10, 'seed' : 0, 'map_generation_mode' : 'load', 'map_index' : 0, 'maps_dir' : './'})
    print("Channel initialized, beginning fluctuations...")
    num_flucs = 50
    positions_x,positions_y = [], []
    az_aods,el_aods,az_aoas,el_aoas = [dcp(mychannel.az_aod)],[dcp(mychannel.el_aod)],[dcp(mychannel.az_aoa)],[dcp(mychannel.el_aoa)]
    Pis = [dcp(mychannel.Pi)]
    fig_view,ax_view = plt.subplots()
    for ii in range(num_flucs):
        if np.mod(ii+1,10) == 0: print(f"fluctuation {ii+1} of {num_flucs}")
        mychannel.fluctuation()
        az_aods.append(dcp(mychannel.az_aod))
        el_aods.append(dcp(mychannel.el_aod))
        az_aoas.append(dcp(mychannel.az_aoa))
        el_aoas.append(dcp(mychannel.el_aoa))
        Pis.append(dcp(mychannel.Pi))
        positions_x.append(mychannel.me_position[0])
        positions_y.append(mychannel.me_position[1])
        
        rsss = []
        angles = np.linspace(0,2*np.pi,2048)
        for angle in angles:
            w = avec(angle,mychannel.M)
            rsss.append(np.abs(np.conj(w) @ mychannel.ht)**2)
        ax_view.plot(angles*180/np.pi,10*np.log10(rsss))
    fig,ax = plt.subplots()
    ax.plot(positions_x,positions_y,'.')
    ax.set_ylim([-200,200])
    ax.set_xlim([-200,200])
    ax.set_title('ME Positions')
    
    fig,axes = plt.subplots(4,1)
    fig_p,axes_p = plt.subplots()
    for ii in range(mychannel.total_num_paths):
        az_aods_plt = [az_aods[jj][ii] * 180/np.pi for jj in range(num_flucs+1)]
        if ii ==0:
            axes[0].plot(az_aods_plt,label = f"{ii}",linewidth = 5)
        else:
            axes[0].plot(az_aods_plt,label = f"{ii}")
        
        el_aods_plt = [el_aods[jj][ii] * 180/np.pi for jj in range(num_flucs+1)]
        axes[1].plot(el_aods_plt,label = f"{ii}")
        
        az_aoas_plt = [az_aoas[jj][ii] * 180/np.pi for jj in range(num_flucs+1)]
        axes[2].plot(az_aoas_plt,label = f"{ii}")
        
        el_aoas_plt = [el_aoas[jj][ii] * 180/np.pi for jj in range(num_flucs+1)]
        axes[3].plot(el_aoas_plt,label = f"{ii}")
        
        Pis_plt = [10*np.log10(Pis[jj][ii]) for jj in range(num_flucs+1)]
        if ii ==0:
            axes_p.plot(Pis_plt,linewidth = 5)
        else:
            axes_p.plot(Pis_plt)
        
    view_los_map(mychannel)
    #mychannel = NYU_preset({'num_elements' : 64, 'set_number' : 1, 'scenario' : 'RMa', 'profile' : 'Hexagon', 'noise_variance' : 1e-10, 'seed' : 0})
    #dynamic_motion_demo()
    return 0

def view_nyu2_channel():
    mychannel = NYU2({'num_elements' : 64, 'scenario' : 'RMa', 'center_frequency_ghz' : 28, 'initial_me_position' : (20,50,20), 'initial_me_velocity' : (-1,1,0), 'sigma_u_m_s2' : .05,'correlation_distance' : 30, 'noise_variance' : 1e-10, 'seed' : 1, 'map_generation_mode' : 'load', 'map_index' : 1, 'maps_dir' : './'})
    rsss = []
    angles = np.linspace(0,2*np.pi,2048)
    ht = mychannel.ht 
    for angle in angles:
        w = avec(angle,mychannel.M)
        rsss.append(np.abs(np.conj(w) @ ht)**2)
    fig,ax = plt.subplots()
    ax.plot(angles*180/np.pi,10*np.log10(rsss))

def dynamic_motion_demo():
    
    mychannel = DynamicMotion({'num_elements' : 64, 'sigma_u_degs' : .001, 'initial_angle_degs' : 90,  'fading' : .995, 'time_step': 1, 'num_paths' : 5, 'snr' : 0, 'mode' : 'GaussianJumps', 'seed' : 0})
    
    angles = []
    for nn in np.arange(2000):
        angles.append([mychannel.angles[ll] * 180/np.pi for ll in range(mychannel.L)])
        mychannel.fluctuation(nn,(np.pi/6,5*np.pi/6))
    
    fig,ax = plt.subplots()
    ax.plot(angles)
    ax.set_xlabel('n')
    ax.set_ylabel('Angle (Degrees)')
    plt.show()

def view_los_map(channel):
    los_map = channel.los_map
    sf_los_map = channel.sf_los_map
    sf_nlos_map = channel.sf_nlos_map
    
    fig,ax = plt.subplots()
    ax.imshow(los_map)
    tick_marks = [-200,-100,0,100,200]
    ax.set_xticks([0,100,200,300,400])
    ax.set_yticks([0,100,200,300,400][-1::-1])
    ax.set_xticklabels(tick_marks)
    ax.set_yticklabels(tick_marks)
    
    fig,ax = plt.subplots()
    ax.imshow(sf_los_map)
    tick_marks = [-200,-100,0,100,200]
    ax.set_xticks([0,100,200,300,400])
    ax.set_yticks([0,100,200,300,400][-1::-1])
    ax.set_xticklabels(tick_marks)
    ax.set_yticklabels(tick_marks)
    
    fig,ax = plt.subplots()
    ax.imshow(sf_nlos_map)
    tick_marks = [-200,-100,0,100,200]
    ax.set_xticks([0,100,200,300,400])
    ax.set_yticks([0,100,200,300,400][-1::-1])
    ax.set_xticklabels(tick_marks)
    ax.set_yticklabels(tick_marks)
    
    # ax.xaxis.set_major_locator(MaxNLocator(5))
    # ax.yaxis.set_major_locator(MaxNLocator(5))
    
if __name__ == '__main__':
    main()
