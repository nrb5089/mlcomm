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
from mlcomm.codebooks import *
from mlcomm.util import * 

def main():
    plt.close('all')
    init_figs()

    #cb_graph = load_codebook(filename='demo_binary_codebook', loadpath='./')
    cb_graph = load_codebook(filename='demo_ternary_codebook', loadpath='./')
    
    # fig = show_level(cb_graph,3)
    h = 0
    i = 100
    
    
    
    # fig = show_level_base_set(cb_graph,h)
    # fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\base_set_binary_'+ f'{h}')
    # fig = show_zoom_out(cb_graph,i)
    # fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\zoom_out_binary_' + f'{i}')
    # fig = show_zoom_in(cb_graph,0)
    # fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\zoom_in_binary_' + f'{i}')
    
    fig = show_level_base_set(cb_graph,h)
    fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\base_set_ternary_'+ f'{h}')
    fig = show_zoom_out(cb_graph,i)
    fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\zoom_out_ternary_' + f'{i}')
    # fig = show_zoom_in(cb_graph,i)
    # fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\zoom_in_ternary_' + f'{i}')
    fig = show_subset(cb_graph,[(0,100),(1,100),(2,100),(3,100),(0,109),(1,109),(2,109),(3,109),(0,30),(1,30),(2,30),(3,30),(0,25),(1,25),(2,25),(3,25)  ])
    fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\overlap_ternary_1')
    
    # fig = show_subset(cb_graph,[(0,30),(1,30),(2,30),(3,30),(0,25),(1,25),(2,25),(3,25) ])
    # fig.savefig(r'D:\OneDrive\Projects\mlcomm\mlcomm-private\mlcomm\docs\tutorials\media\overlap_ternary_1')
    plt.show()
    

def build_hosub_codebook():
    """
    Constructs and saves codebook from HOSUB paper [1].
    
    [1] Blinn, Nathan, Jana Boerger, and Matthieu Bloch. "mmWave Beam Steering with Hierarchical Optimal Sampling for Unimodal Bandits." ICC 2021-IEEE International Conference on Communications. IEEE, 2021.
    """
    cb_graph = BinaryHierarchicalCodebook({'depth':6, 'num_elements' : 64, 'num_rf_chains' : 32, 'num_data_streams' : 1, 'min_max_angles_degs' : (30,150)})
    save_codebook(cb_graph, filename='../mlcomm/demo_binary_codebook',savepath = './')
    save_codebook(cb_graph, filename='demo_binary_codebook',savepath = './')
    
def build_dbz_codebook():
    """
    Constructs and saves codebook from DBZ paper.
    """
    cb_graph = TernaryPointedHierarchicalCodebook({'num_initial_non_overlapping' : 5, 'depth' : 4, 'num_elements' : 128, 'num_rf_chains' : 32, 'num_data_streams' : 1, 'min_max_angles_degs' : (30,150)})
    save_codebook(cb_graph, filename='../mlcomm/demo_ternary_codebook',savepath = './')
    save_codebook(cb_graph, filename='demo_ternary_codebook',savepath = './')
    
    
## Plots
def show_level(cb_graph,hh,show_every_other = 1):
    """
    Description
    -----------
    Shows all beamforming vectors at a particular level.
    
    Parameters
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
        
    hh : int
        Level which user wishes to view.

    Returns
    -------
    None.

    """
    
    phi_fine = np.linspace(0,2*np.pi,4096)
    A_fine = np.vstack([avec(angle,cb_graph.M) for angle in phi_fine]).T

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = generate_rgb_gradient(len(cb_graph.level_midxs[hh]))
    for ii,midx in enumerate(cb_graph.level_midxs[hh]):
        rss = 10*np.log10(np.abs(np.conj(cb_graph.nodes[midx].f)@A_fine)**2)
        ax.plot(phi_fine,rss, linewidth = 2.5, color = colors[ii])
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi + np.pi/6, np.pi/6))
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=26, length=10, width=2)
    return fig

def show_level_base_set(cb_graph,hh):
    """
    Description
    -----------
    Shows all base set beamforming vectors at a particular level. That is, the set that minimally covers the 
    coverage region specified, and the zoom-in beamforming vectors for each successive level.
    
    Parameters
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
        
    hh : int
        Level which user wishes to view.

    Returns
    -------
    None.

    """
    phi_fine = np.linspace(0,2*np.pi,4096)
    A_fine = np.vstack([avec(angle,cb_graph.M) for angle in phi_fine]).T

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = generate_rgb_gradient(len(cb_graph.base_midxs[hh]))
    for ii,midx in enumerate(cb_graph.base_midxs[hh]):
        rss = 10*np.log10(np.abs(np.conj(cb_graph.nodes[midx].f)@A_fine)**2)
        ax.plot(phi_fine,rss, linewidth = 2.5, color = colors[ii])
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi + np.pi/6, np.pi/6))
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=26, length=10, width=2)
    return fig

def show_zoom_out(cb_graph,ii):
    """
    Description
    -----------
    Shows zoom-out beamforming vector patterns for all beamforming vectors with index i, meaning common steered angle.
    
    Parameters
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
        
    ii : int
        Index which user wishes to view.

    Returns
    -------
    None.

    """
    phi_fine = np.linspace(0,2*np.pi,4096)
    A_fine = np.vstack([avec(angle,cb_graph.M) for angle in phi_fine]).T

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = generate_rgb_gradient(cb_graph.H)
    midx = cb_graph.nodes[cb_graph.level_midxs[-1][ii]].midx
    for hh in np.arange(cb_graph.H)[-1::-1]:
        # rss = 10*np.log10(np.abs(np.conj(cb_graph.nodes[cb_graph.level_midxs[hh][ii]].f)@A_fine)**2)
        rss = 10*np.log10(np.abs(np.conj(cb_graph.nodes[midx].f)@A_fine)**2)
        midx = cb_graph.nodes[midx].zoom_out_midx
        ax.plot(phi_fine,rss, linewidth = 2.5, color = colors[hh])
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi + np.pi/6, np.pi/6))
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=26, length=10, width=2)
    return fig

def show_zoom_in(cb_graph,ii):
    """
    Description
    -----------
    Shows all zoom-in beamforming vectors for a particular beamforming vector i. That is, the set that minimally covers the 
    coverage region specified, and the zoom-in beamforming vectors for each successive level.
    
    Parameters
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
        
    ii : int
        Index which user wishes to view.

    Returns
    -------
    None.

    """
    phi_fine = np.linspace(0,2*np.pi,4096)
    A_fine = np.vstack([avec(angle,cb_graph.M) for angle in phi_fine]).T
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = generate_rgb_gradient(cb_graph.H)
    
    next_level_midxs = [cb_graph.level_midxs[0][ii]]
    current_level_midxs = dcp(next_level_midxs)
    for hh in np.arange(cb_graph.H):
        next_level_midxs = []
        for midx in current_level_midxs:
            rss = 10*np.log10(np.abs(np.conj(cb_graph.nodes[midx].f)@A_fine)**2)
            ax.plot(phi_fine,rss, linewidth = 2.5, color = colors[hh])
            next_level_midxs.extend(cb_graph.nodes[midx].zoom_in_midxs)
        current_level_midxs = dcp(next_level_midxs)
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi + np.pi/6, np.pi/6))
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=26, length=10, width=2)
    return fig


def show_subset(cb_graph,hituples):
    """
    Description
    -----------
    Shows specified set of beamforming vectors
    
    Parameters
    ----------
    cb_graph : object 
        Type instance of BinaryHierarchicalCodebook or TernaryPointedHierarchicalCodebook.  See description of class types.
    
    hituples : list or array of tuples of ints
        list of tuples correspoding to which beamforming vector the user wishes to view, i.e., (h,i) = (1,2) views h=1, i =2

    Returns
    -------
    None.

    """
    phi_fine = np.linspace(0,2*np.pi,4096)
    A_fine = np.vstack([avec(angle,cb_graph.M) for angle in phi_fine]).T
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = generate_rgb_gradient(cb_graph.H)
    
    for hituple in hituples:
        hh = hituple[0]
        midx = cb_graph.level_midxs[hh][hituple[1]]
        rss = 10*np.log10(np.abs(np.conj(cb_graph.nodes[midx].f)@A_fine)**2)
        ax.plot(phi_fine,rss, linewidth = 2.5, color = colors[hh])
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi + np.pi/6, np.pi/6))
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=26, length=10, width=2)
    return fig
if __name__ == '__main__':
    main()

