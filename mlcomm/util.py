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
import matplotlib
import matplotlib.pyplot as plt

def randcn(M):
    """
    Description
    -----------
    Generates a numpy array of complex floats representing standard circularly symmetric complex Gaussian Noise.
    
    Parameters
    -----------
    M : int
        Length of noise vector required.
        
    Returns
    -------
    noise : numpy ndarray of complex floats
        Vector representing the noise values.
    """
    # if M >1: return  1/np.sqrt(2*M) * (np.random.randn(M) + 1j* np.random.randn(M))       
    if M >1: return  1/np.sqrt(2) * (np.random.randn(M) + 1j* np.random.randn(M))       
    else: return  1/np.sqrt(2) * (np.random.randn() + 1j* np.random.randn()) 
    
def avec(angle,num_elements): 
    """
    Description
    ------------
    Generates array response for a Uniform Linear Array (ULA) centered along the x-axis.
    
    Parameters
    -----------
    angle : float
        Angle in radians of the AoA or AoD.
    
    num_elements: int
        Number of elements in the ULA.
        
    Returns
    --------
    array_response: numpy ndarray of complex float
        Instantaneous, normalized narrowband response of ULA to impining signal, or transmitted singal at designated 'angle'.
    """
    return 1/np.sqrt(num_elements) * np.exp(1j * np.pi * np.cos(angle) * (np.arange(num_elements)-(num_elements-1)/2))


## Data Processing Utilities

def pad_and_mean(X,max_len):
    '''row wise mean for vectors in a list or matrix X'''
    for ii,row in enumerate(X):
        try:
            row.extend(np.zeros(max_len-len(row)))
        except:
            try:
                X[ii] = np.concatenate([row,np.zeros(max_len-len(row))])
            except: 
                print(f'ROWLENGTH THAT FAILED: {len(row)} with maxlen {max_len}')
    return np.mean(np.vstack(X),0)

def pad_and_mean_last(X,max_len):
    '''row wise mean for vectors in a list or matrix X'''
    
    for ii,row in enumerate(X):
        last_element = row[-1]
        try:
            row.extend(last_element*np.ones(max_len-len(row)))
        except:
            try:
                X[ii] = np.concatenate([row,last_element*np.ones(max_len-len(row))])
            except: print(f'ROWLENGTH THAT FAILED: {len(row)} with maxlen {max_len}')
    return np.mean(np.vstack(X),0)

def pad_vec(x,max_len):
    try:
        x.extend(np.zeros(max_len-len(x)))
    except:
        x = np.concatenate([x,np.zeros(max_len-len(x))])
    return np.array(x)

def pad_vec_last(x,max_len):
    last_element = x[-1]
    try:
        x.extend(last_element*np.ones(max_len-len(x)))
    except:
        x = np.concatenate([x,last_element*np.ones(max_len-len(x))])
    return np.array(x)

def deg2rad(degs): return np.pi/180 * degs

def rad2deg(rads): return 180/np.pi * rads

## Plot Utilities

def generate_rgb_gradient(n_points, mode='gt'):
    # Define the key colors (red, orange, yellow, green, cyan, blue, violet), normalized between 0 and 1
    
    if mode == 'rainbow':
        key_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [1.0, 0.647, 0.0],  # Orange
            [1.0, 1.0, 0.0],  # Yellow
            [0.0, 0.5, 0.0],  # Green
            [0.0, .65, .65],  # Cyan
            [0.0, 0.0, 1.0],  # Blue
            [0.58, 0.0, 0.827]  # Violet
        ])
    
    if mode == 'gt':
        key_colors = np.array([[235,200,0],[192,192,192],[0,0,175]])/256
        if n_points < 15: 
            if n_points == 2:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.0, 0.0, 0.5]       # Navy Blue
                ]
            if n_points == 3:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.667, 0.667, 0.833], # Silver
                    [0.0, 0.0, 0.5]       # Navy Blue
                ]
            elif n_points == 4:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.833, 0.833, 0.417],  # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.0, 0.0, 0.5]       # Navy Blue
                ]
            elif n_points == 5:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.833, 0.75, 0.416],  # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.333, 0.5, 0.667],   # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 6:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.833, 0.75, 0.416],  # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.5, 0.583, 0.75],    # Intermediate between Silver and Navy Blue
                    [0.333, 0.5, 0.667],   # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 7:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.857, 0.75, 0.291],  # Intermediate between Gold and Silver
                    [0.714, 0.667, 0.583], # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.476, 0.583, 0.75],  # Intermediate between Silver and Navy Blue
                    [0.238, 0.417, 0.625], # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 8:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.875, 0.75, 0.25],   # Intermediate between Gold and Silver
                    [0.75, 0.667, 0.5],    # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.571, 0.643, 0.786], # Intermediate between Silver and Navy Blue
                    [0.429, 0.571, 0.714], # Intermediate between Silver and Navy Blue
                    [0.286, 0.5, 0.643],   # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 9:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.889, 0.75, 0.222],  # Intermediate between Gold and Silver
                    [0.778, 0.667, 0.444], # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.556, 0.667, 0.722], # Intermediate between Silver and Navy Blue
                    [0.444, 0.583, 0.722],  # Intermediate between Silver and Navy Blue
                    [0.333, 0.5, 0.667],   # Intermediate between Silver and Navy Blue
                    [0.222, 0.417, 0.611], # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 10:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.9, 0.75, 0.2],      # Intermediate between Gold and Silver
                    [0.8, 0.667, 0.4],    # Intermediate between Gold and Silver
                    [0.7, 0.667, 0.6],    # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.5, 0.667, 0.833],   # Intermediate between Silver and Navy Blue
                    [0.333, 0.667, 0.722], # Intermediate between Silver and Navy Blue
                    [0.222, 0.583, 0.611], # Intermediate between Silver and Navy Blue
                    [0.111, 0.5, 0.556],   # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 11:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.909, 0.75, 0.182], # Intermediate between Gold and Silver
                    [0.818, 0.667, 0.364], # Intermediate between Gold and Silver
                    [0.727, 0.667, 0.545], # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.5, 0.667, 0.833],   # Intermediate between Silver and Navy Blue
                    [0.364, 0.667, 0.833], # Intermediate between Silver and Navy Blue
                    [0.273, 0.583, 0.667], # Intermediate between Silver and Navy Blue
                    [0.182, 0.5, 0.556],   # Intermediate between Silver and Navy Blue
                    [0.091, 0.417, 0.556], # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 12:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.917, 0.75, 0.167], # Intermediate between Gold and Silver
                    [0.833, 0.667, 0.333], # Intermediate between Gold and Silver
                    [0.75, 0.667, 0.5],   # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.667], # Silver
                    [0.583, 0.667, 0.75],  # Intermediate between Silver and Navy Blue
                    [0.5, 0.667, 0.833],   # Intermediate between Silver and Navy Blue
                    [0.417, 0.667, 0.75],  # Intermediate between Silver and Navy Blue
                    [0.333, 0.5, 0.667],   # Intermediate between Silver and Navy Blue
                    [0.25, 0.417, 0.611],  # Intermediate between Silver and Navy Blue
                    [0.167, 0.333, 0.611], # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 13:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.923, 0.75, 0.154], # Intermediate between Gold and Silver
                    [0.846, 0.667, 0.308], # Intermediate between Gold and Silver
                    [0.769, 0.667, 0.462], # Intermediate between Gold and Silver
                    [0.692, 0.667, 0.615], # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.538, 0.667, 0.75],  # Intermediate between Silver and Navy Blue
                    [0.462, 0.667, 0.75],  # Intermediate between Silver and Navy Blue
                    [0.385, 0.583, 0.667], # Intermediate between Silver and Navy Blue
                    [0.308, 0.5, 0.611],   # Intermediate between Silver and Navy Blue
                    [0.231, 0.417, 0.611], # Intermediate between Silver and Navy Blue
                    [0.154, 0.333, 0.556], # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
            elif n_points == 14:
                return [
                    [1.0, 0.84, 0.0],    # Gold
                    [0.929, 0.75, 0.143], # Intermediate between Gold and Silver
                    [0.857, 0.667, 0.286], # Intermediate between Gold and Silver
                    [0.786, 0.667, 0.429], # Intermediate between Gold and Silver
                    [0.714, 0.667, 0.571], # Intermediate between Gold and Silver
                    [0.667, 0.667, 0.833], # Silver
                    [0.571, 0.667, 0.75],  # Intermediate between Silver and Navy Blue
                    [0.5, 0.667, 0.75],    # Intermediate between Silver and Navy Blue
                    [0.429, 0.583, 0.667], # Intermediate between Silver and Navy Blue
                    [0.357, 0.5, 0.611],   # Intermediate between Silver and Navy Blue
                    [0.286, 0.417, 0.611], # Intermediate between Silver and Navy Blue
                    [0.214, 0.333, 0.556], # Intermediate between Silver and Navy Blue
                    [0.143, 0.25, 0.556],  # Intermediate between Silver and Navy Blue
                    [0.0, 0.0, 0.5]        # Navy Blue
                ]
        else:
            n_key_colors = len(key_colors)
            gradient = np.zeros((n_points, 3))
            
            # Number of points for each segment between key colors
            segment_length = n_points // (n_key_colors - 1)
        
            for i in range(n_key_colors - 1):
                # Linear interpolation between key_colors[i] and key_colors[i + 1]
                for j in range(segment_length):
                    alpha = j / (segment_length - 1)
                    color = (1 - alpha) * key_colors[i] + alpha * key_colors[i + 1]
                    
                    # Populate the gradient array
                    gradient[i * segment_length + j, :] = color
        
            # If n_points is not divisible by (n_key_colors - 1), fill in the remaining colors
            remaining = n_points % (n_key_colors - 1)
            if remaining:
                gradient[-remaining:, :] = key_colors[-1]
        
            return gradient

def init_figs():
    plt.close('all')
    # Options to generate nice figures
    fig_width_pt = float(640.0)  # Get this from LaTeX using \showthe\column-width
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width/1.5, fig_height/2.3] #For beampattern_zoom_fig_v2 only
    fig_size = [fig_width, fig_height]
    
    params_ieee = {
        'axes.labelsize': 16,
        'font.size': 16,
        'legend.fontsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'text.usetex': True,
        # 'text.latex.preamble': '\\usepackage{sfmath}',
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'figure.figsize': fig_size,
        'axes.grid': True
    }
    
    ############## Choose parameters you like
    matplotlib.rcParams.update(params_ieee)
    return
    
def convert_seconds(seconds):
    """
    Converts a given number of seconds into hours, minutes, and seconds.

    Parameters:
    -----------
    seconds : int or float
        The total number of seconds.

    Returns:
    --------
    str
        A formatted string in the form of "HH:MM:SS".
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(secs):02} (hrs:mins:secs)"