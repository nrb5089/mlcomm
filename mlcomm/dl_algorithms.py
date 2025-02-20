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
import os

USE_CPU_ONLY = True
if USE_CPU_ONLY: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import tensorflow as tf
from tensorflow import keras
from copy import deepcopy as dcp
import pickle
import sys
sys.path.insert(0,'../tests/')

from algorithms import *
from codebooks import *
from channels import *
from util import *



class LSTMTracking(AlgorithmTemplate):
    """
    LSTM-Based Angle Tracking Algorithm.

    This class implements an angle tracking algorithm using a Long Short-Term Memory (LSTM) 
    neural network. It follows the method proposed in [1] for beam tracking in mmWave systems.

    References:
    -----------
    [1] Burghal, Daoud, Naveed A. Abbasi, and Andreas F. Molisch. 
        "A machine learning solution for beam tracking in mmWave systems." 
        2019 53rd Asilomar Conference on Signals, Systems, and Computers. IEEE, 2019.

    Attributes:
    -----------
    cb_graph : object
        The codebook graph associated with the simulation.
    model : keras.Model
        The LSTM-based angle prediction model.
    steered_angles : numpy.ndarray
        Array of beamforming angles available in the codebook graph.
    outage_recovery : boolean
        Uses oracle-like intervention to correct beam steering during outage
        
    Methods:
    --------
    __init__(params):
        Initializes the LSTMTracking algorithm with the given parameters.
    initialize_model():
        Builds and compiles the LSTM-based angle prediction model.
    train(weights_save_file_name):
        Trains the model using supervised learning and periodically saves weights.
    load_model(angle_error=False):
        Loads the trained model weights from a specified file.
    run_alg(time_horizon):
        Runs the tracking algorithm over a specified time horizon.
    log_and_update_channel(nn):
        Updates the logs and simulates channel fluctuations.
    """

    def __init__(self, params):
        """
        Initializes the LSTMTracking algorithm with the provided parameters.
    
        Parameters:
        -----------
        params : dict
            A dictionary containing:
            - 'cb_graph' : object
                The codebook graph used in the simulation.
            - 'channel' : object
                The communication channel instance, dynamically replaced during training.
        """
        super().__init__(params)
        self.steered_angles = np.array([self.cb_graph.nodes[midx].steered_angle for midx in self.cb_graph.level_midxs[-1]])
        self.outage_recovery = params['outage_recovery']
        
    def build_model(self):
        """
        Builds and the LSTM-based angle prediction model.
    
        The model consists of:
        - A fully connected layer (20 neurons, linear activation).
        - An LSTM layer (40 units) with `return_sequences=True`.
        - Another fully connected layer (20 neurons, linear activation).
        - A final output layer (1 neuron, linear activation) for angle prediction.
    
        Returns:
        --------
        None
        """
        self.model = keras.Sequential([
            keras.layers.Input(shape=(1, 3)),  # (batch_size, time_steps=1, features)
            keras.layers.Dense(20, activation="linear"),
            keras.layers.LSTM(40, return_sequences=True),
            keras.layers.Dense(20, activation="linear"),
            keras.layers.Dense(1, activation="linear")  # Output: predicted angle
        ])
        
    def initialize_model(self):
        """
        Compiles the LSTM-based angle prediction model for training.
    
        Returns:
        --------
        None
        """
        def cosine_loss(y_true, y_pred):
            """Computes 1/2 * (1 - cos(true_angle - predicted_angle))"""
            diff = y_true - y_pred
            return 0.5 * (1 - tf.math.cos(diff))
    
        self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=cosine_loss)
    
    def train(self, weights_save_file_name):
        """
        Trains the LSTM model using supervised learning.
    
        The training data consists of simulated AoA (Angle of Arrival) estimates and 
        complex channel samples. The model is updated every batch, and weights are 
        saved periodically.
    
        Parameters:
        -----------
        weights_save_file_name : str
            The filename where the model weights will be saved periodically.
        """
        cb_graph = self.cb_graph
        self.initialize_model()

        BATCH_SIZE = 2000
        num_episodes = 60000
        seed = -1
        sigma_angle = np.pi / 180  # Small angle variation

        for episode in range(num_episodes):
            seed += 1
            np.random.seed(seed)
            sigma_u = np.random.choice([.0001, .00025, .0005, .001])
            snr = np.random.choice(np.arange(10, 51, 5))
            inputs = []
            labels = []

            # Generate training set
            aoa_aod_degs = np.random.uniform(cb_graph.min_max_angles[0], cb_graph.min_max_angles[1]) * 180 / np.pi
            channel = DynamicMotion({
                'num_elements': cb_graph.M,
                'sigma_u_degs': sigma_u,
                'initial_angle_degs': aoa_aod_degs,
                'fading': 0.995,
                'time_step': 1,
                'num_paths': NUM_PATHS,
                'snr': snr,
                'scenario': 'LOS',
                'mode': 'WGNA',
                'seed': seed
            })
            self.channel = channel

            for _ in range(BATCH_SIZE):
                node_to_sample = cb_graph.nodes[cb_graph.level_midxs[-1][np.argmin(np.abs(self.channel.angles[0] + sigma_angle * np.random.randn() - self.steered_angles))]]
                y = self.sample(node_to_sample, mode='complex')
                inputs.append(np.array([np.real(y), np.imag(y), dcp(self.channel.angles[0])]))
                self.channel.fluctuation()
                labels.append(dcp(self.channel.angles[0]))

            labels = np.array(labels)
            inputs = np.expand_dims(inputs, axis=1)

            # Train the model
            history = self.model.fit(inputs, labels, batch_size=BATCH_SIZE, epochs=1)

            if episode % 100 == 0:
                print(f"Weights saved after episode: {episode}")
                self.model.save_weights(weights_save_file_name)

    def load_model(self, angle_error=True):
        """
        Loads the trained model weights.

        Parameters:
        -----------
        angle_error : bool, optional
            If True, loads weights from a model that includes angle error compensation during training.

        Notes:
        ------
        - Training was conducted over 60,000 episodes.
        - The model was trained on SNR values randomly selected from [10, 15, ..., 50].
        - Motion perturbation values were randomly selected from [.0001, .00025, .0005, .001].
        - The model uses `return_sequences=True` to capture past information.
        """
        self.build_model()
        weights_file = "model_weights_batch_16FEB_return_seq_no_act_angle_error.weights.h5" if angle_error else "model_weights_batch_16FEB_return_seq_no_act.weights.h5"
        self.model.load_weights(weights_file)

    def run_alg(self, time_horizon):
        """
        Executes the tracking algorithm over a specified time horizon.

        Parameters:
        -----------
        time_horizon : int
            The number of time steps for which the algorithm will run.
        """
        self.load_model()
        cb_graph = self.cb_graph

        # Begins with perfect estimate
        estimated_angle = dcp(self.channel.angles[0])  # Initial estimate
        self.comm_node = cb_graph.nodes[cb_graph.level_midxs[-1][np.argmin(np.abs(estimated_angle - self.steered_angles))]]

        for nn in range(time_horizon):
            
            #Outage Detection
            if np.abs(estimated_angle - self.channel.angles[0]) <= 2/3 * (self.cb_graph.min_max_angles[1]-self.cb_graph.min_max_angles[0]) / self.cb_graph.M:
                y = self.sample(self.comm_node, mode='complex')
                input_data = np.array([[[np.real(y), np.imag(y), dcp(estimated_angle)]]])
                predicted_angle = self.model.predict(input_data, verbose=0).flatten()[0]
                self.log_and_update_channel(nn)
            else:
                #Case of Outage Detection, uses oracle-like beam recovery in one time step.
                self.log_and_update_channel(nn)
                predicted_angle = dcp(self.channel.angles[0])
            
            estimated_angle = dcp(predicted_angle)
            self.comm_node = cb_graph.nodes[cb_graph.level_midxs[-1][np.argmin(np.abs(estimated_angle - self.steered_angles))]]

    def log_and_update_channel(self, nn):
        """
        Logs results and updates the communication channel.

        Parameters:
        -----------
        nn : int
            Current time step in the simulation.
        """
        if self.comm_node is None:
            self.log_data['relative_spectral_efficiency'].append(0.0)
            self.log_data['path'].append(np.nan)
        else:
            self.log_data['relative_spectral_efficiency'].append(self.calculate_relative_spectral_efficiency(self.comm_node))
            self.log_data['path'].append(self.comm_node.midx)
            self.channel.fluctuation(nn, (self.cb_graph.min_max_angles[0], self.cb_graph.min_max_angles[1]))
            self.set_best()

