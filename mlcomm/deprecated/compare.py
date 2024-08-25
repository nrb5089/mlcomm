#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:34:23 2019

@author: nate
"""

import numpy as np
import matplotlib.pyplot as plt
import beams
import hba
import ucb_sense
import uba
import agile_link_mab

plt.close('all')



'''
This is a simple script designed to show comparison between different MAB methods
'''
T = 2000
channel = beams.Beams()


normal_agent = ucb_sense.Agent(channel.sample_signal,channel,mode_init = 'transmit')
normal_agent.run_sim1(T)

hierarchical_agent = hba.hba(channel.sample_signal,channel)
hierarchical_agent.run_sim(T = T)

sensing_agent = ucb_sense.Agent(channel.sample_signal,channel)
sensing_agent.run_sim4(T) #scaled MABs

uba_agent = uba.uba(channel.sample_signal,channel)
uba_agent.run_sim(T)

al_agent = agile_link_mab.AgileLink(channel.sample_signal,channel,num_hashings = 4)
al_agent.run_sim(T)

plt.figure(0)
plt.plot(np.cumsum(np.sum(normal_agent.r,1)))
plt.plot(np.cumsum(hierarchical_agent.r))
plt.plot(np.cumsum(np.sum(sensing_agent.r,)))
plt.plot(np.cumsum(uba_agent.r))
plt.plot(np.cumsum(al_agent.rew))
plt.title('Cumulative Rewards')
plt.xlabel('Time')
plt.ylabel('Rewards')
plt.grid(axis = 'both')
plt.legend(['Normal','HBA','Sensing-Scaling','UBA','AL'])

plt.figure(1)
plt.semilogy(np.cumsum(normal_agent.R))   
plt.semilogy(np.cumsum(hierarchical_agent.R)) 
plt.semilogy(np.cumsum(sensing_agent.R))
plt.semilogy(np.cumsum(uba_agent.R))
plt.semilogy(np.cumsum(al_agent.Reg))
plt.semilogy(np.arange(0,T),'k--')
plt.title('Cumulative Regret')
plt.xlabel('Time')
plt.ylabel('Rewards')
plt.grid(axis = 'both')
plt.legend(['Normal','HBA','Sensing-Scaling','UBA','AL'])

    