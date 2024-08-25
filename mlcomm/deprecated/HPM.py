"""
@author: Jana


Implementation of Hierarchical Posterior Matching Algorithm from
Chiu et al.: Active Learning and CSI Acquisition for mmWave Initial Alignment

"""


import matplotlib.pyplot as plt
import beams
import math
import numpy as np
from array_play_codebooks import create_codebook


def pi_d_l_k(pi, l, k, n):  # n is target resolution
    numb_in_slices = (n / (2 ** l))
    interval_start = numb_in_slices * k
    d_sum = sum(pi[int(interval_start):int(interval_start + numb_in_slices - 1)])
    return d_sum


class HPM:

    def __init__(self,
                 eps=0.01,
                 delta=1/128,
                 T=100):  # time stopping criteria

        self.eps = eps
        self.delta = delta
        self.target_res = (1/self.delta)
        self.T = T  # stopping criterion
        self.S = int(math.log2(1/self.delta))  # 7

        # codebook W_S
        self.w_s = create_codebook(int(self.target_res))

        # initialise pi(t)_i's - distribution P(phi = theta_i | z_1:t, w_1:t) i = 1,2,.., 1/delta
        self.pi = np.array([self.delta] * int(self.target_res))

    def run_sim(self):
        # codeword selection from W^S

        for t in range(self.T):
            k = 0
            l_star = 0
            for l in range(1, self.S+1):
                # print(l, k)
                if pi_d_l_k(self.pi, l, k, self.target_res) > 1/2:

                    # select larger descent
                    l_star = l
                    # argmax part
                    k2 = pi_d_l_k(self.pi, l, k*2, self.target_res)
                    k21 = pi_d_l_k(self.pi, l, k*2+1, self.target_res)

                    if k2 > k21:
                        k = k*2
                    elif k2 < k21:
                        k = k*2+1
                    else:
                        k = k*2 + np.random.randint(0, 2)

                else:
                    # choose node closer to 0.5: current node or parent node
                    node_val = abs(pi_d_l_k(self.pi, l_star+1, k, self.target_res)-1/2)  # current node
                    parent_val = abs(pi_d_l_k(self.pi, l_star, math.floor(k/2), self.target_res)-1/2)  # parent

                    if node_val < parent_val:
                        l_new, k_new = l_star+1, k
                    elif parent_val <= node_val:  # when tie, prefer parent
                        l_new, k_new = l_star, math.floor(k/2)
                    break

            # Codeword selection result
            w_new = self.w_s[l_new-1][k_new]  # -1 because no codebook for zeroth node

            # Take next measurement
            #TODO: implement with w_new

            # Posterior update by Bayes' Rule
            #TODO: implement update rule

            # case: stopping-criterion = fixed length (FL)
            # implemented in outermost loop --> t in range(n)
            # make sure to comment out other stopping-criterion

            # case: stopping-criterion = variable length (VL)
            if np.amax(self.pi) > 1-self.eps:
                break  # should break to final beamforming


        # final beamforming vector design
        (l_hat, k_hat) = (self.S, self.pi.argmax())


        # final beamforming vector
        w_hat = self.w_s[l_hat-1][k_hat]

if __name__=='__main__':
    # channel = beams.Beams()
    sim = HPM()
    sim.run_sim()




