#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 15/11/21 09:41
@description:  
@version: 1.0
"""


import numpy as np
from models.base import BaseModel
from ileenet.objective.EnergyLoss import EnergyLoss
import copy
import argparse
import time


class ParamSen:

    def __init__(self, gm, args):
        self.gm = gm
        self.args = args
        self.m_1, self.m_2 = None, None
        self.model = BaseModel(args)  # Model

    def energy_loss(self, params):
        output = self.model(self.gm['Data'], self.gm['Fs'], self.gm['Scale'], params)
        criteria = EnergyLoss(output)
        energy_loss = criteria()
        return energy_loss

    def get_param_range(self):
        return {'h_w': [self.args.hw_lb, self.args.hw_rb],
                'h_c': [self.args.hc_lb, self.args.hc_rb],
                'g': [self.args.g_lb, self.args.g_rb],
                'r_k': [self.args.rk_lb, self.args.rk_rb],
                'r_c': [self.args.rc_lb, self.args.rc_rb],
                'mu': [self.args.mu_lb, self.args.mu_rb]
                }

    def monte_carlo(self):
        np.random.seed(self.args.seed)
        self.m_1 = np.zeros((self.args.n, self.args.k))  # Sample matrix
        self.m_2 = np.zeros((self.args.n, self.args.k))  # Resample matrix
        param_range = self.get_param_range()
        for idx, param in enumerate(param_range.keys()):
            lb = param_range[param][0]
            rb = param_range[param][1]
            self.m_1[:, idx] = np.random.uniform(lb, rb, self.args.n)
            self.m_2[:, idx] = np.random.uniform(lb, rb, self.args.n)
        return self.m_1, self.m_2

    def get_n_j(self, param_idx):
        n_j = copy.deepcopy(self.m_2)
        n_j[:, param_idx] = self.m_1[:, param_idx]
        return n_j

    def get_y(self, m):
        y = np.zeros(self.args.n)
        for idx in range(self.args.n):
            y[idx] = self.energy_loss(m[idx, :])
        return y

    def sobol_indx(self):
        y = self.get_y(self.m_1)
        var_y = np.var(y)  # Variance of output
        y_1 = y
        y_3 = self.get_y(self.m_2)
        s = np.zeros(self.m_1.shape[1])
        for param_idx in range(self.m_1.shape[1]):
            n_j = self.get_n_j(param_idx)
            y_2 = self.get_y(n_j)
            u_j = np.mean(np.multiply(y_1, y_2 - y_3))
            s_i = u_j / var_y
            s[param_idx] = s_i
            print(s_i)
        print(np.sum(s))
        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sensitivity analysis hyper-parameters
    parser.add_argument('--seed', default=100, type=int)  # Random seed
    parser.add_argument('--n', default=1024, type=int)  # NUmber of Monte Carlo simulations
    parser.add_argument('--k', default=6, type=int)  # NUmber of parameters
    # Parameter ranges
    parser.add_argument('--hw_lb', default=1500, type=float)
    parser.add_argument('--hw_rb', default=2005, type=float)
    parser.add_argument('--hc_lb', default=200, type=float)
    parser.add_argument('--hc_rb', default=405, type=float)
    parser.add_argument('--g_lb', default=0.5, type=float)
    parser.add_argument('--g_rb', default=1.5, type=float)
    parser.add_argument('--rk_lb', default=900, type=float)
    parser.add_argument('--rk_rb', default=1200, type=float)
    parser.add_argument('--rc_lb', default=900, type=float)
    parser.add_argument('--rc_rb', default=1200, type=float)
    parser.add_argument('--mu_lb', default=0.1, type=float)
    parser.add_argument('--mu_rb', default=0.5, type=float)
    # Fixed parameters
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    parser.add_argument('--xi', default=0.05, type=float)  # Crucial damping coefficient
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--h', default=3000, type=float)  # Story height
    parser.add_argument('--mb', default=3.545, type=float)  # Mass block
    parser.add_argument('--alpha', default=0.9, type=float)  # PT coefficient
    parser.add_argument('--f_y', default=1137, type=float)  # Yield strength of PT bar
    parser.add_argument('--d', default=32, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=2, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    # Analysis hyper-parameters
    parser.add_argument('--gm', default='D1a-25%-y', type=str)  # Ground motion
    parser.add_argument('--dataset', default='PEER', type=str, help='PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--fs', default=256, type=float)  # Sampling frequency
    parser.add_argument('--scale', default=9800, type=float)  # Scale factor
    parser.add_argument('--integrator', default='Newmark', type=str)  # Integrator
    parser.add_argument('--algo', default='modified_newton', type=str)  # Iteration algorithm
    parser.add_argument('--convergence', default='disp', type=str)  # Convergence criteria
    parser.add_argument('--tol', default=1e-8, type=float)  # Convergence tolerance
    parser.add_argument('--max_iter', default=200, type=int)  # Maximum number of iterations
    args = parser.parse_args()
    gm_data = np.load('./data/ground motion/{}/{}.npy'.format(args.dataset, args.gm))
    gm = {'Data': gm_data, 'Fs': args.fs, 'Scale': args.scale}
    t1 = time.time()
    exp = ParamSen(gm, args)
    m_1, m_2 = exp.monte_carlo()
    y = exp.get_y(m_1)
    s = exp.sobol_indx()
    t2 = time.time()
    print('Time cost: {:.2f}s.'.format(t2 - t1))
