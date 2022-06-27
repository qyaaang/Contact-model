#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 3/02/22 14:50
@description:  
@version: 1.0
"""


import numpy as np
from models.base import BaseModel
from ileenet.objective.Probability import Probability
from ileenet.objective.EnergyLoss import EnergyLoss
import argparse
import json
import time
import os


class ParamSpace:

    def __init__(self, args):
        self.args = args
        self.model = BaseModel(args)  # Model
        with open('./data/ground motion/synthetic/Acc_scaled_lib.json') as f:
            self.acc_scaled_lib = json.load(f)

    def __call__(self, *args, **kwargs):
        self.param_space()

    def failure_prob(self, params):
        probs = []
        for gm_idx in self.acc_scaled_lib[self.args.hazard].keys():
            gm = np.array(self.acc_scaled_lib[self.args.hazard][gm_idx]['Acceleration'])
            output = self.model(gm, params)
            criteria = Probability(output)
            if self.args.hazard == 'SLE':
                prob = criteria('acc', self.args.dof_tag, self.args.factor, self.args.a_thresh, self.args.beta)
            else:
                prob = criteria('drift', self.args.dof_tag, self.args.h, self.args.d_thresh, self.args.beta)
            probs.append(prob)
        pof = np.mean(np.array(probs))
        return pof

    def energy_loss(self, params):
        energies = []
        for gm_idx in self.acc_scaled_lib[self.args.hazard].keys():
            gm = np.array(self.acc_scaled_lib[self.args.hazard][gm_idx]['Acceleration'])
            output = self.model(gm, params)
            criteria = EnergyLoss(output)
            energy_loss = criteria()
            energies.append(energy_loss)
        mean_energy_loss = np.mean(np.array(energies))
        return mean_energy_loss

    def make_param_ranges(self):
        return {'g': [self.args.g_lb, self.args.g_rb],
                'r_k': [self.args.rk_lb, self.args.rk_rb],
                'r_c': [self.args.rc_lb, self.args.rc_rb],
                'mu': [self.args.mu_lb, self.args.mu_rb]
                }

    def make_sampling_matrix(self):
        np.random.seed(10)
        param_ranges = self.make_param_ranges()
        k = len(param_ranges.keys())  # Number of parameters
        param_matrix = np.zeros((self.args.num_samples, k))  # Sample matrix
        for idx, param in enumerate(param_ranges.keys()):
            lb = param_ranges[param][0]
            rb = param_ranges[param][1]
            param_matrix[:, idx] = np.random.uniform(lb, rb, self.args.num_samples)
        np.save('./results/Optimization study/param_matrix_{}.npy'.format(self.args.num_samples), param_matrix)
        return param_matrix

    def param_space(self):
        if not os.path.exists('./results/Optimization study/param_matrix_{}.npy'.format(self.args.num_samples)):
            param_matrix = self.make_sampling_matrix()
        else:
            param_matrix = np.load('./results/Optimization study/param_matrix_{}.npy'.format(self.args.num_samples))
        zs = np.zeros((self.args.num_samples, 2))
        for i in range(self.args.num_samples):
            t_0 = time.time()
            param = param_matrix[i, :]
            try:
                pof = self.failure_prob(param)
                mel = self.energy_loss(param)
            except:
                pof = 0
                mel = 0
            t_1 = time.time()
            zs[i, 0] = pof
            zs[i, 1] = mel
            print('Sample: {}\tParam: {}\tPOF: {:.2f}\tMEL: {:.2f}\tTime cost: {:.2f}s.'.
                  format(i + 1, [round(z, 2) for z in param], pof, mel, t_1 - t_0))
        np.save('./results/Optimization study/z_{}_{}.npy'.format(self.args.hazard, self.args.num_samples), zs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameter ranges
    parser.add_argument('--num_samples', default=500, type=int)  # Number of samples
    parser.add_argument('--g_lb', default=0.5, type=float)
    parser.add_argument('--g_rb', default=5.0, type=float)
    parser.add_argument('--rk_lb', default=500, type=float)
    parser.add_argument('--rk_rb', default=2000, type=float)
    parser.add_argument('--rc_lb', default=500, type=float)
    parser.add_argument('--rc_rb', default=2000, type=float)
    parser.add_argument('--mu_lb', default=0.1, type=float)
    parser.add_argument('--mu_rb', default=0.5, type=float)
    # Fixed parameters
    parser.add_argument('--mb', default=5, type=float)
    parser.add_argument('--h_w', default=1500, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=800, type=float)  # Height of column section
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    parser.add_argument('--xi', default=0.05, type=float)  # Crucial damping coefficient
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--h', default=4000, type=float)  # Story height
    parser.add_argument('--alpha', default=1.0, type=float)  # PT coefficient
    parser.add_argument('--f_y', default=451, type=float)  # Yield strength of PT bar
    parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=1, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    parser.add_argument('--rot', default=0.5, type=float)  # Rotation coefficient
    # Analysis hyper-parameters
    parser.add_argument('--hazard', default='SLE', type=str)  # Ground motion
    parser.add_argument('--dataset', default='synthetic', type=str,
                        help='synthetic, PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--fs', default=100, type=float)  # Sampling frequency
    parser.add_argument('--factor', default=9800, type=float)  # GM factor
    parser.add_argument('--integrator', default='Newmark', type=str)  # Integrator
    parser.add_argument('--algo', default='momentum_newton', type=str)  # Iteration algorithm
    parser.add_argument('--convergence', default='disp', type=str)  # Convergence criteria
    parser.add_argument('--tol', default=1e-8, type=float)  # Convergence tolerance
    parser.add_argument('--max_iter', default=200, type=int)  # Maximum number of iterations
    # Criteria hyper-parameters
    parser.add_argument('--dof_tag', default=9, type=int)  # DOF tag
    parser.add_argument('--a_thresh', default=0.3, type=float)  # Threshold acceleration
    parser.add_argument('--d_thresh', default=3, type=float)  # Threshold drift ratio
    parser.add_argument('--beta', default=0.05, type=float)  # Logarithmic standard deviation
    args = parser.parse_args()
    exp = ParamSpace(args)
    exp()
