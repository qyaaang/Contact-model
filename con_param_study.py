#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 25/01/22 16:31
@description:  
@version: 1.0
"""


from models.base import BaseModel
from ileenet.objective.EnergyLoss import EnergyLoss
from ileenet.objective.Probability import Probability
import numpy as np
import argparse
import json
import time


class ParamStudy:

    def __init__(self, args):
        self.args = args  # Initial parameters
        self.model = BaseModel(args)  # Model
        self.param_ranges = {'Gap': [args.g, args.g_lb, args.g_rb],
                             'Stiffness': [args.r_k, args.rk_lb, args.rk_rb],
                             'Damping': [args.r_c, args.rc_lb, args.rc_rb],
                             'Friction': [args.mu, args.mu_lb, args.mu_rb]}
        with open('./data/ground motion/synthetic/Acc_scaled_lib.json') as f:
            self.acc_scaled_lib = json.load(f)

    def __call__(self, *args, **kwargs):
        self.criteria()

    def failure_prob(self, gm, params):
        output = self.model(gm, params)
        criteria = Probability(output)
        if self.args.hazard == 'SLE':
            prob = criteria('acc', self.args.dof_tag, self.args.factor, self.args.a_thresh, self.args.beta)
        else:
            prob = criteria('drift', self.args.dof_tag, self.args.h, self.args.d_thresh, self.args.beta)
        return prob, criteria.z

    def energy_loss(self, gm, params):
        output = self.model(gm, params)
        criteria = EnergyLoss(output)
        energy_loss = criteria()
        return energy_loss

    def gen_params_lib(self, num_params=4):
        np.random.seed(10)
        params_lib = np.zeros((self.args.num_samples, num_params))
        lb = self.param_ranges[self.args.param_name][1]
        rb = self.param_ranges[self.args.param_name][2]
        samples = np.random.uniform(lb, rb, self.args.num_samples)
        sample_idx = list(self.param_ranges.keys()).index(self.args.param_name)
        for idx in range(num_params):
            if idx == sample_idx:
                params_lib[:, idx] = samples
            else:
                fix_param = self.param_ranges[list(self.param_ranges.keys())[idx]][0]
                params_lib[:, idx] = fix_param * np.ones_like(samples)
        return params_lib

    def criteria(self):
        params_lib = self.gen_params_lib()
        sample_idx = list(self.param_ranges.keys()).index(self.args.param_name)
        result = {}
        for param_idx in range(self.args.num_samples):
            t_0 = time.time()
            params = params_lib[param_idx, :]
            print('\033[1;32mParam {}: {}\033[0m'.format(param_idx + 1, list(params)))
            zs, probs, energies = [], [], []
            for gm_idx in self.acc_scaled_lib[self.args.hazard].keys():
                gm = np.array(self.acc_scaled_lib[self.args.hazard][gm_idx]['Acceleration'])
                prob, z = self.failure_prob(gm, params)
                probs.append(prob)
                zs.append(z)
                energy = self.energy_loss(gm, params)
                energies.append(energy)
                print('GM:{}\tCumulative probability:{:.2f}\tPRQ:{:.5f}\tEnergy loss:{:.2f}'.
                      format(int(gm_idx) + 1, prob, z, energy))
            t_1 = time.time()
            print('Probability of failure:{:2f}\tMean energy loss:{:2f}\tTime cost:{:.2f}s'.
                  format(np.mean(np.array(probs)), np.mean(np.array(energies)), t_1 - t_0))
            result[params[sample_idx]] = {'z': zs, 'Probability': probs, 'Energy': energies}
        result = json.dumps(result, indent=2)
        with open('./results/Parametric study/{}_{}.json'.
                  format(self.args.hazard, self.args.param_name), 'w') as f:
            f.write(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameter ranges
    parser.add_argument('--param_name', default='Gap', type=str)  # Parameter name
    parser.add_argument('--num_samples', default=25, type=int)  # Number of samples
    parser.add_argument('--g', default=1, type=float)  # Gap
    parser.add_argument('--g_lb', default=0.5, type=float)
    parser.add_argument('--g_rb', default=2.0, type=float)
    parser.add_argument('--r_k', default=1000, type=float)  # Ratio of stiffness after contact to before
    parser.add_argument('--rk_lb', default=500, type=float)
    parser.add_argument('--rk_rb', default=2000, type=float)
    parser.add_argument('--r_c', default=1000, type=float)  # Ratio of damping coefficient after contact to before
    parser.add_argument('--rc_lb', default=500, type=float)
    parser.add_argument('--rc_rb', default=2000, type=float)
    parser.add_argument('--mu', default=0.1, type=float)  # Friction coefficient
    parser.add_argument('--mu_lb', default=0.1, type=float)
    parser.add_argument('--mu_rb', default=0.5, type=float)
    # Fixed parameters
    parser.add_argument('--mb', default=5, type=float)  # Mass block
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
    exp = ParamStudy(args)
    exp()
