#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 6/02/22 12:06
@description:  
@version: 1.0
"""


import numpy as np
from models.base_optim import BaseModel
from ileenet.objective.Probability import Probability
from ileenet.objective.EnergyLoss import EnergyLoss
import json
import argparse


class OptimObj:

    def __init__(self, args):
        self.args = args
        with open('./data/ground motion/synthetic/Acc_scaled_lib.json') as f:
            self.acc_scaled_lib = json.load(f)

    def failure_prob(self, hazard, params):
        model = BaseModel(self.args, hazard)  # Model
        probs = []
        for gm_idx in self.acc_scaled_lib[hazard].keys():
            gm = np.array(self.acc_scaled_lib[hazard][gm_idx]['Acceleration'])
            output = model(gm, params)
            criteria = Probability(output)
            if hazard == 'SLE':
                prob = criteria('acc', self.args.dof_tag, self.args.factor, self.args.a_thresh, self.args.beta)
            else:
                prob = criteria('drift', self.args.dof_tag, self.args.h, self.args.d_thresh, self.args.beta)
            probs.append(prob)
        pof = np.mean(np.array(probs))
        return pof

    def energy_loss(self, hazard, params):
        model = BaseModel(self.args, hazard)  # Model
        energies = []
        for gm_idx in self.acc_scaled_lib[hazard].keys():
            gm = np.array(self.acc_scaled_lib[hazard][gm_idx]['Acceleration'])
            output = model(gm, params)
            criteria = EnergyLoss(output)
            energy_loss = criteria()
            energies.append(energy_loss)
        mean_energy_loss = np.mean(np.array(energies))
        return mean_energy_loss

    def objective(self, params):
        pof_sle = self.failure_prob('SLE', params)
        mel_sle = self.energy_loss('SLE', params)
        pof_mce = self.failure_prob('MCE', params)
        mel_mce = self.energy_loss('MCE', params)
        obj = 0.25 * pof_sle + 0.25 * (mel_sle / 100000) + 0.25 * pof_mce + 0.25 * (mel_mce / 40000000)
        print(obj)
        print('POF_SLE: {}\tMEL_SLE: {}\tPOF_MCE: {}\tMEL: {}'.format(pof_sle, mel_sle, pof_mce, mel_mce))
        return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Fixed parameters
    parser.add_argument('--mb', default=5, type=float)
    parser.add_argument('--h_w', default=1500, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=800, type=float)  # Height of column section
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    parser.add_argument('--xi1', default=0.05, type=float)  # Crucial damping coefficient
    parser.add_argument('--xi2', default=0.0001, type=float)  # Crucial damping coefficient
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--h', default=4000, type=float)  # Story height
    parser.add_argument('--alpha', default=1.0, type=float)  # PT coefficient
    parser.add_argument('--f_y', default=451, type=float)  # Yield strength of PT bar
    parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=1, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    parser.add_argument('--rot1', default=0.5, type=float)  # Rotation coefficient
    parser.add_argument('--rot2', default=0.05, type=float)  # Rotation coefficient
    # Analysis hyper-parameters
    parser.add_argument('--dataset', default='synthetic', type=str,
                        help='synthetic, PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--fs', default=100, type=float)  # Sampling frequency
    parser.add_argument('--factor', default=9800, type=float)  # GM factor
    parser.add_argument('--factor1', default=9800, type=float)  # GM factor
    parser.add_argument('--factor2', default=4000, type=float)  # GM factor
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
    optim_params = np.array([0.502, 743.639, 665.722, 0.104])
    optim = OptimObj(args)
    optim.objective(optim_params)
