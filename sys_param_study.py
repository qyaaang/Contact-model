#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 13/01/22 21:55
@description:  
@version: 1.0
"""


from models.base import BaseModel
from ileenet.objective.EnergyLoss import EnergyLoss
from ileenet.objective.Probability import Probability
from utlis.FreqRatio import FreqRatio
import numpy as np
import argparse
import json
import time


class ParamStudy:

    def __init__(self, args):
        self.args = args  # Initial parameters
        self.model = BaseModel(args)  # Model
        with open('./data/ground motion/synthetic/Acc_scaled_lib.json') as f:
            self.acc_scaled_lib = json.load(f)

    def __call__(self, *args, **kwargs):
        self.criteria(args[0])

    def get_freq_ratio(self, params):
        self.model.register_params(params)
        mdl = self.model.init_model()
        mode = FreqRatio(mdl)
        f1 = mode('1')
        f2 = mode('2')
        return f1[0] / f2[0]

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

    def criteria(self, params):
        zs, probs, energies = [], [], []
        f_ratio = self.get_freq_ratio(params)
        print('f_w / f_f = {:.2f}'.format(f_ratio))
        for idx in self.acc_scaled_lib[self.args.hazard].keys():
            t_0 = time.time()
            gm = np.array(self.acc_scaled_lib[self.args.hazard][idx]['Acceleration'])
            prob, z = self.failure_prob(gm, params)
            probs.append(prob)
            zs.append(z)
            energy = self.energy_loss(gm, params)
            energies.append(energy)
            t_1 = time.time()
            print('GM:{}\tCumulative probability:{:.2f}\tPRQ:{:.5f}\tEnergy loss:{:.2f}\tTime cost:{:.2f}s'.
                  format(int(idx) + 1, prob, z, energy, t_1 - t_0))
        result = {'z': zs, 'Probability': probs, 'Energy': energies}
        result = json.dumps(result, indent=2)
        with open('./results/Parametric study/Mass ratio/{}_{:.2f}_{}_{}_{}_{}.json'.
                  format(self.args.hazard, f_ratio,
                         self.args.g, self.args.r_k, self.args.r_c, self.args.mu), 'w') as f:
            f.write(result)
        print('Probability of failure:{:2f}\tMean energy loss:{:2f}'.
              format(np.mean(np.array(probs)), np.mean(np.array(energies))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Initial parameters to be tuned
    parser.add_argument('--mb', default=15, type=float)  # Mass block
    # Fixed parameters
    parser.add_argument('--g', default=1., type=float)  # Gap
    parser.add_argument('--r_k', default=1000, type=float)  # Ratio of stiffness after contact to before
    parser.add_argument('--r_c', default=1000, type=float)  # Ratio of damping coefficient after contact to before
    parser.add_argument('--mu', default=0.1, type=float)  # Friction coefficient
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
    params = np.array([args.mb, args.g, args.r_k, args.r_c, args.mu])  # Initial parameters
    exp(params)
