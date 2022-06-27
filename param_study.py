#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 6/25/2022 21:40
@description:  
@version: 1.0
"""


from models.simple.base import BaseModel
import numpy as np
import argparse
import time


class ParamStudy:

    def __init__(self, gm, args):
        self.gm = gm  # Ground motion
        self.args = args  # Initial parameters
        self.model = BaseModel(args)  # Model

    def run(self, params):
        output = self.model(self.gm, params)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--eff_h', default=6310, type=float)  # Effective height
    parser.add_argument('--h', default=8200, type=float)  # Overall height
    parser.add_argument('--xi', default=0.1, type=float)  # Damping coefficient
    parser.add_argument('--h_w', default=2000, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=400, type=float)  # Height of column section
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--k_c1', default=0.001, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    parser.add_argument('--rou', default=2.4e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    # Analysis parameters
    parser.add_argument('--gm', default='D2-180%-FF-y', type=str)  # Ground motion
    parser.add_argument('--dataset', default='test', type=str, help='PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--integrator', default='Newmark', type=str)  # Integrator
    parser.add_argument('--algo', default='momentum_newton', type=str)  # Iteration algorithm
    parser.add_argument('--convergence', default='disp', type=str)  # Convergence criteria
    parser.add_argument('--tol', default=1e-8, type=float)  # Convergence tolerance
    parser.add_argument('--max_iter', default=200, type=int)  # Maximum number of iterations
    args = parser.parse_args()
    gm_data = np.load('./data/ground motion/{}/{}.npy'.format(args.dataset, args.gm))
    exp = ParamStudy(gm_data, args)
    # gaps = np.arange(0.5, 2.1, 0.1)
    gaps = np.arange(1000000, 7000000, 500000)
    drifts = np.zeros((len(gaps), len(gm_data)))
    accels = np.zeros((len(gaps), len(gm_data)))
    for idx, gap in enumerate(gaps):
        t_0 = time.time()
        params = np.array([4.537, 1.5, gap, 0, 0.1])
        model = exp.run(params)
        t_1 = time.time()
        drifts[idx, :] = model.u[9, :]
        accels[idx, :] = model.u_tt[9, :]
        print('Gap: {:.1f}\tPeak drift: {}\tPeak acceleration: {}\tTime cost: {:.2f}s'.
              format(gap, np.max(np.abs(model.u[9, :]) / args.eff_h),
                     np.max(np.abs(model.u_tt[9, :]) / 9800), t_1 - t_0))
    np.save('./results/Parameter_study/Drift_{}_k.npy'.format(args.gm), drifts)
    np.save('./results/Parameter_study/Accel_{}_k.npy'.format(args.gm), accels)
