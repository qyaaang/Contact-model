#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 8/02/22 16:53
@description:  
@version: 1.0
"""

from models.ileebuilding_optim import ILEEBuilding
import numpy as np
from scipy.optimize import minimize, Bounds
import argparse
import matplotlib.pyplot as plt
import matplotlib
import time


class DiffSim:

    def __init__(self, gm, args):
        self.gm = gm  # Ground motion
        self.args = args  # Initial parameters
        self.model = ILEEBuilding(args)  # Model

    def loss(self, params):
        output = self.model(self.gm, params)
        u1_pred = 0.5 * (output.u[21, :] + output.u[3, :])
        u2_pred = 0.5 * (output.u[24, :] + output.u[6, :])
        u_w1_pred = output.u[12, :]
        u_w2_pred = output.u[15, :]
        u1_real = np.load('./data/ground truth/{}/Drift-L1-A-NS.npy'.format(self.args.gm))
        u2_real = np.load('./data/ground truth/{}/Drift-Roof-A-NS.npy'.format(self.args.gm))
        u_w_real = np.load('./data/ground truth/{}/Wall-A-Theta-IP.npy'.format(self.args.gm))
        loss1 = np.linalg.norm(u1_pred - u1_real, ord=2)
        loss2 = np.linalg.norm(u2_pred - u2_real, ord=2)
        # loss3 = np.linalg.norm(u_w1_pred - self.args.h * u_w_real, ord=2)
        # loss4 = np.linalg.norm(u_w2_pred - 2 * self.args.h * u_w_real, ord=2)
        # loss = (loss1 + loss2 + loss3 + loss4) / 4
        loss = (loss1 + loss2) / 2
        print(loss)
        return loss

    def search_params(self, params):
        bounds = Bounds([self.args.g0_1_lb, self.args.g0_2_lb, self.args.rk_lb, self.args.rc_lb,
                         self.args.rotc_lb, self.args.rotw_lb, self.args.mu_lb, self.args.h_lb,
                         self.args.mb1_lb, self.args.mb2_lb, self.args.fpi_lb, self.args.d_lb,
                         self.args.xi_lb],
                        [self.args.g0_1_rb, self.args.g0_2_rb, self.args.rk_rb, self.args.rc_rb,
                         self.args.rotc_rb, self.args.rotw_rb, self.args.mu_rb, self.args.h_rb,
                         self.args.mb1_rb, self.args.mb2_rb, self.args.fpi_rb, self.args.d_rb,
                         self.args.xi_rb])
        t1 = time.time()
        params_optim = minimize(self.loss, params, method='SLSQP', jac='2-point',
                                options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
        t2 = time.time()
        print('Time cost: {:.2f}s'.format(t2 - t1))
        np.save('./results/Case study/optim_params_{}.npy'.format(self.args.gm), params_optim.x)
        print(params_optim)
        return params_optim

    def output(self, params):
        output = self.model(self.gm, params)
        matplotlib.rcParams['text.usetex'] = True
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(np.arange(0, output.accel_len) / 256, 0.5 * (output.u[21, :] + output.u[3, :]), ls='-', c='b',
                 label='L1-Col')
        data_real = np.load('./data/ground truth/{}/Drift-L1-A-NS.npy'.format(self.args.gm))
        ax1.plot(np.arange(0, output.accel_len) / 256, data_real, ls='--', c='r', label='Exp')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$u$')
        ax1.set_xlim(0)
        ax1.legend()
        u1_mean = 0.5 * (output.u[21, :] + output.u[3, :])
        loss1 = np.linalg.norm(u1_mean - data_real, ord=1)
        print('L1-Col loss: {}'.format(loss1 / output.accel_len))
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(np.arange(0, output.accel_len) / 256, 0.5 * (output.u[24, :] + output.u[6, :]), ls='-', c='b',
                 label='L2-Col')
        data_real = np.load('./data/ground truth/{}/Drift-Roof-A-NS.npy'.format(self.args.gm))
        ax2.plot(np.arange(0, output.accel_len) / 256, data_real, ls='--', c='r', label='Exp')
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$u$')
        ax2.set_xlim(0)
        ax2.legend()
        u2_mean = 0.5 * (output.u[24, :] + output.u[6, :])
        loss2 = np.linalg.norm(u2_mean - data_real, ord=1)
        print('L2-Col loss: {}'.format(loss2 / output.accel_len))
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(np.arange(0, output.accel_len) / 256, output.u[12, :], ls='-', c='b', label='L1-Wall')
        data_real = np.load('./data/ground truth/{}/Wall-A-Theta-IP.npy'.format(self.args.gm))
        ax3.plot(np.arange(0, output.accel_len) / 256, self.args.h * data_real, ls='--', c='r', label='L1-Exp')
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$u$')
        ax3.set_xlim(0)
        ax3.legend()
        loss3 = np.linalg.norm(output.u[12, :] - self.args.h * data_real, ord=1)
        print('L1-Wall loss: {}'.format(loss3 / output.accel_len))
        fig4, ax4 = plt.subplots(figsize=(10, 3))
        ax4.plot(np.arange(0, output.accel_len) / 256, output.u[15, :], ls='-', c='b', label='L2-Wall')
        data_real = np.load('./data/ground truth/{}/Wall-A-Theta-IP.npy'.format(self.args.gm))
        ax4.plot(np.arange(0, output.accel_len) / 256, 2 * self.args.h * data_real, ls='--', c='r', label='L2-Exp')
        ax4.set_xlabel(r'$t$')
        ax4.set_ylabel(r'$u$')
        ax4.set_xlim(0)
        ax4.legend()
        loss4 = np.linalg.norm(output.u[15, :] - 2 * self.args.h * data_real, ord=1)
        print('L2-Wall loss: {}'.format(loss4 / output.accel_len))
        fig5, ax5 = plt.subplots(figsize=(10, 3))
        data_real = np.load('./data/ground truth/{}/D-L1-Wall-A-t2w-H.npy'.format(self.args.gm))
        # ax5.plot(np.arange(0, output.accel_len) / 256, output.g['7'], ls='-', c='b', label='L1-Gap-L')
        # ax5.plot(np.arange(0, output.accel_len) / 256, output.g['8'], ls='--', c='r', label='L1-Gap-R')
        ax5.plot(np.arange(0, output.accel_len) / 256, output.g['7'] + output.g['8'], ls='-', c='b',
                 label='L1-Gap')
        ax5.plot(np.arange(0, output.accel_len) / 256, data_real, ls='--', c='r', label='L1-Exp')
        ax5.axhline(y=0, ls='--', c='k')
        ax5.set_xlabel(r'$t$')
        ax5.set_ylabel(r'$g$')
        ax5.set_xlim(0)
        ax5.legend()
        fig6, ax6 = plt.subplots(figsize=(10, 3))
        ax6.plot(np.arange(0, output.accel_len) / 256, 0.5 * (output.u[21, :] + output.u[3, :]), ls='--', c='r',
                 label='L1-Col')
        ax6.plot(np.arange(0, output.accel_len) / 256, output.u[12, :], ls='-', c='b', lw=0.5, label='L1-Wall')
        ax6.legend()
        plt.tight_layout()
        plt.show()
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Initial parameters to be tuned
    parser.add_argument('--g0_1', default=0.8, type=float)  # Initial gap at level 1
    parser.add_argument('--g0_2', default=0.6, type=float)  # Initial gap at level 2
    parser.add_argument('--r_k', default=500, type=float)  # Ratio of stiffness after contact to before
    parser.add_argument('--r_c', default=300, type=float)  # Ratio of damping coefficient after contact to before
    parser.add_argument('--rot_c', default=0.6, type=float)  # Rotation coefficient of column
    parser.add_argument('--rot_w', default=0.3, type=float)  # Rotation coefficient of wall

    parser.add_argument('--mu', default=0.1, type=float)  # Friction coefficient
    parser.add_argument('--h', default=4000, type=float)  # Story height
    parser.add_argument('--mb_1', default=26.1, type=float)  # Mass block at level 1
    parser.add_argument('--mb_2', default=17.85, type=float)  # Mass block at level 2
    parser.add_argument('--f_pi', default=451, type=float)  # Initial stress applied to the PT tendon
    parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--xi', default=0.11, type=float)  # Crucial damping coefficient

    parser.add_argument('--h_w', default=2000, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=400, type=float)  # Height of column section
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    # Fixed parameters
    # parser.add_argument('--h_w', default=2000, type=float)  # Height of wall section
    # parser.add_argument('--h_c', default=400, type=float)  # Height of column section
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    # parser.add_argument('--xi', default=0.11, type=float)  # Crucial damping coefficient
    # parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    # parser.add_argument('--h', default=4000, type=float)  # Story height
    # parser.add_argument('--mb', default=3.545, type=float)  # Mass block
    # parser.add_argument('--f_pi', default=451, type=float)  # Initial stress applied to the PT tendon
    # parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=2, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    # Analysis hyper-parameters
    parser.add_argument('--gm', default='D2-180%-FF-y', type=str)  # Ground motion
    parser.add_argument('--dataset', default='test', type=str, help='PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--integrator', default='Newmark', type=str)  # Integrator
    parser.add_argument('--algo', default='momentum_newton', type=str)  # Iteration algorithm
    parser.add_argument('--convergence', default='disp', type=str)  # Convergence criteria
    parser.add_argument('--tol', default=1e-8, type=float)  # Convergence tolerance
    parser.add_argument('--max_iter', default=200, type=int)  # Maximum number of iterations
    # Constraints
    parser.add_argument('--g0_1_lb', default=0.5, type=float)
    parser.add_argument('--g0_1_rb', default=2, type=float)
    parser.add_argument('--g0_2_lb', default=0.5, type=float)
    parser.add_argument('--g0_2_rb', default=2, type=float)
    parser.add_argument('--rk_lb', default=100, type=float)
    parser.add_argument('--rk_rb', default=1500, type=float)
    parser.add_argument('--rc_lb', default=100, type=float)
    parser.add_argument('--rc_rb', default=1500, type=float)
    parser.add_argument('--rotc_lb', default=0.01, type=float)
    parser.add_argument('--rotc_rb', default=0.5, type=float)
    parser.add_argument('--rotw_lb', default=0.01, type=float)
    parser.add_argument('--rotw_rb', default=0.5, type=float)
    parser.add_argument('--mu_lb', default=0.1, type=float)
    parser.add_argument('--mu_rb', default=0.5, type=float)
    parser.add_argument('--h_lb', default=3000, type=float)
    parser.add_argument('--h_rb', default=5000, type=float)
    parser.add_argument('--mb1_lb', default=10, type=float)
    parser.add_argument('--mb1_rb', default=100, type=float)
    parser.add_argument('--mb2_lb', default=10, type=float)
    parser.add_argument('--mb2_rb', default=100, type=float)
    parser.add_argument('--fpi_lb', default=100, type=float)
    parser.add_argument('--fpi_rb', default=1000, type=float)
    parser.add_argument('--d_lb', default=10, type=float)
    parser.add_argument('--d_rb', default=100, type=float)
    parser.add_argument('--xi_lb', default=0.0001, type=float)
    parser.add_argument('--xi_rb', default=0.15, type=float)
    args = parser.parse_args()
    gm_data = np.load('./data/ground motion/{}/{}.npy'.format(args.dataset, args.gm))
    params = np.array([args.g0_1, args.g0_2, args.r_k, args.r_c, args.rot_c, args.rot_w,
                       args.mu, args.h, args.mb_1, args.mb_2, args.f_pi, args.d, args.xi])  # Initial parameters
    exp = DiffSim(gm_data, args)
    # opt_params = np.array([1.34577785e+00, 9.01683838e-01, 5.45888030e+02, 1,
    #                        4.99675544e-01, 4.95406443e-01, 0.1, 4000, 26.1, 17.85, 451, 25, 0.11])
    # opt_params = np.array([2.82577785e+00, 1.61683838e-01, 8.45888030e+02, 1,
    #                        0.6, 0.2, 0.1, 4000, 26.1, 17.85, 451, 25, 0.11])
    # opt_params = np.array([2.85, 1.65, 1000, 1,
    #                        0.5, 0.1, 0.1, 4000, 26.1, 17.85, 451, 25, 0.11])
    # opt_params = np.array([2.68, 2.36, 500, 300,
    #                        0.5, 0.1, 0.1, 4000, 26.1, 17.85, 443, 32, 0.05])
    # opt_params = np.array([3.01, 2.55, 1000, 500,
    #                        0.5, 0.2, 0.1, 4000, 26.1, 17.85, 443, 32, 0.05])
    opt_params = np.array([1.5, 1.25, 1000, 500,
                           0.5, 0.05, 0.1, 4000, 26.1, 17.85, 443, 32, 0.05])
    model = exp.output(opt_params)
    # exp.search_params(params)
