#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/11/21 18:26
@description:  
@version: 1.0
"""


from models.ileebuilding import ILEEBuilding
from ileenet.optimizer.SGD import SGD
from ileenet.optimizer.Adam import Adam
import numpy as np
from scipy.optimize import minimize, Bounds
import argparse
import matplotlib.pyplot as plt
import matplotlib
import json
import copy
import time


class CaseStudy:

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
        loss1 = np.linalg.norm(u1_pred - u1_real, ord=1)
        loss2 = np.linalg.norm(u2_pred - u2_real, ord=1)
        loss3 = np.linalg.norm(u_w1_pred - self.args.h * u_w_real, ord=1)
        loss4 = np.linalg.norm(u_w2_pred - 2 * self.args.h * u_w_real, ord=1)
        loss = (loss1 + loss2 + loss3 + loss4) / 4
        # print(loss)
        return loss

    def gradient(self, params, h=0.001):
        grads = np.zeros_like(params)
        for idx in range(len(params)):
            param_tmp = copy.deepcopy(params)
            a = param_tmp[idx]
            param_tmp[idx] = a + h
            val_1 = self.loss(param_tmp)
            param_tmp[idx] = a - h
            val_2 = self.loss(param_tmp)
            grads[idx] = (val_1 - val_2) / (2 * h)
        return grads

    def update_params(self, params):
        lr = np.array([1e-6, 1e-6, 1e-2, 1e-2, 1e-7, 1e-7])
        if self.args.optimizer == 'SGD':
            optimizer = SGD(lr=lr, momentum=self.args.momentum)
            v = np.zeros_like(params)
            s, r = None, None
        else:
            optimizer = Adam(lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
            s, r = np.zeros_like(params), np.zeros_like(params)
            v = None
        param_names = ['g0_1', 'g0_2', 'r_k', 'r_c', 'rot_c', 'rot_w']
        optim_history = {}
        for epoch in range(self.args.num_epoch):
            optim_history[epoch + 1] = {}
            t1 = time.time()
            grads = self.gradient(params)
            
            if self.args.optimizer == 'SGD':
                params, v = optimizer.step(params, grads, v)
            else:
                params, s, r = optimizer.step(params, grads, s, r, epoch + 1)
            loss = self.loss(params)
            for idx, param_name in enumerate(param_names):
                optim_history[epoch + 1][param_name] = params[idx]
            optim_history[epoch + 1]['Energy loss'] = loss
            t2 = time.time()
            print('\033[1;31mEpoch: {}\033[0m\n'
                  '\033[1;32mLoss: {:.2f}\033[0m\n'
                  '\033[1;35mGrads: {}\033[0m\n'
                  '\033[1;33mg0_1: {:.6f} g0_2:{:.6f} r_k:{:.6f} r_c:{:.6f} rot_c:{:.6f} rot_w:{:.6f}\033[0m\t'
                  '\033[1;34mTime cost: {:.2f}s\033[0m'.
                  format(epoch + 1, loss, grads, 
                    params[0], params[1], params[2], params[3], params[4], params[5], t2 - t1))
        optim_history = json.dumps(optim_history, indent=2)
        with open('./results/Case study/optim history_{}_{}_{}_{}.json'.
                  format(self.args.gm, self.args.optimizer, self.args.lr, self.args.num_epoch), 'w') as f:
            f.write(optim_history)
        return params

    def search_params(self, params):
        bounds = Bounds([self.args.g0_1_lb, self.args.g0_2_lb, self.args.rk_lb, self.args.rc_lb],
                        [self.args.g0_1_rb, self.args.g0_2_rb, self.args.rk_rb, self.args.rc_rb])
        t1 = time.time()
        params_optim = minimize(self.loss, params, method='SLSQP', jac='2-point',
                                options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
        t2 = time.time()
        print('Time cost: {:.2f}s'.format(t2 - t1))
        np.save('./results/Case study/optim_params_{}_{}_{}_{}_{}.npy'.
                format(self.args.gm, self.args.g0_1, self.args.g0_2, self.args.r_k, self.args.r_c), params_optim.x)
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
    # parser.add_argument('--g0_1', default=0.872851, type=float)  # Initial gap at level 1
    # parser.add_argument('--g0_2', default=0.002481, type=float)  # Initial gap at level 2
    # parser.add_argument('--r_k', default=490.551811, type=float)  # Ratio of stiffness after contact to before
    # parser.add_argument('--r_c', default=105.225618, type=float)  # Ratio of damping coefficient after contact to before
    # parser.add_argument('--rot_c', default=0.597058, type=float)  # Rotation coefficient of column
    # parser.add_argument('--rot_w', default=0.457517, type=float)  # Rotation coefficient of wall
    # Fixed parameters
    parser.add_argument('--h_w', default=2000, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=400, type=float)  # Height of column section
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--mu', default=0.01, type=float)  # Friction coefficient
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    parser.add_argument('--xi', default=0.11, type=float)  # Crucial damping coefficient
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--h', default=4000, type=float)  # Story height
    parser.add_argument('--mb', default=3.545, type=float)  # Mass block
    parser.add_argument('--f_pi', default=451, type=float)  # Initial stress applied to the PT tendon
    parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=2, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    # Analysis hyper-parameters
    parser.add_argument('--gm', default='D1a-25%-y', type=str)  # Ground motion
    parser.add_argument('--dataset', default='test', type=str, help='PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--integrator', default='Newmark', type=str)  # Integrator
    parser.add_argument('--algo', default='modified_newton', type=str)  # Iteration algorithm
    parser.add_argument('--convergence', default='disp', type=str)  # Convergence criteria
    parser.add_argument('--tol', default=1e-8, type=float)  # Convergence tolerance
    parser.add_argument('--max_iter', default=200, type=int)  # Maximum number of iterations
    # Optimize hyper-parameters
    parser.add_argument('--optimizer', default='SGD', type=str)  # Optimizer
    parser.add_argument('--grad_tol', default=1e-8, type=float)  # Gradient norm tolerance
    parser.add_argument('--lr', default=1e-8, type=float)  # Learning rate
    parser.add_argument('--momentum', default=0.9, type=float)  # Momentum
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--num_epoch', default=10, type=int)  # Number of epoch
    # Constraints
    parser.add_argument('--g0_1_lb', default=0.1, type=float)
    parser.add_argument('--g0_1_rb', default=5, type=float)
    parser.add_argument('--g0_2_lb', default=0.1, type=float)
    parser.add_argument('--g0_2_rb', default=5, type=float)
    parser.add_argument('--g_lb', default=0.5, type=float)
    parser.add_argument('--g_rb', default=10, type=float)
    parser.add_argument('--rk_lb', default=700, type=float)
    parser.add_argument('--rk_rb', default=1500, type=float)
    parser.add_argument('--rc_lb', default=700, type=float)
    parser.add_argument('--rc_rb', default=1500, type=float)
    args = parser.parse_args()
    gm_data = np.load('./data/ground motion/{}/{}.npy'.format(args.dataset, args.gm))
    params = np.array([args.g0_1, args.g0_2, args.r_k, args.r_c, args.rot_c, args.rot_w])  # Initial parameters
    exp = CaseStudy(gm_data, args)
    # model = exp.output(params)
    exp.update_params(params)
    # exp.search_params(params)
