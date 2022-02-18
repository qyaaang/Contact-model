#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 9/02/22 14:36
@description:  
@version: 1.0
"""


import torch
from torch import nn, optim
from models.autoencoder import AutoEncoder
from models.ileebuilding import ILEEBuilding
from scipy.fftpack import fft, ifft
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


class DiffSim:

    def __init__(self, args):
        self.args = args
        self.FEM = ILEEBuilding(args)   # Finite element model
        self.AE = AutoEncoder(args)  # AutoEncoder

    def fem_layer(self, gm, params):
        fem_outputs = np.zeros((4, len(gm)))
        output = self.FEM(gm, params)
        u_f1 = 0.5 * (output.u[21, :] + output.u[3, :])
        u_f2 = 0.5 * (output.u[24, :] + output.u[6, :])
        u_w1 = output.u[12, :]
        u_w2 = output.u[15, :]
        fem_outputs[0, :] = u_f1
        fem_outputs[1, :] = u_f2
        fem_outputs[2, :] = u_w1
        fem_outputs[3, :] = u_w2
        return fem_outputs

    @staticmethod
    def fft(data, sampling_rate=256):
        num_freqs = int(2 ** np.ceil(np.log2(len(data))))
        freq_domain = (sampling_rate / 2) * np.linspace(0, 1, int(num_freqs / 2))  # Frequency domain
        amplitude = fft(data, num_freqs)  # FFT
        amplitude = amplitude.real
        amplitude = 2 * np.abs(amplitude[0: int(len(freq_domain) / 2)])
        amplitude[0] = 0
        freq_domain_half = freq_domain[0: int(len(freq_domain) / 2)]
        norm_amplitude = amplitude / np.max(amplitude)
        return norm_amplitude

    def fft_layer(self, x):
        num_freqs = int(2 ** np.ceil(np.log2(x.shape[1])) / 4)
        fft_outputs = np.zeros((4, num_freqs))
        for i in range(x.shape[0]):
            f = self.fft(x[i, :])
            fft_outputs[i, :] = f
        return fft_outputs

    def fft_target(self):
        u_f1_real = np.load('./data/ground truth/{}/Drift-L1-A-NS.npy'.format(self.args.gm))
        u_f2_real = np.load('./data/ground truth/{}/Drift-Roof-A-NS.npy'.format(self.args.gm))
        uw_real = np.load('./data/ground truth/{}/Wall-A-Theta-IP.npy'.format(self.args.gm))
        u_w1_real = self.args.h * uw_real
        u_w2_real = 2 * self.args.h * uw_real
        num_freqs = int(2 ** np.ceil(np.log2(u_f1_real.shape[0])) / 4)
        fft_targets = np.zeros((4, num_freqs))
        fft_targets[0, :] = self.fft(u_f1_real)
        fft_targets[1, :] = self.fft(u_f2_real)
        fft_targets[2, :] = self.fft(u_w1_real)
        fft_targets[3, :] = self.fft(u_w2_real)
        return fft_targets

    def weights_init(self):
        for name, param in self.AE.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.01, 0.01)
            else:
                nn.init.zeros_(param)

    def select_optimizer(self):
        assert self.args.optimizer == 'SGD' or 'Adam'
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(self.AE.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum
                                  )
        else:
            optimizer = optim.Adam(self.AE.parameters(),
                                   betas=(0.5, 0.999),
                                   lr=self.args.lr
                                   )
        return optimizer

    def train(self, x):
        optimizer = self.select_optimizer()  # Select optimizer
        self.weights_init()  # Initialize weights
        losses = []
        criterion = nn.MSELoss()
        for epoch in range(self.args.num_epochs):
            t_0 = time.time()
            x = torch.tensor(x, dtype=torch.float32)
            x_hat = self.AE(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_1 = time.time()
            print('\033[1;31mEpoch: {}\033[0m\t'
                  '\033[1;32mLoss: {}\033[0m\t'
                  '\033[1;33mTime cost: {:2f}s\033[0m'
                  .format(epoch + 1, loss, t_1 - t_0))
            losses.append(loss.item())
        return x_hat, losses

    def find_gap(self, gaps):
        for i in range(gaps.shape[1]):
            params = np.array([gaps[0, i], gaps[1, i], 500, 300, 0.1, 0.6, 0.2])  # Initial parameters
            u = self.fem_layer(gm_data, params)
            f = self.fft_layer(u)
            x_hat, losses = self.train(f)
            y = x_hat.detach().cpu().numpy()
            np.save('./results/Case study/Target Fourier_{}.npy'.format(self.args.gm), f)
            np.save('./results/Case study/Predicted Fourier_{}.npy'.format(self.args.gm), y)
            np.save('./results/Case study/Learning history_{}.npy'.format(self.args.gm), np.array(losses))
            fig, ax = plt.subplots()
            freq = np.linspace(0, 128, self.args.input_size)
            ax.plot(freq, f[2, :], c='b', label='Ground truth')
            ax.plot(freq, y[2, :], c='r', ls='--', label='Predicted')
            x_major_locator = plt.MultipleLocator(16)
            ax.xaxis.set_major_locator(x_major_locator)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Normalized amplitude')
            ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Initial parameters to be tuned
    parser.add_argument('--g0_1', default=0.8, type=float)  # Initial gap at level 1
    parser.add_argument('--g0_2', default=0.6, type=float)  # Initial gap at level 2
    # parser.add_argument('--g0_1', default=1.783, type=float)  # Initial gap at level 1
    # parser.add_argument('--g0_2', default=1.104, type=float)  # Initial gap at level 2
    # parser.add_argument('--g0_1', default=1.85, type=float)  # Initial gap at level 1
    # parser.add_argument('--g0_2', default=1.25, type=float)  # Initial gap at level 2
    # parser.add_argument('--g0_1', default=1.88320308732829, type=float)  # Initial gap at level 1
    # parser.add_argument('--g0_2', default=1.20443061604334, type=float)  # Initial gap at level 2
    parser.add_argument('--r_k', default=500, type=float)  # Ratio of stiffness after contact to before
    parser.add_argument('--r_c', default=300, type=float)  # Ratio of damping coefficient after contact to before
    parser.add_argument('--mu', default=0.1, type=float)  # Friction coefficient
    parser.add_argument('--rot_c', default=0.6, type=float)  # Rotation coefficient of column
    parser.add_argument('--rot_w', default=0.2, type=float)  # Rotation coefficient of wall
    # Fixed parameters
    parser.add_argument('--h_w', default=2000, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=400, type=float)  # Height of column section
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    # parser.add_argument('--xi', default=0.11, type=float)  # Crucial damping coefficient
    parser.add_argument('--xi', default=0.05, type=float)  # Crucial damping coefficient
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--h', default=4000, type=float)  # Story height
    parser.add_argument('--mb_1', default=26.1, type=float)  # Mass block at level 1
    parser.add_argument('--mb_2', default=17.85, type=float)  # Mass block at level 2
    # parser.add_argument('--f_pi', default=451, type=float)  # Initial stress applied to the PT tendon
    # parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--f_pi', default=443, type=float)  # Initial stress applied to the PT tendon
    parser.add_argument('--d', default=32, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=2, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=float)  # Density of material
    parser.add_argument('--e', default=3e4, type=float)  # Young's elasticity modulus of material
    # Training parameters
    parser.add_argument('--input_size', default=8192, type=int)  # Input size
    parser.add_argument('--hidden_size', default=256, type=int)  # Hidden size
    parser.add_argument('--num_epochs', default=1000, type=int)  # Number of epochs
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr', default=1, type=float)
    # Analysis hyper-parameters
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
    params = np.array([args.g0_1, args.g0_2, args.r_k, args.r_c, args.mu, args.rot_c, args.rot_w])  # Initial parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp = DiffSim(args)
    # u = exp.fem_layer(gm_data, params)
    # f = exp.fft_layer(u)
    # f_t = exp.fft_target()
    # x_hat = exp.train(f)
    # fig, ax = plt.subplots()
    # ax.plot(f[2, :], c='b')
    # y = x_hat.detach().cpu().numpy()
    # ax.plot(y[2, :], c='r', ls='--')
    # plt.show()
    # np.save('./results/spectra_{}.npy'.format(args.gm), y)
    # gaps = np.array([[1.58320308732829, 1.68320308732829, 1.78320308732829],
    #                  [0.900443061604334, 1.00443061604334, 1.10443061604334]])
    # gaps = np.array([[1.88320308732829, 1.98320308732829, 2.08320308732829],
    #                  [1.20443061604334, 1.30443061604334, 1.40443061604334]])
    # for i in range(gaps.shape[1]):
    #     params = np.array([gaps[0, i], gaps[1, i], 500, 300, 0.1, 0.6, 0.2])  # Initial parameters
    #     exp = DiffSim(args)
    #     u = exp.fem_layer(gm_data, params)
    #     f = exp.fft_layer(u)
    #     f_t = exp.fft_target()
    #     fig, ax = plt.subplots()
    #     ax.plot(f[1, :], c='b')
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     x_hat = exp.train(f)
    #     y = x_hat.detach().cpu().numpy()
    #     ax.plot(y[1, :], c='r', ls='--')
    # plt.show()
    gaps = np.array([[3.30085862],
                     [2.44430977]])
    exp.find_gap(gaps)
