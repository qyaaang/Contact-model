#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 30/01/22 15:22
@description:  
@version: 1.0
"""


import numpy as np
import copy
import time


class Sobel:

    def __init__(self, criteria, param_ranges, n):
        self.criteria = criteria  # Criteria function
        self.param_ranges = param_ranges  # Parameter ranges
        self.n = n  # Number of Monte Carlo simulations
        self.m_1, self.m_2 = None, None  # Sampling matrices

    def __call__(self, *args, **kwargs):
        self.make_sampling_matrix()
        return self.sobel_index()

    def make_sampling_matrix(self):
        """
        Generate sampling matrices using Monte Carlo simulation
        """
        np.random.seed(10)
        k = len(self.param_ranges.keys())  # Number of parameters
        self.m_1 = np.zeros((self.n, k))  # Sample matrix
        self.m_2 = np.zeros((self.n, k))  # Resample matrix
        for idx, param in enumerate(self.param_ranges.keys()):
            lb = self.param_ranges[param][0]
            rb = self.param_ranges[param][1]
            self.m_1[:, idx] = np.random.uniform(lb, rb, self.n)
            self.m_2[:, idx] = np.random.uniform(lb, rb, self.n)

    def get_n_j(self, param_idx):
        """
        Compute N_j
        :param param_idx: Parameter index
        :return: N_j
        """
        n_j = copy.deepcopy(self.m_2)
        n_j[:, param_idx] = self.m_1[:, param_idx]
        return n_j

    def get_y(self, m):
        """
        Compute model output
        :param m: Sampling matrix
        :return: Model output
        """
        y = np.zeros(self.n)
        for idx in range(self.n):
            y[idx] = self.criteria(m[idx, :])
        return y

    def sobel_index(self):
        """
        Compute Sobel index
        :return: Sobel index
        """
        y = self.get_y(self.m_1)
        var_y = np.var(y)  # Variance of output
        print(var_y)
        mean_y = np.mean(y)
        y_1 = y
        y_3 = self.get_y(self.m_2)
        sobel_indices = np.zeros(self.m_1.shape[1])
        param_names = list(self.param_ranges.keys())
        for param_idx in range(self.m_1.shape[1]):
            t_0 = time.time()
            n_j = self.get_n_j(param_idx)
            y_2 = self.get_y(n_j)
            # u_j = np.mean(np.multiply(y_1, y_2 - y_3))
            u_j = np.mean(np.multiply(y_1, y_2)) - np.power(mean_y, 2)
            s_i = u_j / var_y
            sobel_indices[param_idx] = s_i
            t_1 = time.time()
            print('{}: {:.5f}\tTime cost: {:.2f}'.format(param_names[param_idx], s_i, t_1 - t_0))
        # print(np.sum(sobel_indices))
        return sobel_indices
