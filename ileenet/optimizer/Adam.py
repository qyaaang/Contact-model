#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 9/11/21 09:39
@description:  
@version: 1.0
"""


import numpy as np


class Adam:

    def __init__(self, lr, betas):
        self.lr = lr  # Learning rate
        self.betas = betas  # Exponential decay rates for the moment estimates

    def step(self, params, grads, s, r, t):
        """
        Update parameters
        :param params: Current parameters
        :param grads: Current gradients
        :param s: Updated biased first moment estimate
        :param r: Updated biased second raw moment estimate
        :param t: Current time step
        :return: Updated parameters
        """
        s = self.betas[0] * s + (1 - self.betas[0]) * grads
        r = self.betas[1] * r + (1 - self.betas[1]) * np.multiply(grads, grads)
        s_hat = s / (1 - np.power(self.betas[0], t + 1))  # Bias-corrected first moment estimate
        r_hat = r / (1 - np.power(self.betas[1], t + 1))  # Bias-corrected second raw moment estimate
        delta_param = np.multiply(s_hat, 1 / (np.sqrt(r_hat) + 1e-8))
        params -= np.multiply(self.lr, delta_param)
        return params
