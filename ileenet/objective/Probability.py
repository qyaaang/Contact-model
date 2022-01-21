#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 13/01/22 20:47
@description:  
@version: 1.0
"""


import numpy as np
from scipy.stats import norm


class Probability:

    def __init__(self, model):
        self.model = model  # Model
        self.prq = None  # Peak response quantity type
        self.dof = 0  # DOF tag
        self.factor = 0  # Factor
        self.z = 0  # Peak response quantity

    def __call__(self, *args, **kwargs):
        self.register_prq(args[0])
        self.register_dof(args[1])
        self.register_factor(args[2])
        return self.gaussian_cdf(args[3])

    def register_prq(self, prq):
        """
        Register peak response quantity type
        :param prq: Peak response quantity type
        """
        assert prq == 'acc' or 'drift' 'Unexpected peak response quantity'
        self.prq = prq

    def register_dof(self, dof):
        """
        Register DOF tag
        :param dof: DOF tag
        """
        self.dof = dof

    def register_factor(self, factor):
        """
        Register factor
        :param factor: Factor
        """
        self.factor = factor

    def get_prq(self):
        """
        Get peak response quantity
        :return: Peak response quantity
        """
        if self.prq == 'acc':
            self.z = np.max(np.abs(self.model.u_tt[self.dof, :])) / self.factor
        else:
            self.z = np.max(np.abs(self.model.u[self.dof, :])) / self.factor
        # print(self.z)

    def gaussian_cdf(self, z_thresh, beta=0.05):
        """
        Compute Gaussian cumulative probability
        :param z_thresh: Peak response quantity threshold
        :param beta: Logarithmic standard deviation
        :return: Gaussian cumulative probability
        """
        self.get_prq()
        return norm.cdf(np.log(self.z / z_thresh) / beta)
