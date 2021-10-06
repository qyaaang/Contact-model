#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 6/10/21 11:34 AM
@description:  
@version: 1.0
"""


import numpy as np


class Convergence:

    def __init__(self, criteria):
        self.criteria = criteria

    def __call__(self, *args, **kwargs):
        if self.criteria == 'disp':
            return self.disp_norm_criteria(*args)

    def disp_norm_criteria(self, delta_u, u_trial):
        """
        Norm displacement increment convergence criteria
        :param delta_u: Displacement increment
        :param u_trial: Trial displacement
        :return:
        """
        delta_norm = np.linalg.norm(delta_u, ord=np.inf)
        u_trial_norm = np.linalg.norm(u_trial, ord=np.inf)
        return delta_norm, u_trial_norm

    def unbalance_norm_criteria(self):
        pass

    def energy_norm_criteria(self):
        pass
