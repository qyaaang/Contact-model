#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 29/12/21 10:50
@description:  
@version: 1.0
"""


import numpy as np


class MomentumNewton:

    def __init__(self, tag):
        self.tag = tag  # Problem tag

    def __call__(self, *args, **kwargs):
        if self.tag == 'Contact':
            u_trial, delta_u = self.solve_current_step_contact(*args)
        else:
            u_trial, delta_u = self.solve_current_step(*args)
        return u_trial, delta_u

    def solve_current_step(self, u_trial, k, f):
        """
        Solve current step
        :param u_trial: Trial displacements
        :param k: Current stiffness matrix
        :param f: External loads
        :return: Updated trial displacements, displacement increment
        """
        r = f.reshape(-1, 1) - np.dot(k, u_trial)  # Unbalance force
        delta_u = np.dot(np.linalg.inv(k), r)  # Displacement increment
        u_trial += delta_u  # Update trial displacement
        return u_trial, delta_u

    def solve_current_step_contact(self, u_trial, k, f, fc):
        """
        Slove current step for contact problem
        :param u_trial: Trial displacements
        :param k: Current stiffness matrix
        :param f: External loads
        :param fc: Contact force
        :return: Updated trial displacements, displacement increment
        """
        r = f.reshape(-1, 1) - (np.dot(k, u_trial) - fc.reshape(-1, 1))  # Unbalance force
        delta_u = np.dot(np.linalg.inv(k), r)  # Displacement increment
        u_trial += delta_u  # Update trial displacement
        return u_trial, delta_u
