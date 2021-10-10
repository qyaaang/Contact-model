#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/10/21 6:45 PM
@description:  
@version: 1.0
"""


import numpy as np


class ModifiedNewton:

    def __init__(self, tag):
        self.tag = tag  # Problem tag

    def __call__(self, *args, **kwargs):
        if self.tag == 'Contact':
            u_trial, delta_u = self.solve_current_step_contact(*args)
        else:
            u_trial, delta_u = self.solve_current_step(*args)
        return u_trial, delta_u

    def get_init_k(self, model, c_0, c_1, step, m, c, k):
        """
        Get initial stiffness matrix
        :param model: Model
        :param c_0: Parameter 1
        :param c_1: Parameter 2
        :param step: Current step
        :param m: Mass matrix
        :param c: Damping matrix
        :param k: Stiffness matrix
        :return: Initial stiffness matrix
        """
        u_trial = model.u[:, step]  # Initial trial displacement
        model.update_g(u_trial, step)  # Update gap
        model.update_kc(step)  # Update contact element stiffness matrix
        model.assemble_kc()  # Assemble contact stiffness matrix
        kc = model.kc[model.active_dof_i, model.active_dof_j]  # Instant global contact stiffness matrix
        k_0_hat = c_0 * m + c_1 * c + k + kc  # Instant equivalent stiffness matrix
        return k_0_hat

    def solve_current_step(self, u_trial, k_0, f):
        """
        Slove current step for non-contact problem
        :param u_trial: Trial displacements
        :param k_0: Initial stiffness matrix
        :param f: External loads
        :return: Updated trial displacements, displacement increment
        """
        r = f.reshape(-1, 1) - np.dot(k_0, u_trial)  # Unbalance force
        delta_u = np.dot(np.linalg.inv(k_0), r)  # Displacement increment
        u_trial += delta_u  # Update trial displacement
        return u_trial, delta_u

    def solve_current_step_contact(self, u_trial, k_0, f, fc):
        """
        Slove current step for contact problem
        :param u_trial: Trial displacements
        :param k_0: Initial stiffness matrix
        :param f: External loads
        :param fc: Contact force
        :return: Updated trial displacements, displacement increment
        """
        r = f.reshape(-1, 1) - (np.dot(k_0, u_trial) - fc.reshape(-1, 1))  # Unbalance force
        delta_u = np.dot(np.linalg.inv(k_0), r)  # Displacement increment
        u_trial += delta_u  # Update trial displacement
        return u_trial, delta_u
