#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 4/10/21 4:30 PM
@description:  
@version: 1.0
"""


import numpy as np


class TRBDF2:

    def __init__(self, model):
        self.model = model
        self.algo = None  # Algorithm
        self.algo_tag = None  # Algorithm tag
        self.convergence = None  # Convergence criteria
        self.tol = 0  # Iteration tolerance
        self.num_iter = 0  # Number of iterations
        # Parameters at the first sub-step
        self.a_0 = 16 / (self.model.d_t ** 2)
        self.a_1 = 4 / self.model.d_t
        self.a_2 = 8 / self.model.d_t
        self.a_3 = 1
        self.a_4 = 1
        self.a_5 = 0
        self.a_6 = self.model.d_t / 4
        self.a_7 = self.model.d_t / 4
        # Parameters at the second sub-step
        self.b_0 = 9 / (self.model.d_t ** 2)
        self.b_1 = 3 / self.model.d_t
        self.b_2 = 12 / (self.model.d_t ** 2)
        self.b_3 = 3 / (self.model.d_t ** 2)
        self.b_4 = 4 / self.model.d_t
        self.b_5 = 1 / self.model.d_t
        self.dof_i, self.dof_j, self.dof = self.model.active_dof_i, self.model.active_dof_j, self.model.active_dof

    def algorithm(self, tag):
        """
        Define algorithm tag
        :param tag: Algorithm tag
        """
        self.algo_tag = tag
        if len(self.model.c_ele.keys()) != 0:
            prob_tag = 'Contact'
        else:
            prob_tag = 'Non-contact'
        if tag == 'newton':
            from ileenet.algorithm.NewtonRaphson import NewtonRaphson
            self.algo = NewtonRaphson(prob_tag)

    def test(self, tag, tol, num_iter):
        """
        Define criteria for iteration convergence
        :param tag: Convergence criteria
        :param tol: Convergence tolerance
        :param num_iter: Maximum number of iterations
        """
        from ileenet.algorithm.Convergence import Convergence
        self.convergence = Convergence(tag)
        self.tol = tol
        self.num_iter = num_iter

    def solve_first_sub_step(self, step, k, u, f):
        """
        Solve first sub-step
        :param step: Current step
        :param k: Current stiffness matrix
        :param u: Displacements
        :param f: External loads
        :return: Displacements at middle time step
        """
        l = 0  # Iteration step
        u_trial = self.model.u[:, step]  # Initial trial displacement
        while True:
            l += 1
            u_trial_dof = u_trial[self.dof].reshape(-1, 1)  # Trial displacement at active dof
            u_trial_dof, delta_u = self.algo(u_trial_dof, k, f)  # Update trial displacement
            u_trial[self.dof] = u_trial_dof.reshape(-1)
            delta_norm, u_trial_norm = self.convergence(delta_u, u_trial_dof)
            if delta_norm < self.tol * u_trial_norm:
                u[:, step + 1] = u_trial_dof.reshape(-1)
                break
            if l > self.num_iter:
                raise RuntimeError('Newton-Raphson iterations did not converge at the first sub-step!')
        return u[:, step + 1]

    def solve_second_sub_step(self, step, k, u, f):
        """
        Solve second sub-step
        :param step: Current step
        :param k: Current stiffness matrix
        :param u: Displacements
        :param f: External loads
        """
        l = 0  # Iteration step
        u_trial = u[:, step]  # Initial trial displacement
        while True:
            l += 1
            u_trial_dof = u_trial.reshape(-1, 1)  # Trial displacement at active dof
            u_trial_dof, delta_u = self.algo(u_trial_dof, k, f)  # Update trial displacement
            u_trial = u_trial_dof.reshape(-1)
            delta_norm, u_trial_norm = self.convergence(delta_u, u_trial_dof)
            if delta_norm < self.tol * u_trial_norm:
                self.model.u[self.dof, step + 1] = u_trial_dof.reshape(-1)
                break
            if l > self.num_iter:
                raise RuntimeError('Newton-Raphson iterations did not converge at the second sub-step!')

    def solve_non_contact(self):
        """
        Solve non-contact equation
        """
        u, u_t, u_tt = self.model.u[self.dof, :], self.model.u_t[self.dof, :], self.model.u_tt[self.dof, :]
        u_m, u_m_t, u_m_tt = self.model.u[self.dof, :], self.model.u_t[self.dof, :], self.model.u_tt[self.dof, :]
        m = self.model.m[self.dof_i, self.dof_j]  # Mass matrix
        k = self.model.k[self.dof_i, self.dof_j]  # Stiffness matrix
        c = self.model.alpha_m * m + self.model.beta_k * k  # Damping matrix
        a_g = self.model.a_g[self.dof, :]  # Ground motion accelerations at nodes
        f = -np.dot(m, a_g) + self.model.f[self.dof, :]  # known equivalent nodal forces
        u_tt[:, 0] = np.dot(np.linalg.inv(m), f[:, 0] - np.dot(c, u_t[:, 0]) - np.dot(k, u[:, 0]))
        for step in range(self.model.accel_len - 1):
            k_hat_1 = self.a_0 * m + self.a_1 * c + k  # Equivalent stiffness matrix
            f_hat_1 = 0.5 * (f[:, step] + f[:, step + 1]) + \
                np.dot(m, self.a_0 * u[:, step] + self.a_2 * u_t[:, step] + self.a_3 * u_tt[:, step]) + \
                np.dot(c, self.a_1 * u[:, step] + self.a_4 * u_t[:, step] + self.a_5 * u_tt[:, step])
            # First sub-step, Newmark method
            u_m[:, step + 1] = self.solve_first_sub_step(step, k_hat_1, u_m, f_hat_1)
            u_m_tt[:, step + 1] = self.a_0 * (u_m[:, step + 1] - u[:, step]) - \
                self.a_2 * u_t[:, step] - self.a_3 * u_tt[:, step]
            u_m_t[:, step + 1] = u_t[:, step] + self.a_6 * u_tt[:, step] + self.a_7 * u_m_tt[:, step + 1]
            # Second sub-step, three-point Euler backward method
            k_hat_2 = self.b_0 * m + self.b_1 * c + k  # Equivalent stiffness matrix
            f_hat_2 = f[:, step + 1] + \
                np.dot(m, self.b_2 * u_m[:, step + 1] - self.b_3 * u[:, step] + self.b_4 * u_m_t[:, step + 1] -
                       self.b_5 * u_t[:, step]) + \
                np.dot(c, self.b_4 * u_m[:, step + 1] - self.b_5 * u[:, step])
            self.solve_second_sub_step(step, k_hat_2, u_m, f_hat_2)
            u[:, step + 1] = self.model.u[self.dof, step + 1]
            u_t[:, step + 1] = self.b_1 * u[:, step + 1] - self.b_4 * u_m[:, step + 1] + self.b_5 * u[:, step]
            u_tt[:, step + 1] = self.b_1 * u_t[:, step + 1] - self.b_4 * u_m_t[:, step + 1] + self.b_5 * u_t[:, step]
        self.model.u_t[self.dof, :] = u_t
        self.model.u_tt[self.dof, :] = u_tt

    def solve_first_sub_step_contact(self, step, m, c, k, u, f):
        """
        Solve first  sub-step for contact problem
        :param step: Current step
        :param m: Mass matrix
        :param c: Damping matrix
        :param k: Current stiffness matrix
        :param u: Displacement
        :param f: External loads
        :return: Displacements at middle time step
        """
        l = 0  # Iteration step
        u_trial = self.model.u[:, step]  # Initial trial displacement
        self.model.init_g()  # Initial gap
        while True:
            l += 1
            self.model.update_kc(u_trial, step)  # Update contact element stiffness matrix
            self.model.assemble_kc()  # Assemble contact stiffness matrix
            self.model.update_fc(u_trial, step)  # Assemble nodal contact force vector
            self.model.assemble_fc()  # Assemble nodal contact force vector
            kc = self.model.kc[self.dof_i, self.dof_j]  # Instant global contact stiffness matrix
            # print(kc[0, 0])
            fc = self.model.fc[self.dof]  # Instant contact force vector
            k_hat = self.a_0 * m + self.a_1 * c + k + kc  # Instant equivalent stiffness matrix
            u_trial_dof = u_trial[self.dof].reshape(-1, 1)  # Trial displacement at active dof
            u_trial_dof, delta_u = self.algo(u_trial_dof, k_hat, f, fc)  # Update trial displacement
            u_trial[self.dof] = u_trial_dof.reshape(-1)
            delta_norm, u_trial_norm = self.convergence(delta_u, u_trial_dof)
            if delta_norm < self.tol * u_trial_norm:
                u[:, step + 1] = u_trial_dof.reshape(-1)
                u_tmp = np.zeros_like(self.model.u[:, step])
                u_tmp[self.dof] = u[:, step + 1]
                self.model.update_g(u_tmp, step + 1)
                break
            if l > self.num_iter:
                raise RuntimeError('Newton-Raphson iterations did not converge at the first sub-step!')
        return u[:, step + 1]

    def solve_second_sub_step_contact(self, step, m, c, k, u, f):
        """
        Solve first sub-step for contact problem
        :param step: Current step
        :param m: Mass matrix
        :param c: Damping matrix
        :param k: Current stiffness matrix
        :param u: Displacements
        :param f: External loads
        """
        l = 0  # Iteration step
        u_trial = np.zeros_like(self.model.u[:, step])
        u_trial[self.dof] = u[:, step]  # Initial trial displacement
        self.model.init_g()
        while True:
            l += 1
            self.model.update_kc(u_trial, step)  # Update contact element stiffness matrix
            self.model.assemble_kc()  # Assemble contact stiffness matrix
            self.model.update_fc(u_trial, step)  # Assemble nodal contact force vector
            self.model.assemble_fc()  # Assemble nodal contact force vector
            kc = self.model.kc[self.dof_i, self.dof_j]  # Instant global contact stiffness matrix
            print(kc[0, 0])
            fc = self.model.fc[self.dof]  # Instant contact force vector
            k_hat = self.b_0 * m + self.b_1 * c + k + kc  # Instant equivalent stiffness matrix
            u_trial_dof = u_trial[self.dof].reshape(-1, 1)  # Trial displacement at active dof
            u_trial_dof, delta_u = self.algo(u_trial_dof, k_hat, f, fc)  # Update trial displacement
            u_trial[self.dof] = u_trial_dof.reshape(-1)
            delta_norm, u_trial_norm = self.convergence(delta_u, u_trial_dof)
            if delta_norm < self.tol * u_trial_norm:
                self.model.u[self.dof, step + 1] = u_trial_dof.reshape(-1)
                self.model.update_g(self.model.u[:, step + 1], step + 1)
                break
            if l > self.num_iter:
                raise RuntimeError('Newton-Raphson iterations did not converge at the second sub-step!')

    def solve_contact(self):
        """
        Solve contact equation
        """
        u, u_t, u_tt = self.model.u[self.dof, :], self.model.u_t[self.dof, :], self.model.u_tt[self.dof, :]
        u_m, u_m_t, u_m_tt = self.model.u[self.dof, :], self.model.u_t[self.dof, :], self.model.u_tt[self.dof, :]
        m = self.model.m[self.dof_i, self.dof_j]  # Mass matrix
        k = self.model.k[self.dof_i, self.dof_j]  # Stiffness matrix
        self.model.assemble_cc()
        cc = self.model.cc[self.dof_i, self.dof_j]  # Contact damping matrix
        c = self.model.alpha_m * m + self.model.beta_k * k + cc  # Damping matrix
        a_g = self.model.a_g[self.dof, :]  # Ground motion accelerations at nodes
        f = -np.dot(m, a_g) + self.model.f[self.dof, :]  # known equivalent nodal forces
        u_tt[:, 0] = np.dot(np.linalg.inv(m), f[:, 0] - np.dot(c, u_t[:, 0]) - np.dot(k, u[:, 0]))
        for step in range(self.model.accel_len - 1):
            print(step)
            f_hat_1 = 0.5 * (f[:, step] + f[:, step + 1]) + \
                np.dot(m, self.a_0 * u[:, step] + self.a_2 * u_t[:, step] + self.a_3 * u_tt[:, step]) + \
                np.dot(c, self.a_1 * u[:, step] + self.a_4 * u_t[:, step] + self.a_5 * u_tt[:, step])
            # First sub-step, Newmark method
            u_m[:, step + 1] = self.solve_first_sub_step_contact(step, m, c, k, u_m, f_hat_1)
            u_m_tt[:, step + 1] = self.a_0 * (u_m[:, step + 1] - u[:, step]) - \
                self.a_2 * u_t[:, step] - self.a_3 * u_tt[:, step]
            u_m_t[:, step + 1] = u_t[:, step] + self.a_6 * u_tt[:, step] + \
                self.a_7 * u_m_tt[:, step + 1]
            # Second sub-step, three-point Euler backward method
            f_hat_2 = f[:, step + 1] + \
                np.dot(m, self.b_2 * u_m[:, step + 1] - self.b_3 * u[:, step] +
                       self.b_4 * u_m_t[:, step + 1] - self.b_5 * u_t[:, step]) + \
                np.dot(c, self.b_4 * u_m[:, step + 1] - self.b_5 * u[:, step])
            self.solve_second_sub_step_contact(step, m, c, k, u_m, f_hat_2)
            u[:, step + 1] = self.model.u[self.dof, step + 1]
            u_t[:, step + 1] = self.b_1 * u[:, step + 1] - self.b_4 * u_m[:, step + 1] + \
                self.b_5 * u[:, step]
            u_tt[:, step + 1] = self.b_1 * u_t[:, step + 1] - self.b_4 * u_m_t[:, step + 1] + \
                self.b_5 * u_t[:, step]
        self.model.u_t[self.dof, :] = u_t
        self.model.u_tt[self.dof, :] = u_tt

    def integrator(self):
        """
        TRBDF2 integral scheme
        """
        if len(self.model.c_ele.keys()) == 0:
            self.solve_non_contact()
        if len(self.model.c_ele.keys()) != 0:
            self.solve_contact()
