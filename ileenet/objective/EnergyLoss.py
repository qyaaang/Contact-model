#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 18/10/21 2:40 PM
@description:  
@version: 1.0
"""


import numpy as np


class EnergyLoss:

    def __init__(self, model):
        self.model = model
        self.contact_time = {}
        self.dof_i, self.dof_j, self.dof = self.model.active_dof_i, self.model.active_dof_j, self.model.active_dof

    def __call__(self, *args, **kwargs):
        self.get_contact_time_idx()
        energy_loss = self.contact_energy_loss()
        return energy_loss

    def get_contact_time_idx(self):
        """
        Get contact time index
        """
        for ele in self.model.g.keys():
            gap = self.model.g[ele]  # Gap history of contact element
            self.contact_time[ele] = {}
            n = 0  # Contact time
            for idx in range(len(gap) - 1):
                if gap[idx] > 0 > gap[idx + 1]:
                    n += 1
                    self.contact_time[ele][n] = [idx]
                if gap[idx] < 0 < gap[idx + 1]:
                    self.contact_time[ele][n].append(idx + 1)

    def get_contact_velocity(self, idx_1, idx_2):
        """
        Get velocity before and after contact
        :param idx_1: Time index before contact
        :param idx_2: Time index after contact
        :return: Velocity before and after contact
        """
        v_1 = self.model.u_t[self.dof, idx_1]
        v_2 = self.model.u_t[self.dof, idx_2]
        return v_1, v_2

    def get_contact_disp(self, idx_1, idx_2):
        """
        Get displacement before and after contact
        :param idx_1: Time index before contact
        :param idx_2: Time index after contact
        :return: Displacement before and after contact
        """
        u_1 = self.model.u[self.dof, idx_1]
        u_2 = self.model.u[self.dof, idx_2]
        return u_1, u_2

    def kinetic_energy(self, v):
        """
        Compute system kinetic energy
        :param v: Velocity
        :return: System kinetic energy
        """
        m = self.model.m[self.dof_i, self.dof_j]
        return 0.5 * np.dot(np.dot(v.T, m), v)

    def elastic_energy(self, u):
        """
        Compute system elastic energy
        :param u: Displacement
        :return: System elastic energy
        """
        k = self.model.k[self.dof_i, self.dof_j]
        return 0.5 * np.dot(np.dot(u.T, k), u)

    def inertia_energy(self, time_idx_1, time_idx_2):
        """
        Compute inertia energy during a contact period
        :param time_idx_1: Time index contact begins
        :param time_idx_2: Time index contact ends
        :return: Inertia energy
        """
        a_g = self.model.a_g[self.dof, :]
        u_tt = self.model.u_tt[self.dof, :]
        u_t = self.model.u_t[self.dof, :]
        d_t = self.model.d_t
        m = self.model.m[self.dof_i, self.dof_j]
        energy = 0.
        for time_idx in range(time_idx_1, time_idx_2):
            meta_energy = -np.dot(np.dot(u_t[:, time_idx].T, m), a_g[:, time_idx] + u_tt[:, time_idx]) * d_t
            energy += meta_energy
        return energy

    def damping_energy(self, time_idx_1, time_idx_2):
        """
        Compute damping energy during a contact period
        :param time_idx_1: Time index contact begins
        :param time_idx_2: Time index contact ends
        :return: Damping energy
        """
        u_t = self.model.u_t[self.dof, :]
        d_t = self.model.d_t
        m = self.model.m[self.dof_i, self.dof_j]
        k = self.model.k[self.dof_i, self.dof_j]
        c = self.model.alpha_m * m + self.model.beta_k * k
        energy = 0.
        for time_idx in range(time_idx_1, time_idx_2):
            meta_energy = np.dot(np.dot(u_t[:, time_idx].T, c), u_t[:, time_idx]) * d_t
            energy += meta_energy
        return energy

    def contact_energy_loss(self):
        """
        Compute energy loss due to contact
        :return: Kinetic energy loss
        """
        energy_loss = 0.
        for ele in self.contact_time.keys():
            for contact_idx in self.contact_time[ele].keys():
                try:
                    time_idx_1 = self.contact_time[ele][contact_idx][0]
                    time_idx_2 = self.contact_time[ele][contact_idx][1]
                    v_1, v_2 = self.get_contact_velocity(time_idx_1, time_idx_2)
                    u_1, u_2 = self.get_contact_disp(time_idx_1, time_idx_2)
                    e_k1 = self.kinetic_energy(v_1)  # Kinetic energy before contact
                    e_k2 = self.kinetic_energy(v_2)  # Kinetic energy after contact
                    e_e1 = self.elastic_energy(u_1)  # Elastic energy before contact
                    e_e2 = self.elastic_energy(u_2)  # Elastic energy after contact
                    delta_e_m = (e_k2 + e_e2) - (e_k1 + e_e1)  # Change in mechanic energy
                    e_i = self.inertia_energy(time_idx_1, time_idx_2)  # Inertia energy
                    e_d = self.damping_energy(time_idx_1, time_idx_2)  # Damping energy
                    e_c = delta_e_m - e_i - e_d  # Energy loss due to contact
                    if e_c > 0:
                        e_c = 0
                except:
                    e_c = 0
                energy_loss += e_c
                # print(delta_e_m, e_i, e_d, e_c)
        # print(energy_loss)
        return energy_loss
