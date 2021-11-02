#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 1/11/21 15:40
@description:  
@version: 1.0
"""


import numpy as np


class IMI:

    def __init__(self, model):
        self.model = model
        self.m = None  # Element mass matrix
        self.k = None  # Element stiffness matrix
        self.num_dof = self.model.k_ele['1'].shape[0]
        self.dof_i, self.dof_j = self.get_activate_dof()

    def __call__(self, *args, **kwargs):
        mode_shape = self.mode(*args)
        return self.freq(mode_shape)

    def get_m(self, ele_tag):
        """
        Get element mass matrix
        :param ele_tag: Element tag
        :return: Element mass matrix
        """
        m = np.zeros((self.num_dof, self.num_dof))
        for node in self.model.ele[ele_tag]:
            node_ = node - 2 * (int(ele_tag) - 1)  # Node changes
            dof_i = [[dof] for dof in range(3 * int(node_) - 3, 3 * int(node_))]
            dof_j = [[dof for dof in range(3 * int(node_) - 3, 3 * int(node_))]]
            m[dof_i, dof_j] = np.diag(self.model.masses[str(node)])
        self.m = m[self.dof_i, self.dof_j]

    def get_k(self, ele_tag):
        """
        Get element stiffness matrix
        :param ele_tag: Element tag
        :return: Element stiffness matrix
        """
        k = self.model.k_ele[ele_tag]
        self.k = k[self.dof_i, self.dof_j]

    def get_activate_dof(self):
        """
        Get activate DOF
        :return:
        """
        all_dof = [dof + 1 for dof in range(self.num_dof)]
        active_dof = list(set(all_dof) - {1, 2, 3})
        active_dof_i = [[dof - 1] for dof in active_dof]
        active_dof_j = [[dof - 1 for dof in active_dof]]
        return active_dof_i, active_dof_j

    def init_a(self):
        """
        Initialize mode shape vector
        :return: Initial mode shape vector
        """
        # a = np.zeros((len(self.dof_i), 1))
        # a[0, 0] = 1.
        a = np.random.rand(len(self.dof_i), 1)
        return a

    def matrix_iter(self, num_iter, e):
        """
        Matrix iteration
        :param num_iter: Number of iterations
        :param e: Iteration matrix
        :return: Normalized mode shape vector
        """
        a = self.init_a()
        for _ in range(num_iter):
            b = np.dot(e, a)
            m_bar = b[np.argmax(np.abs(b))]
            a = b / m_bar
        return a

    def mode(self, ele_tag, num_iter=100):
        """
        Compute mode shape matrix
        :param ele_tag: Element tag
        :param num_iter: Number of iterations
        :return: Mode shape matrix
        """
        self.get_m(ele_tag)
        self.get_k(ele_tag)
        e = np.dot(np.linalg.inv(self.k), self.m)  # Iteration matrix
        i = np.eye(self.k.shape[0])
        s = i  # Clean matrix
        mode_shape = np.zeros((self.k.shape[0], self.k.shape[0]))
        for idx in range(self.k.shape[0]):
            e = np.dot(e, s)
            phai = self.matrix_iter(num_iter, e)
            m_star = np.dot(np.dot(phai.T, self.m), phai)
            s -= np.dot(np.dot(phai, phai.T), self.m) / m_star.item()
            mode_shape[:, idx] = phai[:, 0]
        return mode_shape

    def freq(self, mode_shape):
        """
        Compute frequency
        :param mode_shape: Mode shape matrix
        :return: Frequency
        """
        freqs = np.zeros(self.k.shape[0])
        for idx in range(self.k.shape[0]):
            phai = mode_shape[:, idx]
            m_star = np.dot(np.dot(phai.T, self.m), phai)
            k_star = np.dot(np.dot(phai.T, self.k), phai)
            freqs[idx] = np.sqrt(k_star / m_star) / (2 * np.pi)
        return freqs
