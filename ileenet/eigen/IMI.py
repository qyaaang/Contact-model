#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 4/11/21 18:05
@description:  
@version: 1.0
"""


import numpy as np


class IMI:

    def __init__(self, model):
        self.model = model
        self.m = None  # Global mass matrix
        self.k = None  # Global stiffness matrix
        self.dof_i, self.dof_j, _ = self.model.get_activate_dof()

    def __call__(self, *args, **kwargs):
        mode_shape = self.mode()
        return self.freq(mode_shape)

    def matrix_iter(self, num_iter, e):
        """
        Matrix iteration
        :param num_iter: Number of iterations
        :param e: Iteration matrix
        :return: Normalized mode shape vector
        """
        a = np.random.rand(len(self.dof_i), 1)  # Initialize mode shape vector
        for _ in range(num_iter):
            b = np.dot(e, a)
            m_bar = b[np.argmax(np.abs(b))]
            a = b / m_bar
        return a

    def mode(self, num_iter=100):
        """
        Compute mode shape matrix
        :param num_iter: Number of iterations
        :return: Mode shape matrix
        """
        if len(self.model.c_ele.keys()) != 0:
            self.model.update_g(self.model.u[:, 0], 0)
            self.model.update_kc(0)
            self.model.assemble_kc()
            self.k = (self.model.k + self.model.kc)[self.dof_i, self.dof_j]
        else:
            self.k = self.model.k[self.dof_i, self.dof_j]
        self.m = self.model.m[self.dof_i, self.dof_j]
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
