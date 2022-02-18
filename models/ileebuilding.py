#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/11/21 16:39
@description:  
@version: 1.0
"""


from ileenet.modeling.model import Model
from ileenet.integrator.Newmark import Newmark
from ileenet.integrator.TRBDF2 import TRBDF2
from ileenet.eigen.IMI import IMI
import numpy as np


class ILEEBuilding:

    def __init__(self, args):
        self.args = args
        self.params = None  # Parameters
        self.gm_data = None  # Ground motion data
        self.dt = 0.  # Time interval
        self.scale = 0  # Scale

    def __call__(self, gm_data, params):
        self.register_params(params)
        self.register_gm_data(gm_data)
        return self.forward()

    def register_params(self, params):
        """
        Register model parameters
        :param params: Model parameters
        """
        self.params = np.array(params)

    def register_gm_data(self, gm_data, fs=256, scale=9800):
        """
        Register ground motion data
        :param gm_data: Ground motion data
        :param fs: Sampling frequency
        :param scale: Scale
        """
        self.gm_data = gm_data
        self.dt = 1 / fs
        self.scale = scale

    def init_model(self):
        """
        3 *---*---* 9
          |   | 6 |
          |   |   |
          |   |   |
          |   | 5 |
        2 *---*---* 8
          |   |   |
          |   |   |
          |   |   |
          |   |   |
        1 *   * 4 * 7
        :return: Model
        """
        model = Model()
        model.mdl(2, 3)
        model.node(1, -self.params[0], 0)
        model.node(2, -self.params[0], self.args.h)
        model.node(3, -self.params[1], self.args.h * 2)
        model.node(4, 0, 0)
        model.node(5, 0, self.args.h)
        model.node(6, 0, self.args.h * 2)
        model.node(7, self.params[0], 0)
        model.node(8, self.params[0], self.args.h)
        model.node(9, self.params[1], self.args.h * 2)
        m_w = self.args.rou * self.args.h_w * self.args.b_w * self.args.h  # Lumped mass of wall
        m_c = self.args.rou * self.args.h_c * self.args.h_c * self.args.h  # Lumped mass of column
        f_p = self.args.f_pi * (np.pi * self.args.d ** 2 / 4) * self.args.num_bar  # PT force
        m_p = f_p / 9800  # Equivalent mass due to PT
        m_w_all = m_w + m_p / 2
        m_c_all_1 = m_c + self.args.mb_1 / 2
        m_c_all_2 = m_c + self.args.mb_2 / 2
        model.mass(1, 0.5 * m_c_all_1, 0.5 * m_c_all_1, 0.5 * m_c_all_1)
        model.mass(2, 0.5 * (m_c_all_1 + m_c_all_2), 0.5 * (m_c_all_1 + m_c_all_2), 0.5 * (m_c_all_1 + m_c_all_2))
        model.mass(3, 0.5 * m_c_all_2, 0.5 * m_c_all_2, 0.5 * m_c_all_2)
        model.mass(4, 0.5 * m_w_all, 0.5 * m_w_all, 0.5 * m_w_all)
        model.mass(5, 1.0 * m_w_all, 1.0 * m_w_all, 1.0 * m_w_all)
        model.mass(6, 0.5 * m_w_all, 0.5 * m_w_all, 0.5 * m_w_all)
        model.mass(7, 0.5 * m_c_all_1, 0.5 * m_c_all_1, 0.5 * m_c_all_1)
        model.mass(8, 0.5 * (m_c_all_1 + m_c_all_2), 0.5 * (m_c_all_1 + m_c_all_2), 0.5 * (m_c_all_1 + m_c_all_2))
        model.mass(9, 0.5 * m_c_all_2, 0.5 * m_c_all_2, 0.5 * m_c_all_2)
        a_w = self.args.h_w * self.args.b_w
        i_w = (self.args.b_w * self.args.h_w ** 3) / 12
        a_c = self.args.h_c * self.args.h_c
        i_c = (self.args.h_c * self.args.h_c ** 3) / 12
        model.hertz_contact_2(1, self.params[0],
                              self.args.k_c1, self.params[2] * self.args.k_c1,
                              self.args.c_c1, self.params[3] * self.args.c_c1,
                              self.params[4], self.args.g_tol)
        model.hertz_contact_2(2, self.params[1],
                              self.args.k_c1, self.params[2] * self.args.k_c1,
                              self.args.c_c1, self.params[3] * self.args.c_c1,
                              self.params[4], self.args.g_tol)
        model.beam_element(1, 1, 2, a_c, self.args.e, self.params[5] * i_c)
        model.beam_element(2, 2, 3, a_c, self.args.e, self.params[5] * i_c)
        model.beam_element(3, 4, 5, a_w, self.args.e, self.params[6] * i_w)
        model.beam_element(4, 5, 6, a_w, self.args.e, self.params[6] * i_w)
        model.beam_element(5, 7, 8, a_c, self.args.e, self.params[5] * i_c)
        model.beam_element(6, 8, 9, a_c, self.args.e, self.params[5] * i_c)
        model.contact_element(7, 5, 2, 1)
        model.contact_element(8, 8, 5, 1)
        model.contact_element(9, 6, 3, 2)
        model.contact_element(10, 9, 6, 2)
        model.fix(1, 1, 1, 1)
        model.fix(4, 1, 1, 1)
        model.fix(7, 1, 1, 1)
        model.load(6, 0, -f_p, 0)
        return model

    def forward(self):
        """
        Solve model
        :return: solved model
        """
        model = self.init_model()
        model.ground_motion(int(self.args.gm_dir), self.gm_data, self.dt, self.scale)
        model.analysis('transient')
        model.system()
        mode = IMI(model)
        freq = mode()
        f1, f2 = freq[0], freq[1]
        alpha = 2 * f1 * f2 * self.args.xi / (f1 + f2)
        beta = 2 * self.args.xi / (f1 + f2)
        model.rayleigh(alpha, beta)
        integrator = Newmark(model) if self.args.integrator == 'Newmark' else TRBDF2(model)
        integrator.algorithm(self.args.algo)
        integrator.test(self.args.convergence, self.args.tol, self.args.max_iter)
        integrator()
        return model
