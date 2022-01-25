#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 5/11/21 11:51
@description:  
@version: 1.0
"""


from ileenet.modeling.model import Model
from ileenet.integrator.Newmark import Newmark
from ileenet.integrator.TRBDF2 import TRBDF2
from ileenet.eigen.IMI import IMI
import numpy as np


class BaseModel:

    def __init__(self, args):
        self.args = args
        self.params = None  # Parameters
        self.gm_data = None  # Ground motion data

    def __call__(self, gm_data, params):
        self.register_gm_data(gm_data)
        self.register_params(params)
        return self.forward()

    def register_params(self, params):
        """
        Register model parameters
        :param params: Model parameters
        """
        self.params = np.array(params)

    def register_gm_data(self, gm_data):
        """
        Register ground motion data
        :param gm_data: Ground motion data
        """
        self.gm_data = gm_data

    def init_model(self):
        """
        2 *---* 4
          |   |
          |   |
          |   |
          |   |
        1 *   * 3
        :return: Model
        """
        # h_w, h_c, g, r_k, r_c, mu = self.params[0], self.params[1], self.params[2], \
        #     self.params[3], self.params[4], self.params[5]
        model = Model()
        model.mdl(2, 3)
        model.node(1, 0, 0)
        model.node(2, 0, self.args.h)
        model.node(3, self.params[0], 0)
        model.node(4, self.params[0], self.args.h)
        m_w = self.args.rou * self.args.h_w * self.args.b_w * self.args.h  # Lumped mass of wall
        m_c = self.args.rou * self.args.h_c * self.args.h_c * self.args.h  # Lumped mass of column
        f_p = self.args.alpha * self.args.f_y * (np.pi * self.args.d ** 2 / 4) * self.args.num_bar  # PT force
        m_p = f_p / 9800  # Equivalent mass due to PT
        m_w_all = m_w + m_p
        m_c_all = m_c + self.args.mb
        # print(m_c_all / m_w)
        model.mass(1, 0.5 * m_w_all, 0.5 * m_w_all, 0.5 * m_w_all)
        model.mass(2, 0.5 * m_w_all, 0.5 * m_w_all, 0.5 * m_w_all)
        model.mass(3, 0.5 * m_c_all, 0.5 * m_c_all, 0.5 * m_c_all)
        model.mass(4, 0.5 * m_c_all, 0.5 * m_c_all, 0.5 * m_c_all)
        a_w = self.args.h_w * self.args.b_w
        i_w = (self.args.b_w * self.args.h_w ** 3) / 12
        a_c = self.args.h_c * self.args.h_c
        i_c = (self.args.h_c * self.args.h_c ** 3) / 12
        model.hertz_contact_2(1, self.params[0],
                              self.args.k_c1, self.params[1] * self.args.k_c1,
                              self.args.c_c1, self.params[2] * self.args.c_c1,
                              self.params[3], self.args.g_tol)
        model.beam_element(1, 1, 2, a_w, self.args.e, self.args.rot * i_w)
        model.beam_element(2, 3, 4, a_c, self.args.e, self.args.rot * i_c)
        model.contact_element(3, 4, 2, 1)
        model.fix(1, 1, 1, 1)
        model.fix(3, 1, 1, 1)
        model.load(2, 0, -f_p, 0)
        return model

    def forward(self):
        """
        Solve model
        :return: solved model
        """
        model = self.init_model()
        model.ground_motion(int(self.args.gm_dir), self.gm_data, 1 / self.args.fs, self.args.factor)
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