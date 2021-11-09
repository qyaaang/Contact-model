#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 5/11/21 14:06
@description:  
@version: 1.0
"""


class SGD:

    def __init__(self, lr, momentum):
        self.lr = lr  # Learning rate
        self.momentum = momentum  # Momentum

    def step(self, params, grads, v):
        """
        Update parameters
        :param params: Current parameters
        :param grads: Current gradients
        :param v: Current velocity
        :return: Updated parameters
        """
        v = self.momentum * v - self.lr * grads
        params += v
        return params, v
