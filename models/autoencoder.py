#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 9/02/22 14:27
@description:  
@version: 1.0
"""


from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.input_size, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, args.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, args.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
