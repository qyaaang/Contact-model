#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 6/02/22 12:08
@description:  
@version: 1.0
"""


import argparse


def input_args():
    parser = argparse.ArgumentParser()
    # Fixed parameters
    parser.add_argument('--mb', default=5, type=float)
    parser.add_argument('--h_w', default=1500, type=float)  # Height of wall section
    parser.add_argument('--h_c', default=800, type=float)  # Height of column section
    parser.add_argument('--k_c1', default=10, type=float)  # Stiffness before contact
    parser.add_argument('--c_c1', default=0.001, type=float)  # Damping coefficient before contact
    parser.add_argument('--g_tol', default=1e-8, type=float)  # Gap tolerance
    parser.add_argument('--xi1', default=0.05, type=float)  # Crucial damping coefficient
    parser.add_argument('--xi2', default=0.0001, type=float)  # Crucial damping coefficient
    parser.add_argument('--b_w', default=150, type=float)  # Width of wall section
    parser.add_argument('--h', default=4000, type=float)  # Story height
    parser.add_argument('--alpha', default=1.0, type=float)  # PT coefficient
    parser.add_argument('--f_y', default=451, type=float)  # Yield strength of PT bar
    parser.add_argument('--d', default=25, type=float)  # Diameter of PT bar
    parser.add_argument('--num_bar', default=1, type=int)  # Number of PT bars
    parser.add_argument('--rou', default=2.5e-9, type=int)  # Density of material
    parser.add_argument('--e', default=3e4, type=int)  # Young's elasticity modulus of material
    parser.add_argument('--rot1', default=0.5, type=float)  # Rotation coefficient
    parser.add_argument('--rot2', default=0.05, type=float)  # Rotation coefficient
    # Analysis hyper-parameters
    parser.add_argument('--dataset', default='synthetic', type=str,
                        help='synthetic, PEER, test')  # Ground motion dataset
    parser.add_argument('--gm_dir', default='1', type=str)  # Ground motion direction
    parser.add_argument('--fs', default=100, type=float)  # Sampling frequency
    parser.add_argument('--factor', default=9800, type=float)  # GM factor
    parser.add_argument('--factor1', default=9800, type=float)  # GM factor
    parser.add_argument('--factor2', default=4000, type=float)  # GM factor
    parser.add_argument('--integrator', default='Newmark', type=str)  # Integrator
    parser.add_argument('--algo', default='momentum_newton', type=str)  # Iteration algorithm
    parser.add_argument('--convergence', default='disp', type=str)  # Convergence criteria
    parser.add_argument('--tol', default=1e-8, type=float)  # Convergence tolerance
    parser.add_argument('--max_iter', default=200, type=int)  # Maximum number of iterations
    # Criteria hyper-parameters
    parser.add_argument('--dof_tag', default=9, type=int)  # DOF tag
    parser.add_argument('--a_thresh', default=0.3, type=float)  # Threshold acceleration
    parser.add_argument('--d_thresh', default=3, type=float)  # Threshold drift ratio
    parser.add_argument('--beta', default=0.05, type=float)  # Logarithmic standard deviation
    args = parser.parse_args()
    return args
