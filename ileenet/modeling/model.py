#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 1/09/21 5:38 PM
@description:  
@version: 1.0
"""


import numpy as np


class Model:

    def __init__(self):
        self.ndm = None  # Spatial dimension of problem
        self.ndf = None  # Number of degrees-of-freedom at nodes
        self.nodes = {}  # Node coordinates
        self.mat = {}  # Materials
        self.ele, self.c_ele = {}, {}  # Elements, contact elements
        self.k_ele, self.kc_ele = {}, {}  # Element, contact element stiffness matrices
        self.cc_ele = {}  # Contact element damping matrices
        self.g_0, self.g = {}, {}  # Initial, instant gap
        self.alpha_c = None  # Contact stiffness coefficient
        self.k = None  # Global elastic stiffness matrix
        self.kc = None  # Global contact stiffness matrix
        self.cc = None  # Global contact damping matrix
        self.alpha_m, self.beta_k = 0, 0  # Rayleigh damping factors
        self.cstrts = {}  # Constraint conditions
        self.fixed_dof = []  # Fixed DOF
        self.masses = {}  # Nodal masses
        self.loads, self.fc_ele = {}, {}  # Local nodal force, contact force vectors
        self.m = None  # Global mass matrix
        self.gm = {}  # Ground motions
        self.accel_len = 0  # Acceleration series length
        self.a_g = None  # Ground motion at nodes
        self.d_t = 0  # Time interval
        self.analysis_tag = None  # Analysis tag
        self.f, self.fc = None, None  # Global nodal force, contact force vector
        self.u, self.u_t, self.u_tt = None, None, None  # Displacement, velocity, acceleration vector
        self.active_dof_i, self.active_dof_j, self.active_dof = None, None, None

    def mdl(self, ndm, ndf):
        """
        Define spatial dimension of problem and number of degrees-of-freedom at nodes
        :param ndm: Spatial dimension of problem
        :param ndf: Number of degrees-of-freedom at nodes
        """
        self.ndm = ndm
        self.ndf = ndf

    def node(self, tag, x, y):
        """
        Create nodes of modeling
        :param tag: Node tag
        :param x: x coordinate
        :param y: y coordinate
        """
        self.nodes[str(tag)] = [x, y]

    def get_num_node(self):
        """
        Get number of nodes
        :return: Number of nodes
        """
        return len(self.nodes.keys())

    def get_num_dof(self):
        """
        Get number of DOF
        :return: Number of DOF
        """
        num_node = self.get_num_node()  # Number of nodes
        return num_node * self.ndf

    def get_activate_dof(self):
        """
        Get activate DOF
        :return:
        """
        num_dof = self.get_num_dof()
        all_dof = [dof + 1 for dof in range(num_dof)]
        active_dof = list(set(all_dof) - set(self.fixed_dof))
        active_dof_i = [[dof - 1] for dof in active_dof]
        active_dof_j = [[dof - 1 for dof in active_dof]]
        active_dof = [dof - 1 for dof in active_dof]
        return active_dof_i, active_dof_j, active_dof

    def mass(self, node_tag, *mass_values):
        """
        Create nodal masses
        :param node_tag: Node mass
        :param mass_values:
        """
        self.masses[str(node_tag)] = mass_values
        if len(mass_values) != self.ndf:
            raise Exception('The length of nodal mass vectors must be equal to the ndf.')

    def hertz_contact_1(self, mat_tag, g_p, k_c1, k_c2, c_c, mu, gap_tol):
        """
        Create hertz contact model
        :param mat_tag: Material tag
        :param g_p: Gap between nodes
        :param k_c1: Contact stiffness before contact
        :param k_c2: Contact stiffness after contact
        :param c_c: Constant damping coefficient
        :param mu: Friction coefficient
        :param gap_tol: Gap tolerance
        """
        self.mat[str(mat_tag)] = {'gap': g_p,
                                  'k_c1': k_c1,
                                  'k_c2': k_c2,
                                  'c_c': c_c,
                                  'mu': mu,
                                  'gap_tol': gap_tol
                                  }

    def hertz_contact_2(self, mat_tag, g_p, k_c1, k_c2, c_c1, c_c2, mu, gap_tol):
        """
        Create hertz contact model
        :param mat_tag: Material tag
        :param g_p: Gap between nodes
        :param k_c1: Contact stiffness before contact
        :param k_c2: Contact stiffness after contact
        :param c_c1: Contact damping coefficient before contact
        :param c_c2: Contact damping coefficient after contact
        :param mu: Friction coefficient
        :param gap_tol: Gap tolerance
        """
        self.mat[str(mat_tag)] = {'gap': g_p,
                                  'k_c1': k_c1,
                                  'k_c2': k_c2,
                                  'c_c1': c_c1,
                                  'c_c2': c_c2,
                                  'mu': mu,
                                  'gap_tol': gap_tol
                                  }

    def beam_element(self, tag, node_i, node_j, a, e, i_z):
        """
        Create a 2D elastic Euler beam element
        :param tag: Element tag
        :param node_i: End node i
        :param node_j: End node j
        :param a: Cross-sectional area of element
        :param e: Young's Modulus
        :param i_z: Second moment of area about the local z-axis
        """
        self.ele[str(tag)] = [node_i, node_j]
        x_i, y_i = self.nodes[str(node_i)][0], self.nodes[str(node_i)][1]
        x_j, y_j = self.nodes[str(node_j)][0], self.nodes[str(node_j)][1]
        delta_x = x_j - x_i
        delta_y = y_j - y_i
        l_ele = np.sqrt(delta_x ** 2 + delta_y ** 2)  # Element length
        c = delta_x / l_ele
        s = delta_y / l_ele
        t = np.array([[c,  s, 0, 0,  0, 0],
                      [-s, c, 0, 0,  0, 0],
                      [0,  0, 1, 0,  0, 0],
                      [0,  0, 0, c,  s, 0],
                      [0,  0, 0, -s, c, 0],
                      [0,  0, 0, 0,  0, 1]])  # Transfer matrix
        e_a = e * a / l_ele
        e_i1 = e * i_z / l_ele
        e_i2 = 6.0 * e * i_z / (l_ele ** 2)
        e_i3 = 12.0 * e * i_z / (l_ele ** 3)
        k_ele = np.array([[e_a,     0,        0, -e_a,     0,        0],
                          [0,    e_i3,     e_i2,    0, -e_i3,     e_i2],
                          [0,    e_i2, 4 * e_i1,    0, -e_i2, 2 * e_i1],
                          [-e_a,    0,        0,  e_a,     0,        0],
                          [0,   -e_i3,    -e_i2,    0,  e_i3,    -e_i2],
                          [0,    e_i2, 2 * e_i1,    0, -e_i2, 4 * e_i1]])  # Local element stiffness matrix
        self.k_ele[str(tag)] = np.dot(np.dot(t.T, k_ele), t)  # Global element stiffness matrix

    def contact_element(self, tag, node_c, node_r, mat_tag):
        """
        Create a 2D contact element
        :param tag: Element tag
        :param node_c: Constrained node
        :param node_r: Restrained node
        :param mat_tag: Material tag
        """
        self.ele[str(tag)] = [node_c, node_r]
        self.c_ele[str(tag)] = {'node': [node_c, node_r],
                                'mat_tag': str(mat_tag)}
        self.k_ele[str(tag)] = np.zeros((2 * self.ndf, 2 * self.ndf))
        self.g_0[str(tag)] = self.mat[str(mat_tag)]['gap']

    def contact_damping(self):
        """
        Contact element damping matrix
        """
        n_c = np.hstack((np.eye(self.ndf), - np.eye(self.ndf)))
        t_c = np.array([[1], [0], [0]])
        for ele in self.c_ele.keys():
            mat_tag = self.c_ele[ele]['mat_tag']
            c_n = self.mat[mat_tag]['c_c']
            beta_c = np.array(([-c_n], [0], [0]))
            self.cc_ele[ele] = np.dot(np.dot(np.dot(n_c.T, beta_c), t_c.T), n_c)

    def init_g(self):
        """
        Initial gap
        """
        for ele in self.c_ele.keys():
            g = self.g_0[ele]
            self.g[ele][0] = g

    def update_g(self, u, step):
        """
        Update instant gap between nodes
        :param u: Nodal displacement to be solved
        :param step: Time step
        """
        for ele in self.c_ele.keys():
            node_c = self.c_ele[ele]['node'][0]
            node_r = self.c_ele[ele]['node'][1]
            u_c = u[(node_c - 1) * 3]
            u_r = u[(node_r - 1) * 3]
            g_0 = self.g[ele][0]
            g_step_1 = u_r - u_c + g_0  # Trial gap at step + 1
            # print(u_c, u_r)
            # if step == 0:
            #     g = self.g_0[ele]
            # else:
            #     g = u_r - u_c + self.g[ele][step - 1]
            self.g[ele][step + 1] = g_step_1

    def instant_k_c(self, ele, step):
        """
        Get instant contact stiffness value
        :param ele: Contact element tag
        :param step: Time step
        :return:
        """
        mat_tag = self.c_ele[ele]['mat_tag']
        gap_tol = self.mat[mat_tag]['gap_tol']  # Gap tolerance
        k_c1 = self.mat[mat_tag]['k_c1']
        k_c2 = self.mat[mat_tag]['k_c2']
        g_step_1 = self.g[ele][step + 1]  # Trial gap at step + 1
        if g_step_1 > gap_tol:
            k_c = k_c1
        else:
            k_c = k_c2
        return k_c

    def instant_c_c(self, ele, step):
        """
        Get instant contact damping coefficient
        :param ele: Contact element tag
        :param step: Time step
        :return:
        """
        # node_c = self.c_ele[ele]['node'][0]
        # node_r = self.c_ele[ele]['node'][1]
        mat_tag = self.c_ele[ele]['mat_tag']
        gap_tol = self.mat[mat_tag]['gap_tol']
        c_c1 = self.mat[mat_tag]['c_c1']
        c_c2 = self.mat[mat_tag]['c_c2']
        # u_c = u[(node_c - 1) * 3]
        # u_r = u[(node_r - 1) * 3]
        # g_step = self.g[ele][step]  # Gap at last time step
        g_step_1 = self.g[ele][step + 1]  # Trial gap at step + 1
        if g_step_1 > gap_tol:
            c_c = c_c1
        else:
            c_c = c_c2
        return c_c

    def update_kc(self, step):
        """
        Update instant contact element stiffness matrix
        :param step: Time step
        """
        n_c = np.hstack((np.eye(self.ndf), - np.eye(self.ndf)))
        t_c = np.array([[1], [0], [0]])
        for ele in self.c_ele.keys():
            mat_tag = self.c_ele[ele]['mat_tag']
            mu = self.mat[mat_tag]['mu']
            alpha_c = np.array([[1], [-mu], [0]])
            k_c = self.instant_k_c(ele, step)
            self.kc_ele[ele] = k_c * np.dot(np.dot(np.dot(n_c.T, alpha_c), t_c.T), n_c)

    def update_cc(self, step):
        """
        Update instant contact element damping matrix
        :param step: Time step
        """
        n_c = np.hstack((np.eye(self.ndf), - np.eye(self.ndf)))
        t_c = np.array([[1], [0], [0]])
        for ele in self.c_ele.keys():
            c_c = self.instant_c_c(ele, step)
            beta_c = np.array(([-c_c], [0], [0]))
            self.cc_ele[ele] = np.dot(np.dot(np.dot(n_c.T, beta_c), t_c.T), n_c)

    def update_fc(self, step):
        """
        Update instant contact force
        :param step: Time step
        """
        n_c = np.hstack((np.eye(self.ndf), - np.eye(self.ndf)))
        for ele in self.c_ele.keys():
            mat_tag = self.c_ele[ele]['mat_tag']
            mu = self.mat[mat_tag]['mu']
            alpha_c = np.array([[1], [-mu], [0]])
            k_c = self.instant_k_c(ele, step)
            # g_step = self.g[ele][step]
            g_0 = self.g[ele][0]
            self.fc_ele[ele] = k_c * np.dot(n_c.T, alpha_c) * g_0

    def fix(self, node_tag, *cstrts):
        """
        Fix degrees of freedom
        :param node_tag: Node needs to be fixed
        :param cstrts: Constraint conditions
        """
        self.cstrts[str(node_tag)] = cstrts
        if len(cstrts) == self.ndf:
            for idx, cstrt in enumerate(cstrts):
                if cstrt == 1:
                    self.fixed_dof.append(self.ndf * (node_tag - 1) + idx + 1)
        else:
            raise Exception('The length of constraint conditions must be equal to the ndf.')

    def load(self, node_tag, *load_values):
        """
        Define nodal force
        :param node_tag: Node at which loads are applied
        :param load_values: Load values
        """
        self.loads[str(node_tag)] = load_values
        if len(load_values) != self.ndf:
            raise Exception('The length of nodal load vectors must be equal to the ndf.')

    def ground_motion(self, gm_dir, accel_series, d_t, factor):
        """
        Define ground motion data
        :param gm_dir: Ground motion direction
        :param accel_series: Acceleration
        :param d_t: Time interval
        :param factor: Factor
        """
        self.gm[str(gm_dir)] = accel_series * factor
        self.accel_len = len(accel_series)
        self.d_t = d_t

    def assemble_k(self, num_dof):
        """
        Assemble element stiffness matrix
        :param num_dof: Number of DOF
        """
        self.k = np.zeros((num_dof, num_dof))  # Global stiffness matrix
        for ele in self.k_ele.keys():
            node_i, node_j = self.ele[ele][0], self.ele[ele][1]
            ele_dof_i = [[3 * node_i - 3], [3 * node_i - 2], [3 * node_i - 1],
                         [3 * node_j - 3], [3 * node_j - 2], [3 * node_j - 1]]
            ele_dof_j = [[3 * node_i - 3, 3 * node_i - 2, 3 * node_i - 1,
                          3 * node_j - 3, 3 * node_j - 2, 3 * node_j - 1]]
            self.k[ele_dof_i, ele_dof_j] = self.k[ele_dof_i, ele_dof_j] + self.k_ele[ele]

    def assemble_kc(self):
        """
        Assemble contact element stiffness matrix
        """
        num_dof = self.get_num_dof()
        self.kc = np.zeros((num_dof, num_dof))  # Global contact stiffness matrix
        for ele in self.kc_ele.keys():
            node_c, node_r = self.c_ele[ele]['node'][0], self.c_ele[ele]['node'][1]
            ele_dof_c = [[3 * node_c - 3], [3 * node_c - 2], [3 * node_c - 1],
                         [3 * node_r - 3], [3 * node_r - 2], [3 * node_r - 1]]
            ele_dof_r = [[3 * node_c - 3, 3 * node_c - 2, 3 * node_c - 1,
                          3 * node_r - 3, 3 * node_r - 2, 3 * node_r - 1]]
            self.kc[ele_dof_c, ele_dof_r] = self.kc[ele_dof_c, ele_dof_r] + self.kc_ele[ele]

    def assemble_m(self, num_dof):
        """
        Assemble mass matrix
        :param num_dof: Number of DOF
        """
        self.m = np.zeros((num_dof, num_dof))  # Global mass matrix
        for node in self.nodes.keys():
            dof_i = [[dof] for dof in range(3 * int(node) - 3, 3 * int(node))]
            dof_j = [[dof for dof in range(3 * int(node) - 3, 3 * int(node))]]
            self.m[dof_i, dof_j] = np.diag(self.masses[node])

    def assemble_cc(self):
        """
        Assemble contact damping matrix
        """
        num_dof = self.get_num_dof()
        self.cc = np.zeros((num_dof, num_dof))  # Global damping matrix
        self.contact_damping()
        for ele in self.cc_ele.keys():
            node_c, node_r = self.c_ele[ele]['node'][0], self.c_ele[ele]['node'][1]
            ele_dof_c = [[3 * node_c - 3], [3 * node_c - 2], [3 * node_c - 1],
                         [3 * node_r - 3], [3 * node_r - 2], [3 * node_r - 1]]
            ele_dof_r = [[3 * node_c - 3, 3 * node_c - 2, 3 * node_c - 1,
                          3 * node_r - 3, 3 * node_r - 2, 3 * node_r - 1]]
            self.cc[ele_dof_c, ele_dof_r] = self.cc[ele_dof_c, ele_dof_r] + self.cc_ele[ele]

    def assemble_gm(self, num_dof):
        """
        Assemble ground motion data
        :param num_dof: Number of DOF
        """
        gm_idx = np.zeros((num_dof, self.ndf))
        accel_data = np.zeros((self.ndf, self.accel_len))
        for gm_dir in self.gm.keys():
            try:
                accel_data[int(gm_dir) - 1, :] = self.gm[gm_dir]
                for node in self.nodes.keys():
                    gm_idx[3 * int(node) - (self.ndf - int(gm_dir) + 1), int(gm_dir) - 1] = 1
            except:
                continue
        self.a_g = np.dot(gm_idx, accel_data)

    def assemble_f(self, num_dof):
        """
        Assemble element nodal force vector
        :param num_dof: Number of DOF
        """
        self.f = np.zeros((num_dof, 1))  # Global nodal force vector
        for node in self.nodes.keys():
            try:
                self.f[3 * int(node) - 3: 3 * int(node), :] = np.array(self.loads[node]).reshape(-1, 1)
            except:
                continue

    def assemble_fc(self):
        """
        Assemble nodal contact force vector
        """
        num_dof = self.get_num_dof()
        self.fc = np.zeros((num_dof, 1))
        for ele in self.c_ele.keys():
            node_c, node_r = self.c_ele[ele]['node'][0], self.c_ele[ele]['node'][1]
            self.fc[3 * int(node_c) - 3: 3 * int(node_c), :] = self.fc_ele[ele][0: self.ndf]
            self.fc[3 * int(node_r) - 3: 3 * int(node_r), :] = self.fc_ele[ele][self.ndf:]

    def rayleigh(self, alpha_m, beta_k):
        """
        Define rayleigh damping
        :param alpha_m: Factor applied to elements or nodes mass matrix
        :param beta_k: Factor applied to elements current stiffness matrix
        """
        self.alpha_m, self.beta_k = alpha_m, beta_k

    def analysis(self, tag):
        """
        Define analysis tag
        :param tag: Analysis tag
        """
        assert tag == 'static' or tag == 'transient', 'Analysis type must be static or transient'
        self.analysis_tag = tag

    def system(self):
        """
        Store the system of equations
        """
        num_dof = self.get_num_dof()
        self.active_dof_i, self.active_dof_j, self.active_dof = self.get_activate_dof()  # Active DOF
        self.assemble_k(num_dof)  # Assemble stiffness matrix
        if self.analysis_tag == 'transient':
            self.assemble_m(num_dof)
            self.assemble_gm(num_dof)
            self.u = np.zeros((num_dof, self.accel_len))
            self.u_t = np.zeros((num_dof, self.accel_len))
            self.u_tt = np.zeros((num_dof, self.accel_len))
        else:
            self.u = np.zeros((num_dof, 1))
        # Contact problem
        if len(self.c_ele.keys()) != 0:
            for ele in self.c_ele.keys():
                self.g[ele] = np.zeros(self.accel_len)
        self.assemble_f(num_dof)
