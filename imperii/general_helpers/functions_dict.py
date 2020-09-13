#!/usr/bin/env python3

# Made 2020, Mingsong Wu
# mingsongwu [at] outlook [dot] sg
# github.com/StarryBlack/StarrySpectra


import math
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
from scipy import optimize

from natsort import natsorted
from sympy.solvers import solve
from sympy import Symbol
import time
import progressbar

class motherFunc():

    def __init__(self):
        print('\nmother-of-all-function-selector initialising..........WONDERFUL DAY.')
        self.function_book = {

        'linear': self.linear_model,
        'sin': self.sin,
        'cos_square': self.cos_square,
        'shm_lightdamp': self.light_damp,
        'shm_basic': self.basic_pendulum,
        'shm_twocoupled': self.two_couple,
        'single_slit_laser': self.singleslitlaser,
        'capacitor_decay': self.capacitor_decay
        }

    def function_select(self, type):
        return self.function_book[type]

    def linear_model(self, p, x):
        m, c = p
        return m * x + c

    def sin(self, p, x):
        amp, period, phi, offset = p
        return amp * (np.sin(x*(2*np.pi)/period + phi)) + offset

    def cos_square(self, p, x):
        amp, period, phi, offset = p
        return amp * (np.cos(x*(2*np.pi)/period + phi))**2 + offset

    def light_damp(self, p, x):
        amplitude, period, phi, offset, gamma = p
        return  amplitude * np.exp(- x * gamma / 2) * np.cos((2 * np.pi / period) * x + phi) + offset

    def basic_pendulum(self, p, x):
        amplitude, period, phi, offset = p
        return amplitude * np.cos((2 * np.pi / period) * x + phi) + offset

    def two_couple(self, p, x):
        amplitude, period1, y_offset, period2, x_offset, gamma = p
        return np.exp(- x * gamma / 2) * amplitude * np.cos(((2 * np.pi) / period2) * (x + x_offset)) * np.cos((2 * np.pi / period1) * (x + x_offset)) + y_offset

    def singleslitlaser(self, p, x):
        i0, scale, centre, offset = p
        return i0 * (np.sin(scale * (x - centre)) / (scale * (x - centre)))**2 + offset

    def capacitor_decay(self, p, x):
        yscale, xscale, xoffset, yoffset = p
        return np.exp(-(x - xoffset) / xscale) + yoffset


    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        print('function-selector self-destructing...done.')
