#!/usr/bin/env python3
#default dependencies
import math
import sys
import os

#non-default dependencies, please use pip3 to install (linux/mac) or conda (windows).
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
from scipy import optimize
from natsort import natsorted
from sympy.solvers import solve
from sympy import Symbol
import time
import progressbar

class hubbleConstant():

    def __init__(self):
        print('\nHubble Constant Telescope Array Data Processor Initialising.....done.\n')
        print('\nIn this universe, our Hydrogen-alpha spectral line is 656.28nm, and our vacuum speed of light is 299792458m...\n')
        self.halpha_nm = 656.28
        self.c = 299792458
        self.nm = 10**-9

        self.ob_no_dist = []
        self.lamb0 = []
        self.velocity = []
        self.dist_vel = []

        self.data_directory1 = 'data/halpha_spectral_data/'
        self.data_directory2 = 'data/distance_data/'
        try:
            self.spectra = os.listdir(self.data_directory1)
            self.spectra = natsorted(self.spectra)
            self.distancename = os.listdir(self.data_directory2)
            print('Sir, your data has just arrived! They are:', self.spectra)
        except:
            print('Please park your txt/csv spectra data files into directory parent/{}, and distance data into parent/{} where parent is the folder of this script.'.format(self.data_directory1, data_directory2))
            sys.exit(1)

    def goodbadspectra(self):
        self.all_n = []
        self.good_n = []
        self.good_index = []
        self.bad_n = []
        self.bad_index = []
        for i in self.spectra:
            metadata = []
            with open('{}{}'.format(self.data_directory1, i), 'r') as file:
                line1 = file.readline()
                line1_split = line1.split(',')

                for r in line1_split:
                     metadata.append(r.split(': '))
                self.all_n.append(int(metadata[2][1]))

                if metadata[3][1] == 'Good \n':
                    self.good_n.append(int(metadata[2][1]))
                    self.good_index.append(self.spectra.index(i))
                else:
                    self.bad_n.append(int(metadata[2][1]))
                    self.bad_index.append(self.spectra.index(i))

        print('\nSir, these measurements are useful:', self.good_n, 'I have kept a copy of the data indexes as well, accessible as hub.good_index.\n')
        print('On the other hand, these measurements are dismal:', self.bad_n, 'I have kept a copy of the data indexes as well, accesible as hub.bad_index.\n')

    def dataparse(self, xname, yname, directory, filename, skiprows, xerrname = 'xerr', yerrname = 'yerr', delimiter = ',', domain = [0,-1]):
        data = np.loadtxt('{}{}'.format(directory, filename), delimiter = delimiter, skiprows = skiprows)
        setattr(self, xname, data[domain[0]:domain[1],0])
        setattr(self, yname, data[domain[0]:domain[1],1])

        print('{} parsed and stored.'.format(filename))

    def linegaussfinder(self, xdata, ydata):
        self.rgradient = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        self.rintercept = ydata[0] - self.rgradient * xdata[0]
        self.residuals = []
        for x in range(len(xdata)):
            ry = self.rgradient * xdata[x] + self.rintercept
            self.residuals.append(ydata[x] - ry)
        self.max_res = max(self.residuals)
        # print(self.max_res)
        self.peak_pos = xdata[self.residuals.index(self.max_res)]
        # print(self.peak_pos)
        self.rsigma = np.std(self.residuals)
        # print(self.rsigma)

    def fit_function(self, type):
        if type == 'line':
            def line_func(x, m, c):
                return m * x + c
            return line_func
        elif type == 'linegauss':
            def linegauss_func(x, m, c, a, sigma, mean):
                return m * x + c + a * np.exp(-(x - mean)**2 / (2 * sigma**2))
            return linegauss_func

    def fit(self, type, x, y, initials, err = []):
        if len(err) == 0:
            self.p, self.pcov = optimize.curve_fit(self.fit_function(type = type), x, y, p0 = initials)
            print(initials)
        elif len(err) > 0:
            self.p, self.pcov = optimize.curve_fit(self.fit_function(type = type), x, y, p0 = initials, sigma = err)

    def fitplot(self, type, x, y, xlabel, ylabel, err = [], show = True, errorbar = False, freeze = False):
        if show:
            x_fit = np.linspace(min(x), max(x) + 20, 1000)
            func = self.fit_function(type = type)
            if type == 'linegauss':
                m, c, a, sigma, mean = self.p
                y_fit = func(x_fit, m, c, a, sigma, mean)

            elif type == 'line':
                m, c = self.p
                y_fit = func(x_fit, m, c)
            fig, ax = plt.subplots()
            # plt.xlim(xrange)
            # plt.ylim(yrange)

            # xleft, xright = xrange
            # ybottom, ytop = yrange
            ax.set_aspect(abs(((max(x) + 20) - min(x))/(max(y) - min(y))) * 1.0)
            ax.tick_params(labelsize = 15, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)
            # fig.suptitle('{}'.format(title), fontsize=15)
            plt.xlabel('{}'.format(xlabel), fontsize = 20)
            plt.ylabel('{}'.format(ylabel), fontsize = 20)

            if errorbar:
                plt.errorbar(x, y, yerr = err, fmt='o', mew=2, ms=0.5, capsize = 1.5)
                plt.plot(x_fit, y_fit)
            else:
                plt.plot(x, y)
                plt.plot(x_fit, y_fit)

            if freeze == False:
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()
            else:
                plt.show()

    #uses sympy methods Symbol and Solve
    def algebra_solve(self, param, type):
        if type == 'redshift':
            x = Symbol('x')
            a = solve(param / self.halpha_nm - ((1. + x / self.c) / (1. - x / self.c))**0.5, x)
            return a[0]
        if type == 'explicit':
            return (1 - 2 * self.halpha_nm**2 / (param**2 + self.halpha_nm**2)) * self.c


    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

################################################################
hub = hubbleConstant()
hub.goodbadspectra()

for a in hub.distancename:
    hub.dataparse(xname = 'ob_no', yname = 'distance', directory = hub.data_directory2, filename = a, skiprows = 1)
    for b in range(len(hub.ob_no)):
        temp = []
        temp.append(hub.ob_no[b])
        temp.append(hub.distance[b])
        hub.ob_no_dist.append(temp)
        hub.ob_no_dist = sorted(hub.ob_no_dist)

for a in hub.good_index:
    temp = []
    hub.dataparse(xname = 'wavelength', yname = 'intensity', directory = hub.data_directory1, filename = hub.spectra[a], skiprows = 2)
    hub.linegaussfinder(xdata = hub.wavelength, ydata = hub.intensity)
    hub.fit(type = 'linegauss', x = hub.wavelength, y = hub.intensity, initials = [hub.rgradient, hub.rintercept, hub.max_res, hub.rsigma, hub.peak_pos])

    temp.append(hub.all_n[a])
    temp.append(hub.p[4])
    temp.append((hub.pcov[4][4])**0.5)
    hub.lamb0.append(temp)

    hub.fitplot(type = 'linegauss', x = hub.wavelength, y = hub.intensity, xlabel = 'Wavelength (nm)', ylabel = 'Intensity (A.U.)', show = False)

print('Master, after curve fitting, the observation numbers, mean wavelength, and their corresponding error (sqrt(fit variance)) have been recorded as hub.lamb0\n')

print('Algebraic Solver Engine Intialised:')
for i in progressbar.progressbar(range(len(hub.lamb0))):
    temp = []
    temp.append(hub.lamb0[i][0])
    temp.append(hub.algebra_solve(param = hub.lamb0[i][1], type = 'explicit'))
    error = 4 * hub.lamb0[i][1] * (hub.halpha_nm**2) * (hub.nm**3) * ((hub.lamb0[i][1] * hub.nm)**2 + (hub.halpha_nm * hub.nm)**2)**-2 * hub.c * (hub.lamb0[i][2] * hub.nm)
    temp.append(error)
    hub.velocity.append(temp)
print('The collated observation numbers, velocity, and velocity errors are HERE!!!!! I have arranged them by ascending observation numbers, accessible as hub.velocity:\n')

hub.velocity = sorted(hub.velocity)

# stupid way to get a master list of lists containing individual ob_no, x, y, yerr datasets, converted to km for velocity values as well
temp0 = []
temp1 = []
temp2 = []
temp3 = []
tempset = [temp0, temp1, temp2, temp3]
for i in range(len(hub.velocity)):
    for z in range(len(hub.ob_no_dist)):
        if hub.velocity[i][0] == hub.ob_no_dist[z][0]:
            temp0.append(hub.ob_no_dist[z][0])
            temp1.append(hub.ob_no_dist[z][1])
            temp2.append(hub.velocity[i][1] / 1000)
            temp3.append(hub.velocity[i][2] / 1000)

for a in tempset:
    hub.dist_vel.append(a)

# print(len(hub.dist_vel[0]), len(hub.dist_vel[1]), len(hub.dist_vel[2]), len(hub.dist_vel[3]))
hub.fit(type = 'line', x = hub.dist_vel[1], y = hub.dist_vel[2], err = hub.dist_vel[3], initials = [0,0])
print('Now generating plot of Distance vs Redshift')
print('The Hubble Constant at our point of time is:', '%.2f' % hub.p[0],'\u00B1','%.2f' % hub.pcov[0][0]**0.5, 'km/s/Mpc,', 'reported as', '%.f' % hub.p[0],'\u00B1','%.f' % hub.pcov[0][0]**0.5,'km/s/Mpc.')

hub.fitplot(type = 'line', x = hub.dist_vel[1], y = hub.dist_vel[2], xlabel = 'Distance (Mpc)', ylabel = 'Redshift Velocity (km/s)', err = hub.dist_vel[3], show = True, errorbar = True, freeze = True)







#

#
#









    ######################
