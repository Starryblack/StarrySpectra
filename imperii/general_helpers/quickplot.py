#!/usr/bin/env python3
import math
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.odr import *
from scipy import optimize

from natsort import natsorted
from sympy.solvers import solve
from sympy import Symbol
import time
import progressbar

from functions_dict import motherFunc
m = motherFunc()

class quickSilver():

    def __init__(self):
        print('\nPetite Quickplot Assistant Initialising..........WONDERFUL DAY.\n')
        print('\nPlease use q as quickSilver()\n')
        self.datadict = {}
        self.witherror = True
        print('>>>Error Mode is TRUE<<<')
        self.plot_params = {
        'ylabel': 'Pixel Value',
        'xlabel': 'Displacement (nm)',
        'witherror': True,
        'withfit': True,
        'flash': True,
        'save': True,
        'aspect': 1.0,
        'yaxis_dp': '%.0f',
        'xaxis_dp': '%.0f'
        }
        print('\nSir, if you wish to fit a certain function, please select the model with q.model_select("FUNCTION"), where the function names are:\n\n', list(m.function_book.keys()),'\n')
        self.macro4()
        # self.plothist('1.MOV (copy).txt')


    def errormode(self):
        if self.witherror == True:
            self.witherror = False
            self.plot_params['witherror'] = False
            self.plot_params['withfit'] = False
            print('No error mode chosen, assuming data has no error content...')

        elif self.witherror == False:
            self.witherror = True
            self.plot_params['witherror'] = True
            self.plot_params['withfit'] = True
            print('Error mode chosen, expecting data has both x and y errors...')


    def setdir(self, directory):
        self.data_directory = directory
        print('Current local directory set to {}'.format(directory),'\n')
        try:
            self.files = os.listdir(directory)
            self.files = natsorted(self.files)
            print('Sir, your data has just arrived! They are:\n', self.files)
        except:
            print('Please park your video data files into directory parent/{} where parent is the folder of this script.'.format(self.data_directory))
            sys.exit(1)

    def dataparse(self, error = True, lineskip = 0, delimiter = None):
        for i in self.files:
            try:
                sliced_data = []
                data = np.loadtxt('{}{}'.format(self.data_directory, i), skiprows = lineskip, delimiter = delimiter)
                self.x = data[:,0]
                self.y = data[:,1]
                if self.witherror:
                    self.xerr = data[:,2]
                    self.yerr = data[:,3]
                    data_set = [self.x, self.y, self.xerr, self.yerr]

                elif self.witherror == False:
                    data_set = [self.x, self.y]

                for q in data_set:
                    sliced_data.append(q)
                self.datadict[i] = sliced_data
                print(i, 'done!')
            except IsADirectoryError or TypeError:
                    print('\n |',i, '| is not .txt/.csv file (probably a folder?), skipping...\n')
                    pass

    def data_select(self, dataname):
        self.datacurrent = self.datadict[dataname]

    def model_select(self, model):
        self.modelcurrent = model

    def param_guess(self):
        dat = self.datacurrent
        if self.modelcurrent == 'linear':
            gradient = (dat[1][-1] - dat[1][0]) / (dat[0][-1] - dat[0][0])
            intercept = dat[1][0] - gradient * dat[0][0]
            self.initials = [gradient, intercept]

        elif self.modelcurrent == 'cos_square':
            offset = np.min(dat[1])
            amp = np.max(dat[1]) - offset
            period = 360
            #find sign of gradient (rough estimate) * degree value at y0 to account for "leading" or "trailing" phase diff
            phi = 0 - np.sign(dat[1][1]-dat[1][0])*(np.arccos(np.sqrt((dat[1][0]-offset)/amp)))
            self.initials = [amp, period, phi, offset]
            print('Your initial fit guesses are', self.initials)

    def param_guess_manual(self, initials = []):
        self.initials = initials

    def fit(self):
        dat = self.datacurrent
        function = Model(m.function_select(self.modelcurrent))

        # Create a RealData object using our initiated data from above. basically declaring all our variables using the RealData command which the scipy package wants
        input = RealData(dat[0], dat[1], sx = dat[2]**2, sy = dat[3]**2)
        # Set up ODR with the model and data. ODR is orthogonal distance regression (need to google!)
        odr = ODR(input, function, beta0 = self.initials)

        # Run the regression.
        self.out = odr.run()
        print('Fit successful! Parameters are:', self.out.beta, 'with their stdev being:', self.out.sd_beta, 'and residual variance being:', self.out.res_var)



    def plot(self, title):
        dat = self.datacurrent

        xlabel = self.plot_params['xlabel']
        ylabel = self.plot_params['ylabel']
        witherror = self.plot_params['witherror']
        withfit = self.plot_params['withfit']
        flash = self.plot_params['flash']
        save = self.plot_params['save']
        aspect = self.plot_params['aspect']
        yaxis_dp = self.plot_params['yaxis_dp']
        xaxis_dp = self.plot_params['xaxis_dp']


        fig, ax = plt.subplots()
        ax.tick_params(labelsize = 15, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)

        plt.xlabel('{}'.format(xlabel), fontsize = 14)
        plt.ylabel('{}'.format(ylabel), fontsize = 14)
        ax.yaxis.set_major_formatter(FormatStrFormatter(yaxis_dp))
        ax.xaxis.set_major_formatter(FormatStrFormatter(xaxis_dp))

        if witherror:
            plt.errorbar(dat[0], dat[1], xerr = dat[2], yerr = dat[3], fmt = 'o', mew = 0.3, ms = 0.2, capsize = 3, label = 'Data')
        else:
            plt.scatter(dat[0], dat[1], linewidth = 2.1, label = 'Data')
        plt.title(title)
        ax.set_aspect(abs((max(dat[0]) - min(dat[0]))/(max(dat[1]) - min(dat[1]))) * aspect)
        # plt.xlim((40,60))

        if withfit:
            x_fit = np.linspace(min(dat[0]), max(dat[0]), 1000)
            func = m.function_select(self.modelcurrent)
            y_fit = func(self.out.beta, x_fit)
            plt.plot(x_fit, y_fit, linewidth = 1., label = 'Fit Curve')

        plt.legend()
        if save:
            plt.savefig('plotscurrent/{}.eps'.format(title), format = 'eps', bbox_inches='tight')
            # plt.savefig('plots/{}.svg'.format(title), format = 'svg')
        if flash:
            plt.show(block = False)
            plt.pause(1)
            plt.close()
        else:
            plt.show()

    def plothist(self, file):
        data = np.loadtxt(file)
        # print(data)
        x = data
        print(np.std(x))
        print(np.mean(x))
        plt.hist(x, bins = 9)
        plt.show()

    def macro1(self):
        # self.setdir('data/')
        self.dataparse()
        # self.data_select('polariser_angle_vs_voltage_master.csv')
        # self.model_select('cos_square')
        self.param_guess()
        self.fit()

    def macro2(self):
        self.setdir('optics/single_slit/data/txt/')
        self.dataparse()
        self.data_select('hair1 (copy).tif.txt')
        self.model_select('single_slit_laser')
        self.param_guess_manual([3000, np.pi/0.000670/500*0.1, 8, np.min(self.y)])
        # self.param_guess_manual([np.max(self.y), np.pi/0.000670/500*0.06, np.mean(self.x)+0.01, np.min(self.y)])
        # self.param_guess_manual([200, np.pi/0.000670/500*0.06, 12.1, np.min(self.y)])
        print(self.initials)
        self.fit()
        print('Width is', 0.000670*500*self.out.beta[1]/np.pi, '+-', 0.000670*500*self.out.beta[1]/np.pi*np.sqrt((10/670)**2)+(1/500)**2)
        datt = self.datacurrent
        chisquare = 0
        func = m.function_select(self.modelcurrent)
        y_fit = func(self.out.beta, datt[0])
        for i in range(len(datt[1])):
            chisquare += (datt[1][i] - y_fit[i])**2 / (datt[2][i]**2 + datt[3][i]**2)**0.5
        chired = chisquare / (len(datt[1]) - 4)
        print('chired =', chired, 'with DOF =', len(datt[1]) - 4)
        self.plot('')

    def macro3(self):
        self.plot_params['withfit'] = False
        self.setdir('datacurrent/')
        self.dataparse()

        for i in self.files:
            self.data_select(i)
            self.plot(i)

    def macro4(self):
        self.setdir('datacurrent/')
        self.dataparse(lineskip = 0)
        for z in self.files:
            self.data_select(z)
            self.model_select('sin')
            self.param_guess_manual([np.max(self.y) - np.mean(self.y), 1200, np.pi/3, np.mean(self.y)])
            print(self.initials)
            self.fit()
            datt = self.datacurrent
            chisquare = 0
            func = m.function_select(self.modelcurrent)
            y_fit = func(self.out.beta, datt[0])
            for i in range(len(datt[1])):
                chisquare += (datt[1][i] - y_fit[i])**2 / (datt[2][i]**2 + datt[3][i]**2)**0.5
            chired = round(chisquare / (len(datt[1]) - 4), 2)
            temp = '_chired=' + str(chired) + '_DOF=' + str(len(datt[1]) - 4)
            temp = str(z) + temp
            self.plot(title = temp)

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        pass

if __name__ == '__main__':
    with quickSilver() as q:
        # import pdb; pdb.set_trace()
        import code; code.interact(local=locals())
