#!/usr/bin/env python3

# numpy is a package (to be installed) that enables commands to be used for linear algebra operations like matrices
import numpy as np
import math

#matplotlib is the python graph plotting package (to be installed)
import matplotlib.pyplot as plt

# scipy is the package that provides data analysis commands like fitting (to be installed). odr is specfic subpackage within scipy that we are calling. ODR is orthogonal distance regression (need to google!)
from scipy.odr import *
import sys
import os

# os.listdir allows me to automatically read and store the file names of the data files so i dont have to manually type out which file i want the machine to parse. all_files becomes a list of strings of filenames



class curveFitting():

    def __init__(self):
        print('Hello World!')
        self.datastore = os.listdir('data/iv_curves')
        self.datastore.sort()
        self.wavelengths = [469., 564., 610., 404., 633., 599.]
        self.gradients = []
        self.intercepts = []
        self.grad_error = []
        self.inter_error = []

    def dataparse(self, filename, domain = [0, 6]):
        # load txt file in specified directory for easy data import
        # loading our data files (with the appropriate directory) gives a 4 column array representing x,y, xerror and yerror
        data = np.loadtxt('data/iv_curves/{}'.format(filename))
        # selecting column index 0 of data array and set it to variable x
        # the colon actually needs two numbers on each side like 3:10 which means i choose row 3 to 9 (10-1) from the data, but leaving it as just : means i choose all rows of data
        self.x = data[domain[0]:domain[1],0]
        #print(x)
        self.y = data[domain[0]:domain[1],1]
        #print(y)
        self.xerror = data[domain[0]:domain[1],2]
        #print(xerror)
        self.yerror = data[domain[0]:domain[1],3]
        #print(yerror)

    # just defining a function to be used for fitting.
    def function(self, p, x, type = 'linear'):
        self.fittype = type
        if self.fittype == 'linear':
            m, c = p
            return m*x + c
        elif self.fittype == 'expo':
            a, c, g = p
            return a*(2.7182818284590452353602874713527**(1.6*(10**-19))*x - g)**c


    def fitrun(self, initial = [0.,1.]):

        self.initial = initial
        # Create a model for fitting. This step is simply a command under the scipy package such that machine knows that this object can be used for fitting
        model = Model(self.function)
        # Create a RealData object using our initiated data from above. basically declaring all our variables using the RealData command which the scipy package wants
        input = RealData(self.x, self.y, sx = self.xerror, sy = self.yerror)

        # Set up ODR with the model and data. ODR is orthogonal distance regression (need to google!)
        odr = ODR(input, model, beta0 = self.initial)

        # Run the regression.
        self.out = odr.run()

        if self.fittype == 'linear':
            self.gradients.append(self.out.beta[0])
            self.intercepts.append(self.out.beta[1])
            self.grad_error.append(self.out.sd_beta[0])
            self.inter_error.append(self.out.sd_beta[1])
            self.res_var = self.out.res_var
    def plot(self, xlabel = "Voltage/V", ylabel = 'Current/mA', freeze = False):

        fig, ax = plt.subplots()
        # plt.xlim(xrange)
        # plt.ylim(yrange)

        # xleft, xright = xrange
        # ybottom, ytop = yrange
        # ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*aspect)
        ax.tick_params(direction='in', length=6, width=1, bottom = True, top = True, left = True, right = True)
        # fig.suptitle('{}'.format(title), fontsize=15)
        plt.xlabel('{}'.format(xlabel), fontsize = 15)
        plt.ylabel('{}'.format(ylabel), fontsize = 15)

        # this is simply generating a sequence of x data points for the computer to generate the fit line, 1000 points between the two extremes of our x data set
        x_fit = np.linspace(self.x[0], self.x[-1], 1000)

        # this is the actual line, as you can see it receives the out.beta which is our fitted parameters plus the x domain generated in previous step.
        y_fit = self.function(self.out.beta, x_fit)

        # this is simply plotting everything with all information using matplotlib commands.
        plt.errorbar(self.x, self.y, xerr = self.xerror, yerr = self.yerror, linestyle='None', marker='x')
        plt.plot(x_fit, y_fit)

        if freeze == False:
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        else:
            plt.show()

    def misc_cal(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

fit = curveFitting()
print(fit.datastore)
a = 0
for i in fit.datastore:
    fit.dataparse(i)
    fit.fitrun()
    temp = -fit.intercepts[a] / fit.gradients[a]
    print(i,'Threshold voltage =', temp, 'Standard dev =', abs(temp) * math.sqrt((fit.grad_error[a] / fit.gradients[a])**2 + (fit.inter_error[a] / fit.intercepts[a])**2 ), 'Gradient =',fit.gradients[a], 'Y-intercept =',fit.intercepts[a], '\n')
    a += 1
    print('Residual Variance:', fit.res_var)
    fit.plot()


fit = curveFitting()
t = 'thresholdv_vs_wavenumber.txt'
fit.dataparse(t)
fit.fitrun()
print('Planck Constant issssssssssssss drrrrrrrrrrrrrrrrrrrrrrrr', fit.gradients[0]/299792458*(1.6*(10**-19))/10**9, 'with error offf', fit.grad_error[0]/299792458*(1.6*(10**-19))/10**9)
print("gradient error=",fit.grad_error[0])
print('gradient=',fit.gradients[0])
print('intercept=', fit.intercepts[0])
fit.plot(freeze=True)






# print('m = ', out.beta[0])
# print('c = ', out.beta[1])
# print('m standard error = ', out.sd_beta[0])
# print('c standard error = ', out.sd_beta[1],'\n')

#
# planck_constants = []
# for i in range(0,6):
#     planck_constants.append((-intercepts[i]/gradients[i] * wavelengths[i] * 10**-9 * (1.6*10**-19))/299792458)
#
# print('the estimated planck constants are!!!!!', planck_constants)
