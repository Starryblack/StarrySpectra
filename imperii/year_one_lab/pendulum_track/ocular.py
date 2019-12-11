#!/usr/bin/env python3
import math
import sys
import os

import cv2
import PIL as pillow
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
from scipy import optimize

from natsort import natsorted
from sympy.solvers import solve
from sympy import Symbol
import time
import progressbar



#OpenCV does not use RGB, it uses BGR (standing for blue, green, red), requires slicing and swapping the RGB info for each pixel around if want preview with Pillow
class ocularPendulum():

    def __init__(self):
        print('\nVirtual Ocular Machine Initialising..........done.\n')
        print('\nIn this universe, our homosapien optics have a relative luminosity L = 0.2126*RED + 0.7152*GREEN + 0.0722*BLUE in linear RGB space\n')
        self.redL = 0.2126
        self.greenL = 0.7152
        self.blueL = 0.0722
        self.bgrmatrix = [self.blueL, self.greenL, self.redL]
        self.bgrmatrix = np.array(self.bgrmatrix)
        self.framerate = 15

    def setdir(self, directory):
        self.data_directory = directory
        try:
            self.files = os.listdir(directory)
            self.files = natsorted(self.files)
            print('Sir, your data has just arrived! They are:\n', self.files)
        except:
            print('Please park your video data files into directory parent/{} where parent is the folder of this script.'.format(self.data_directory))
            sys.exit(1)

    def videoload(self, filename):
        self.video = cv2.VideoCapture('{}{}'.format(self.data_directory, filename))

    def frameparse(self, object):
        self.check, self.frame = self.video.read()

    def dataparse(self, file):
        data = np.loadtxt('{}{}'.format(self.data_directory, file))
        self.time = data[:,0]
        self.pos = data[:,1]
        self.time_err = data[:,2]
        self.pos_err = data[:,3]

    def framepreview(self, frame_array, mode = 'bgr', flash = True):
        if mode == 'bgr':
            frame_rgb = frame_array[:, :, [2, 1, 0]]
            image = pillow.Image.fromarray(frame_rgb)
            image.show()

        elif mode == 'rgb':
            image = pillow.Image.fromarray(frame_array)
            image.show()

        elif mode == 'gray':
            plt.imshow(frame_array, cmap = 'gray')
            if flash:
                plt.show(block = False)
                plt.pause(0.5)
                plt.close()

    def imagecrop(self, image_array, margins = []):
        lmargin, rmargin, tmargin, bmargin = margins
        self.frame = image_array[bmargin:-tmargin, lmargin:-rmargin]

    def makegray(self, image_array):
        self.gray_frame = np.zeros(shape=(len(image_array),len(image_array[0])))
        for i in range(len(image_array)):
            self.gray_frame[i] = np.rint(np.dot(image_array[i], self.bgrmatrix))

    def makebw(self, image_array, threshold):
        self.bw_frame = np.zeros(shape=(len(image_array),len(image_array[0])))
        for i in range(len(image_array)):
            for j in range(len(image_array[i])):
                if image_array[i][j] >= threshold:
                    self.bw_frame[i][j] = 255
                elif image_array[i][j] < threshold:
                    self.bw_frame[i][j] = 0

    def getpos(self, image_array):
        coordinates = np.where(image_array == 255)
        xarray = coordinates[1]
        yarray = coordinates[0]
        self.xbar = np.mean(xarray)
        self.ybar = np.mean(yarray)
        self.xsigma = np.std(xarray)
        self.ysigma = np.std(yarray)

    # for absolute convenience, i convert all framenumbers to seconds
    def buffer(self, frame_no):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def plot(self, x, y, xlabel, ylabel, title, yerr, xerr, withfit = False, type = 'light_damp', flash = False, save = False):

        fig, ax = plt.subplots()
        ax.tick_params(labelsize = 15, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)
        plt.xlabel('{}'.format(xlabel), fontsize = 20)
        plt.ylabel('{}'.format(ylabel), fontsize = 20)
        plt.errorbar(x, y, yerr = yerr, xerr = xerr, fmt = 'o', mew = 0.3, ms = 0.5, capsize = 3)
        plt.title(title)
        ax.set_aspect(abs((max(x) - min(x))/(max(y) - min(y))) * 1.0)

        if withfit:
            x_fit = np.linspace(min(x), max(x), 1000)
            if type == 'light_damp':
                func = self.fit_function(type)
                y_fit = func(self.out.beta, x_fit)
            plt.plot(x_fit, y_fit, linewidth = 0.5)

        if save:
            plt.savefig('plots/{}.eps'.format(title), format = 'eps')

        if flash:
            plt.show(block = False)
            plt.pause(1)
            plt.close()
        else:
            plt.show()

    def fit_function(self, type = 'light_damp'):
        if type == 'light_damp':
            def light_damp(p, x):
                amplitude, period, phi, offset, gamma = p
                return  amplitude * np.exp(- x * gamma / 2) * np.cos((2 * np.pi / period) * x + phi) + offset
            return light_damp

        elif type == 'basic_pendulum':
            def basic_pendulum(p, x):
                amplitude, period, phi, offset = p
                return amplitude * np.cos((2 * np.pi / period) * x + phi) + offset
            return basic_pendulum

    def fit(self, type, x, y, xerr, yerr, initials):

        model = Model(self.fit_function(type = 'light_damp'))
        # Create a RealData object using our initiated data from above. basically declaring all our variables using the RealData command which the scipy package wants
        input = RealData(x, y, sx = xerr, sy = yerr)
        # Set up ODR with the model and data. ODR is orthogonal distance regression (need to google!)
        odr = ODR(input, model, beta0 = initials)

        # Run the regression.
        self.out = odr.run()

    #macro1 = video extraction and processing into txt
    def macro1(self, file, buffer, dataframes, graythreshold = 220, margins = []):

        start_time = time.time()
        self.videoload(file)
        self.buffer(buffer)
        # self.framepreview(self.frame, mode = 'bgr')
        xpos = []
        ypos = []
        xerr = []
        yerr = []
        time_array = np.linspace(0, dataframes * (1 / self.framerate), dataframes)
        time_err = [1/(2 * self.framerate)] * dataframes
        for i in range(dataframes):
            self.frameparse(self.video)
            self.imagecrop(self.frame, margins)
            # self.framepreview(self.frame, mode = 'bgr')
            self.makegray(self.frame)
            # print(self.gray_frame)
            # self.framepreview(self.gray_frame, mode = 'gray')
            self.makebw(self.gray_frame, threshold = graythreshold)
            # self.framepreview(self.bw_frame, mode = 'gray', flash = True)
            self.getpos(self.bw_frame)
            # print('x =', self.xbar, 'y =', self.ybar, 'xstd =', self.xsigma, 'ystd =', self.ysigma)

            #since the x direction is dominant, we ignore y as it is the minor axis of oscillation, true independent variable should be time
            xpos.append(self.xbar)
            ypos.append(self.ybar)
            xerr.append(self.xsigma)
            yerr.append(self.ysigma)

        np.savetxt('txt_data/{}.txt'.format(file), np.c_[time_array, xpos, time_err, xerr])

        self.plot(x = time_array, y = xpos, xlabel = 'Time (s)', ylabel = 'X coordinate', title = '{}'.format(file), yerr = xerr, xerr = time_err, withfit = False, flash = True, save = True)

        print("\nVideo processing of %s done, it took about %.f seconds\n" % (file, (time.time() - start_time)))

    #macro2 = txt file processing for plots and fits
    def macro2(self, file):
        self.dataparse(file)

        self.centreline_guess = np.mean(self.pos)
        self.amplitude_guess = np.max(self.pos) - self.centreline_guess

        #obtains guess for period via counting time it takes to cut centreline (x intercepts)
        indices = np.nonzero((self.pos[1:] >= self.centreline_guess) & (self.pos[:-1] < self.centreline_guess))[0]
        self.period_guess = 1 / self.framerate * np.mean(np.diff([i - self.pos[i] / (self.pos[i+1] - self.pos[i]) for i in indices]))

        # initial fit guesses, gamma is usually around 0.1. the most crucial parameter is period
        initials = [self.amplitude_guess, self.period_guess, np.pi/4, self.centreline_guess, 0.1]
        # print(initials)
        print('\n',file)
        self.fit(x = self.time, y = self.pos, xerr = self.time_err, yerr = self.pos_err, initials = initials, type = 'light_damp')
        print('Fitted period is: {}'.format(self.out.beta[1]), 'with error of: %.3g' % self.out.sd_beta[1])

        self.plot(x = self.time, y = self.pos, type = 'light_damp', xlabel = 'Time (s)', ylabel = 'X coordinate', title = '{}'.format(file), yerr = self.pos_err, xerr = self.time_err, withfit = True, flash = True, save = True)


    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

################################################################
ocu = ocularPendulum()

# crop margins for every frame, left right top bottom order
margins = [300, 200, 300, 300]
ocu.setdir('data/transfer/')
# ocu.macro1('short_nylon_3.mp4', buffer = 225, dataframes = 300, margins = margins, graythreshold = 200)
# ocu.framepreview(ocu.bw_frame, mode = 'rgb')
# ocu.setdir('txt_data/')
# ocu.macro2('short_nylon_3.mp4.txt')
# #
#
for i in range(len(ocu.files)):
    ocu.macro1(ocu.files[i], buffer = 225, dataframes = 900, margins = margins, graythreshold = 210)
    # ocu.framepreview(ocu.frame, mode = 'bgr')


# ocu.setdir('txt_data/')
# for i in range(len(ocu.files)):
#     try:
#         ocu.macro2(ocu.files[i])
#     except:
#         print('\n',ocu.files[i], 'is not .txt file (probably a folder?), skipping...\n')
#         pass
#




# if __name__ == '__main__':
#     with ocularPendulum as o:
#         print("\n\nUse o as ocularPendulum()\n\n")
#         # import pdb; pdb.set_trace()
#         import code; code.interact(local=locals())
