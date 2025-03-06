import numpy as np
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Ridgeline(object):

    def __init__(self):

        #initialize attributes
        self.X_ridg=[]
        self.Y_ridg=[]
        self.open_angle=[]
        self.open_angle_err=[]
        self.width=[]
        self.width_err=[]
        self.dist=[]
        self.dist_int=[]
        self.intensity=[]
        self.intensity_err=[]

    def get_ridgeline_luca(self,image_data,noise,error,pixel_size,beam,X_ra,Y_dec,angle_for_slices=0,cut_radial=5.0,
                           cut_final=10.0,width=40,j_len=100,chi_sq_val=100.0,err_FWHM=0.1,error_flux_slice=0.1):
        # TODO use actual beam width at angle instead of self.beam_maj
        beam=beam[0]

        # reset attributes
        self.__init__()

        # Find the position of the maximum and create the box for the analysis
        max_index = np.unravel_index(np.nanargmax(image_data, axis=None), image_data.shape)
        max = image_data[max_index[0], max_index[1]]

        position_y = max_index[0]
        position_x = max_index[1]
        position_x_beg = position_x - width
        position_x_fin = position_x + width
        position_y_beg = position_y + 1  # plus 1 so the first slice considered is the first one of the jet (skip the core position)
        position_y_fin = position_y + j_len

        # initializing some parameters
        i = position_y_fin - position_y_beg
        cont = 0
        a_old = 0
        b_old = 0

        for x in range(0, i):

            # initialize arrays
            position_y = position_y_beg + x
            # print(position_y)
            size_x = position_x_fin - position_x_beg
            position_x = int(position_x_beg + size_x / 2)
            position = (position_x, position_y)
            size = (1, size_x)

            # create the array and the parameter (ind) for the slice determination
            cutout = Cutout2D(image_data, position, size)
            data_save = cutout.data

            x_line = []
            data = []

            for y in range(0, size_x):
                pix_val = data_save[0, y]
                data.append(pix_val)
                x_line.append(position_x_beg + y)

            ind = np.argmax(data)

            # conditions for a proper slice determination
            pos_a = position_x_beg + ind
            if (x == 0):
                old_pos_a = pos_a
                a = 0.0
                b = -a * angle_for_slices * np.pi / 180.0
            if (x >= 1):
                a = pos_a - old_pos_a
                b = -a * angle_for_slices * np.pi / 180.0
                if (a < a_old):
                    diff = a_old - a
                    b = b_old + diff * angle_for_slices * np.pi / 180.0
                if (a > a_old):
                    diff = a_old - a
                    b = b_old + diff * angle_for_slices * np.pi / 180.0
                a_old = a
                old_pos_a = pos_a
                b_old = b
            q = position_y - np.sin(b) * (position_x_beg + ind)
            y_line = [q + np.sin(b) * z for z in x_line]
            y_line = np.array(y_line)
            y_line_int = y_line.astype(int)

            # fill out the array for gaussian analysis, check whether the slice is okay and then prepare for the output map
            data = []
            data_err = []
            for y in range(0, size_x):
                indx = x_line[y]
                indy = y_line_int[y]
                image_data = np.array(image_data)
                pix_val = image_data[indy, indx]
                if (pix_val >= cut_radial * noise):
                    data.append(pix_val)
                    data_err.append(pix_val * error)

            if (len(data) <= 5):
                # print('Not this slice')
                cont += 1
                continue

            max_list = np.amax(data)
            size_x = len(data)

            if (max_list <= cut_final * noise):
                # print('Not this slice')
                cont += 1
                continue

            self.X_ridg.append(X_ra[pos_a])
            self.Y_ridg.append(Y_dec[position_y])

            # Single gaussian fit
            X = np.linspace(1.0 * pixel_size, size_x * pixel_size, size_x)
            model = models.Gaussian1D(max_list, size_x * pixel_size / 2.0,
                                      beam / 2 / np.sqrt(2 * np.log(2)))
            fitter = fitting.LevMarLSQFitter()
            fitted_model = fitter(model, X, data)
            # print(fitted_model)

            # Gaussian integral
            amplitude = fitted_model.parameters[0]
            mean = fitted_model.parameters[1]
            std = fitted_model.parameters[2]

            x1 = 1.0 * pixel_size
            x2 = size_x * pixel_size

            gauss = lambda x: amplitude * np.exp(-(x - mean) ** 2 / (std ** 2 * 2.0))
            a = integrate.quad(gauss, x1, x2)

            FWHM = 2.0 * np.sqrt(2.0 * np.log(2)) * std
            # print("The FWHM (convolved) is = " + str(FWHM))
            chi_sq = 0.0
            for z in range(0, size_x):
                chi_sq += ((data[z] - amplitude * np.exp(-(X[z] - mean) ** 2 / (std ** 2 * 2))) ** 2 / (
                            data_err[z] ** 2.0))
            chi_sq_red = float(chi_sq / (size_x - 3))
            # print('The chi_square_red is = ' + str(chi_sq_red))
            if (chi_sq_red < chi_sq_val):
                if ((FWHM ** 2 - beam ** 2) > 0.0):  # TODO check if this condition is actually the right thing to do
                    self.width.append(np.sqrt(FWHM ** 2 - beam ** 2))
                    #print('The FWHM (de-convolved) is = ' + str(np.sqrt(FWHM ** 2.0 - beam ** 2.0)))
                    self.width_err.append(err_FWHM * np.sqrt(FWHM ** 2 - beam ** 2))
                    self.intensity.append(a[0])
                    self.intensity_err.append(a[0]*error_flux_slice)
                    cont += 1
                    self.dist.append(cont * pixel_size)
                    self.dist_int.append(cont * pixel_size)
                    self.open_angle.append(2.0 * np.arctan(0.5 * np.sqrt(FWHM ** 2 - beam ** 2) / (
                                cont * pixel_size)) * 180.0 / np.pi)
                    self.open_angle_err.append(err_FWHM * FWHM * 4 * cont * pixel_size * FWHM / (
                                np.sqrt(FWHM ** 2 - beam ** 2) * (
                                    4.0 * cont ** 2 * pixel_size ** 2 + FWHM ** 2 - beam ** 2)))

                if ((FWHM ** 2 - beam ** 2) < 0.0):
                    cont += 1
                    self.dist_int.append(cont * pixel_size)
                    self.intensity.append(a[0])
                    self.intensity_err.append(a[0]*error_flux_slice)
            if (chi_sq_red > chi_sq_val):
                cont += 1

        return self

    def plot(self,mode="",savefig="",fit=True,start_fit=5,skip_fit=3,avg_fit=3,show=True):

        fig, ax = plt.subplots()

        if mode=="open_angle":
            plt.errorbar(self.dist, self.open_angle, yerr=self.open_angle_err, fmt='o', markersize=5.0)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel('Opening angle [deg]')
            plt.xlabel('Distance [mas]')
            plt.title('Opening angle')
            if savefig!="":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        elif mode=="intensity":
            plt.errorbar(self.dist_int, self.intensity, yerr=self.intensity_err, fmt='o', markersize=5.0)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel('Intensity [Jy/beam]')
            plt.xlabel('Distance [mas]')
            plt.title('Intensity Jet')
            if savefig!="":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        elif mode=="width":
            plt.errorbar(self.dist, self.width, yerr=self.width_err, fmt='o', markersize=5.0)

            if fit==True:
                # -- Fitting function --
                def func(x, a, b):
                    return a * x ** b

                # -- Fitting arrays for: skip the first start_fit points, plus take one point every skip_fit --

                dist_fit = self.dist[start_fit::skip_fit]
                width_fit = self.width[start_fit::skip_fit]
                width_err_fit = self.width_err[start_fit::skip_fit]

                popt, pcov = curve_fit(func, dist_fit, width_fit, sigma=width_err_fit)
                perr = np.sqrt(np.diag(pcov))
                #print('Fit values (a*x**b) with a the first term and b the second -- First method')
                #print(popt)
                #print(perr)

                plt.errorbar(dist_fit, width_fit, yerr=width_err_fit, fmt='o', color='red', markersize=7.0)
                xpoint = np.linspace(self.dist[0], self.dist[len(self.dist) - 1], 1000)
                a = float(popt[0])
                b = float(popt[1])
                plt.text(xpoint[1], self.width[len(self.width) - 2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12,
                         bbox=dict(facecolor='red', alpha=0.5))
                plt.plot(xpoint, func(xpoint, *popt), color='red')

                # -- Fitting arrays for: take an average value every avg_fit points --

                dist_fit = []
                width_fit = []
                width_err_fit = []

                counter = 0
                valuer = 0.0
                valued = 0.0
                valuee = 0.0

                for i in range(0, len(self.dist)):
                    counter = counter + 1
                    if (counter <= avg_fit):
                        valuer = valuer + self.dist[i]
                        valued = valued + self.width[i]
                        valuee = valuee + self.width_err[i]

                    if (counter == avg_fit + 1):
                        valuer = valuer / float(avg_fit)
                        valued = valued / float(avg_fit)
                        valuee = valuee / float(avg_fit)

                        # Fill out the array for the fitting
                        dist_fit.append(valuer)
                        width_fit.append(valued)
                        width_err_fit.append(valuee)

                        # Reset values
                        counter = 1
                        valuer = 0.0
                        valued = 0.0
                        valuee = 0.0

                        valuer = valuer + self.dist[i]
                        valued = valued + self.width[i]
                        valuee = valuee + self.width_err[i]

                popt, pcov = curve_fit(func, dist_fit, width_fit, sigma=width_err_fit)
                perr = np.sqrt(np.diag(pcov))
                #print('Fit values (a*x**b) with a the first term and b the second -- Second method')
                #print(popt)
                #print(perr)

                #print('Valori fit media')
                #print(dist_fit)
                #print(width_fit)
                plt.errorbar(dist_fit, width_fit, yerr=width_err_fit, fmt='o', color='purple', markersize=7.0)
                a = float(popt[0])
                b = float(popt[1])
                plt.text(xpoint[80], self.width[len(self.width) - 2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12,
                         bbox=dict(facecolor='purple', alpha=0.5))
                plt.plot(xpoint, func(xpoint, *popt), color='purple')

            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel('Jet width [mas]')
            plt.xlabel('Distance [mas]')
            plt.title('Collimation profile')
            if savefig!="":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        elif mode=="ridgeline":
            plt.plot(self.X_ridg, self.Y_ridg)
            plt.ylabel('Relative Dec. [mas]')
            plt.xlabel('Relative R.A. [mas]')
            plt.axis("equal")
            plt.gca().invert_xaxis()
            plt.title('Ridgeline')
            if savefig != "":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()
        else:
            raise Exception("Please use valid mode ('open_angle','intensity','width','ridgeline'")
