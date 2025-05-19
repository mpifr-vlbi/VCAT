import numpy as np
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from vcat.helpers import closest_index, calculate_beam_width

#initialize logger
from vcat.config import logger

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

    def get_ridgeline_polar(self,r,theta,polar_image,image,beam,error,slice_width=100,counterjet=False,chi_sq_val=100,start_radius=0,maxfev=10000):

        start_i=closest_index(r[0],start_radius)
        # get slice
        slice_width = slice_width * image.degpp * image.scale
        for i in range(start_i,len(r[0])):
            theta_slice=theta[:,i]
            slice_to_fit=np.array(polar_image[:,i])
            length=len(slice_to_fit)
            diffs=[]

            for k in range(length):
                left_indices = [(k - j) % length for j in range(1, round(length/2))]
                right_indices = [(k + j) % length for j in range(1, round(length/2))]

                left_sum = np.sum(slice_to_fit[left_indices])
                right_sum = np.sum(slice_to_fit[right_indices])
                diffs.append(abs(left_sum - right_sum))

            ridge_index = np.argmin(diffs)
            theta0=theta_slice[ridge_index]
            if theta0 > 90 and not counterjet:
                theta0-=180
            if theta0<-90 and not counterjet:
                theta0+=180
            if theta0>-90 and theta0 <0 and counterjet:
                theta0+=180
            if theta0>=0 and theta0 <90 and counterjet:
                theta0-=180

            print(theta0)

            #fit gaussian for width
            # append X/Y positions
            x0=r[0][i] * np.sin(theta0/180*np.pi)
            y0=r[0][i] * np.cos(theta0/180*np.pi)
            self.X_ridg.append(r[0][i] * np.sin(theta0/180*np.pi))
            self.Y_ridg.append(r[0][i] * np.cos(theta0/180*np.pi))


            x_slice,y_slice=image.get_profile(point1=[x0+slice_width/2*np.sin(theta0+np.pi/2),y0+slice_width/2*np.cos(theta0+np.pi/2)],
                                              point2=[x0-slice_width/2*np.sin(theta0+np.pi/2),y0-slice_width/2*np.cos(theta0+np.pi/2)],show=False)

            def gaussian(x,amplitude,mean,stddev):
                return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

            beam_width=calculate_beam_width(theta0+90,beam[0],beam[1],beam[2])

            initial_guess=[np.max(y_slice),x_slice[round(len(x_slice)/2)],beam_width/2]
            try:
                popt, pcov = curve_fit(gaussian, x_slice, y_slice, p0=initial_guess,maxfev=maxfev)

                #extract fit parameters
                perr = np.sqrt(np.diag(pcov))
                amplitude=popt[0]
                amplitude_err=perr[0]
                width=popt[2]
                width_err=perr[2]
                FWHM=width*2*np.sqrt(2*np.log(2))
                err_FWHM=width_err*2*np.sqrt(2*np.log(2))
                slice_fitted = gaussian(x_slice, *popt)


                #calculate chi_square of fit
                chi_sq = 0.0
                for z in range(len(y_slice)):
                    chi_sq += (y_slice[z] - slice_fitted[z]) ** 2 / ((y_slice[z]*error) ** 2.0)
                chi_sq_red = float(chi_sq / (len(y_slice) - 4))

                logger.debug('The chi_square_red is = ' + str(chi_sq_red))
                if not (chi_sq_red < chi_sq_val):
                    if ((FWHM ** 2 - beam_width ** 2) > 0.0):
                        self.width.append(np.sqrt(FWHM ** 2.0 - beam_width ** 2.0))
                        logger.debug('The FWHM (de-convolved) is = ' + str(np.sqrt(FWHM ** 2.0 - beam_width ** 2.0)))
                        self.width_err.append(err_FWHM * np.sqrt(FWHM ** 2 - beam_width ** 2))

                        self.intensity.append(amplitude)
                        self.intensity_err.append(amplitude_err)
                        self.dist.append(r[0][i])
                        self.dist_int.append(r[0][i])
                        self.open_angle.append(2.0 * np.arctan(0.5 * np.sqrt(FWHM ** 2 - beam_width ** 2) / (
                                r[0][i])) * 180.0 / np.pi)
                        self.open_angle_err.append(err_FWHM * FWHM * 4 * r[0][i] * FWHM / (
                                np.sqrt(FWHM ** 2 - beam_width ** 2) * (
                                4.0 * r[0][i] ** 2 + FWHM ** 2 - beam_width ** 2)))
                    else:
                        self.dist_int.append(r[0][i])
                        self.intensity.append(amplitude)
                        self.intensity_err.append(amplitude_err)
            except:
                pass

        return self

    def get_ridgeline_polar_gauss(self,r,theta,polar_image,beam,error,chi_sq_val=100,start_radius=0,maxfev=10000):

        # TODO use actual beam width at angle instead of self.beam_maj
        beam = beam[0]

        #find roughly the jet direction:
        integrated_jet = np.zeros(len(theta[:, 0]))
        for i in range(len(r[0])):
            integrated_jet += polar_image[:, i] * r[:, i]  # correct for rdTheta in integration
        # find maximum flux
        max_ind = np.argmax(integrated_jet)
        jet_direction = theta[:, 0][max_ind]

        #define wrapped gaussian to use (closed on the circle
        def wrapped_gaussian(theta, A, mu, sigma,baseline):
            """
            A periodic Gaussian that is wrapped around the theta boundary [0, 2pi].
            """
            theta=theta/180*np.pi
            theta_wrapped = (theta + 180) % 360 - 180

            return A * np.exp(-0.5 * ((theta_wrapped - mu) / sigma) ** 2) +baseline

        start_i=closest_index(r[0],start_radius)

        for i in range(start_i,len(r[0])):
            theta_slice=theta[:,i]
            slice_to_fit=polar_image[:,i]

            #fit gaussian to slice
            #start gaussian
            a0=slice_to_fit[closest_index(theta_slice, jet_direction)]-np.min(slice_to_fit)
            try:
                popt, pcov = curve_fit(wrapped_gaussian,theta_slice,slice_to_fit,p0=[a0,jet_direction/180*np.pi,10/180*np.pi,np.min(slice_to_fit)],maxfev=maxfev)

                #extract fit parameters
                perr = np.sqrt(np.diag(pcov))
                amplitude=popt[0]
                amplitude_err=perr[0]
                theta0=popt[1]
                width=popt[2]*r[0][i] #correct for r*dphi
                width_err=perr[2]*r[0][i] #correct for r*dphi
                FWHM=width*2*np.sqrt(2*np.log(2))
                err_FWHM=width_err*2*np.sqrt(2*np.log(2))
                baseline=popt[3]
                slice_fitted = wrapped_gaussian(theta_slice, *popt)

                #append X/Y positions
                self.X_ridg.append(r[0][i]*np.sin(theta0))
                self.Y_ridg.append(r[0][i]*np.cos(theta0))

                #integrate gaussian
                gauss = lambda x: wrapped_gaussian(x,*popt)
                a = integrate.quad(gauss, -180, +180)

                #calculate chi_square of fit
                chi_sq = 0.0
                for z in range(len(slice_to_fit)):
                    chi_sq += (slice_to_fit[z] - slice_fitted[z]) ** 2 / ((slice_to_fit[z]*error) ** 2.0)
                chi_sq_red = float(chi_sq / (len(slice_to_fit) - 4))

                logger.debug('The chi_square_red is = ' + str(chi_sq_red))
                #TODO something is not 100% correctly working with the chi_square values here!!!
                if not (chi_sq_red < chi_sq_val):
                    if ((FWHM ** 2 - beam ** 2) > 0.0):
                        self.width.append(np.sqrt(FWHM ** 2 - beam ** 2))
                        logger.debug('The FWHM (de-convolved) is = ' + str(np.sqrt(FWHM ** 2.0 - beam ** 2.0)))
                        self.width_err.append(err_FWHM * np.sqrt(FWHM ** 2 - beam ** 2))

                        self.intensity.append(amplitude*r[0][i])
                        self.intensity_err.append(amplitude_err*r[0][i])
                        self.dist.append(r[0][i])
                        self.dist_int.append(r[0][i])
                        self.open_angle.append(2.0 * np.arctan(0.5 * np.sqrt(FWHM ** 2 - beam ** 2) / (
                                r[0][i])) * 180.0 / np.pi)
                        self.open_angle_err.append(err_FWHM * FWHM * 4 * r[0][i] * FWHM / (
                                np.sqrt(FWHM ** 2 - beam ** 2) * (
                                4.0 * r[0][i] ** 2 + FWHM ** 2 - beam ** 2)))
                    else:
                        self.dist_int.append(r[0][i])
                        self.intensity.append(amplitude*r[0][i])
                        self.intensity_err.append(amplitude_err*r[0][i])


                # plot every slice for diagnostics
                # slice_fitted = wrapped_gaussian(theta_slice,*popt)
                # plt.plot(theta_slice,wrapped_gaussian(theta_slice,a0,jet_direction/180*np.pi,10/180*np.pi,np.min(slice_to_fit)))
                # plt.plot(theta_slice,slice_to_fit)
                # plt.plot(theta_slice,slice_fitted)
                # plt.show()

            except:
                #fit did not work, do nothing
                pass

        return self

    def get_ridgeline_luca(self,image_data,noise,error,pixel_size,beam,X_ra,Y_dec,counterjet=False,angle_for_slices=0,cut_radial=5.0,
                           cut_final=10.0,width=40,j_len=100,chi_sq_val=100.0,err_FWHM=0.1,error_flux_slice=0.1):

        beam=calculate_beam_width(0,beam[0],beam[1],beam[2]) #we use angle=0 here because we are taking slices on the rotated image along the y axis.

        # reset attributes
        self.__init__()

        # Find the position of the maximum and create the box for the analysis
        max_index = np.unravel_index(np.nanargmax(image_data, axis=None), image_data.shape)

        position_y = max_index[0]
        position_x = max_index[1]
        position_x_beg = position_x - width
        position_x_fin = position_x + width

        # initializing some parameters depending on counterjet or not
        if counterjet:
            position_y_beg = position_y - 1  # minus 1 so the first slice considered is the first one of the counterjet (skip the core position)
            position_y_fin = position_y - j_len
            i = position_y_beg - position_y_fin
        else:
            position_y_beg = position_y + 1  # plus 1 so the first slice considered is the first one of the jet (skip the core position)
            position_y_fin = position_y + j_len
            i = position_y_fin - position_y_beg

        cont = 0
        a_old = 0
        b_old = 0

        for x in range(0, i):

            # initialize arrays
            if counterjet:
                position_y = position_y_beg - x
            else:
                position_y = position_y_beg + x

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
                logger.debug('Not this slice')
                cont += 1
                continue

            max_list = np.amax(data)
            size_x = len(data)

            if (max_list <= cut_final * noise):
                logger.debug('Not this slice')
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
            logger.debug(fitted_model)

            # Gaussian integral
            amplitude = fitted_model.parameters[0]
            mean = fitted_model.parameters[1]
            std = fitted_model.parameters[2]

            x1 = 1.0 * pixel_size
            x2 = size_x * pixel_size

            gauss = lambda x: amplitude * np.exp(-(x - mean) ** 2 / (std ** 2 * 2.0))
            a = integrate.quad(gauss, x1, x2)

            FWHM = 2.0 * np.sqrt(2.0 * np.log(2)) * std
            logger.debug("The FWHM (convolved) is = " + str(FWHM))
            chi_sq = 0.0
            for z in range(0, size_x):
                chi_sq += ((data[z] - amplitude * np.exp(-(X[z] - mean) ** 2 / (std ** 2 * 2))) ** 2 / (
                            data_err[z] ** 2.0))
            chi_sq_red = float(chi_sq / (size_x - 3))
            logger.debug('The chi_square_red is = ' + str(chi_sq_red))
            if (chi_sq_red < chi_sq_val):
                if ((FWHM ** 2 - beam ** 2) > 0.0):  # TODO check if this condition is actually the right thing to do
                    self.width.append(np.sqrt(FWHM ** 2 - beam ** 2))
                    logger.debug('The FWHM (de-convolved) is = ' + str(np.sqrt(FWHM ** 2.0 - beam ** 2.0)))
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
            else:
                cont += 1

        return self

    def jet_to_counterjet_profile(self,counter_ridgeline,savefig="",show=True):

        #initialize arrays
        sep = []
        err_beta = []
        ratio = []

        #get data from jet and counterjet
        intensityj = np.array(self.intensity)
        intensity = np.array(counter_ridgeline.intensity)
        intensityj_err = np.array(self.intensity_err)
        intensitycj_err = np.array(counter_ridgeline.intensity_err)
        distj_int = np.array(self.dist_int)
        distcj_int = np.array(counter_ridgeline.dist_int)

        size = min(len(distj_int), len(distcj_int))

        i = 0
        j = 0
        while i < size and j < size:
            if (distj_int[i] == distcj_int[j]):
                ratio.append(intensityj[i] / intensity[j])
                err_beta.append(np.sqrt((1.0 / intensity[i]) ** 2.0 * intensityj_err[i] ** 2.0 +
                                     (intensityj[i] / (intensity[i] ** 2.0)) ** 2.0 * intensitycj_err[i] ** 2.0))
                sep.append(distj_int[i])
                i += 1
                j += 1
            elif (distj_int[i] > distcj_int[j]):
                j += 1
            elif (distj_int[i] < distcj_int[j]):
                i += 1

        plt.errorbar(sep, ratio, yerr=err_beta, fmt='o', markersize=5.0)
        plt.ylabel('Ratio')
        plt.xlabel('Distance [mas]')
        if savefig!="":
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    def plot(self,mode="",savefig="",fit=True,start_fit=5,skip_fit=3,avg_fit=3,fig="",ax="",show=True):

        if fig=="" and ax=="":
            fig, ax = plt.subplots()

        if mode=="open_angle":
            ax.errorbar(self.dist, self.open_angle, yerr=self.open_angle_err, fmt='o', markersize=5.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Opening angle [deg]')
            ax.set_xlabel('Distance [mas]')
            ax.set_title('Opening angle')
            if savefig!="":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        elif mode=="intensity":
            ax.errorbar(self.dist_int, self.intensity, yerr=self.intensity_err, fmt='o', markersize=5.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Intensity [Jy/beam]')
            ax.set_xlabel('Distance [mas]')
            ax.set_title('Intensity Jet')
            if savefig!="":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        elif mode=="width":
            ax.errorbar(self.dist, self.width, yerr=self.width_err, fmt='o', markersize=5.0)

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
                logger.debug('Fit values (a*x**b) with a the first term and b the second -- First method')
                logger.debug(popt)
                logger.debug(perr)

                ax.errorbar(dist_fit, width_fit, yerr=width_err_fit, fmt='o', color='red', markersize=7.0)
                xpoint = np.linspace(self.dist[0], self.dist[len(self.dist) - 1], 1000)
                a = float(popt[0])
                b = float(popt[1])
                ax.text(xpoint[1], self.width[len(self.width) - 2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12,
                         bbox=dict(facecolor='red', alpha=0.5))
                ax.plot(xpoint, func(xpoint, *popt), color='red')

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
                logger.debug('Fit values (a*x**b) with a the first term and b the second -- Second method')
                logger.debug(popt)
                logger.debug(perr)

                logger.debug('Valori fit media')
                logger.debug(dist_fit)
                logger.debug(width_fit)
                ax.errorbar(dist_fit, width_fit, yerr=width_err_fit, fmt='o', color='purple', markersize=7.0)
                a = float(popt[0])
                b = float(popt[1])
                ax.text(xpoint[80], self.width[len(self.width) - 2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12,
                         bbox=dict(facecolor='purple', alpha=0.5))
                ax.plot(xpoint, func(xpoint, *popt), color='purple')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Jet width [mas]')
            ax.set_xlabel('Distance [mas]')
            ax.set_title('Collimation profile')
            if savefig!="":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        elif mode=="ridgeline":
            ax.plot(self.X_ridg, self.Y_ridg)
            ax.set_ylabel('Relative Dec. [mas]')
            ax.set_xlabel('Relative R.A. [mas]')
            ax.axis("equal")
            ax.invert_xaxis()
            ax.set_title('Ridgeline')
            if savefig != "":
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
            if show:
                plt.show()
        else:
            raise Exception("Please use valid mode ('open_angle','intensity','width','ridgeline'")
