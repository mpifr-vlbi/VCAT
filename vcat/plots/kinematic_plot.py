from os import write
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from astropy.io import fits
from astropy.modeling import models, fitting
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time
import sys
import pexpect
from datetime import datetime
import colormaps as cmaps
import matplotlib.ticker as ticker
from vcat.helpers import get_sigma_levs, getComponentInfo, convert_image_to_polar, wrap_evpas, closest_index, get_date, get_freq, write_mod_file
import vcat.fit_functions as ff
from vcat.kinematics import Component
from vcat.config import logger, font
from scipy.interpolate import interp1d

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = font
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=13
font_size_axis_tick=12

class KinematicPlot(object):
    def __init__(self, pol_plot=False):

        super().__init__()

        self.pol_plot = pol_plot

        if pol_plot:
            self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # Set 0° to top
            self.ax.set_theta_zero_location("N")
            self.ax.set_theta_direction(1)

            # create ticks
            tick_angles_deg = np.arange(0, 360, 30)
            tick_labels = []
            for ang in tick_angles_deg:
                if ang < 180:
                    tick_labels.append(f"{ang // 2}°")
                elif ang == 180:
                    tick_labels.append("+90°/-90°")
                else:
                    tick_labels.append(f"{(ang - 360) // 2}°")

            self.ax.set_xticks(np.deg2rad(tick_angles_deg))
            self.ax.set_xticklabels(tick_labels)
        else:
            self.fig, self.ax = plt.subplots(1, 1)

        self.fig.subplots_adjust(left=0.13, top=0.96, right=0.93, bottom=0.2)

    def plot_kinematics(self, component_collection, color,label="", marker=".",plot_errors=False,snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        dists = []
        dist_err = []

        if component_collection.length() > 0:
            years=component_collection.year.flatten()
            dists=component_collection.dist.flatten()
            dist_err=component_collection.dist_err.flatten()
            evpas=component_collection.evpas.flatten()

            snrs=component_collection.snrs.flatten()

            years=years[snrs>=snr_cut]
            dists=dists[snrs>=snr_cut]
            dist_err=dist_err[snrs>=snr_cut]

            if plot_errors:
                self.ax.errorbar(years,dists,yerr=dist_err,c=color,fmt=marker,label=label)
            else:
                self.ax.scatter(years, dists, c=color, marker=marker,label=label)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = dists[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Distance from Core [mas]', fontsize=font_size_axis_title)

        return years,dists,dist_err

    def plot_fluxs(self, component_collection, color, label="",marker=".",plot_errors=False,snr_cut=1,plot_evpa=False,evpa_len=200):
        years=[]
        fluxs=[]
        fluxs_err=[]

        if component_collection.length() > 0:

            years = component_collection.year.flatten()
            fluxs = component_collection.fluxs.flatten()
            fluxs_err = component_collection.fluxs_err.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            fluxs = fluxs[snrs >= snr_cut]
            fluxs_err = fluxs_err[snrs >= snr_cut]

            if plot_errors:
                self.ax.errorbar(years,fluxs,yerr=fluxs_err,c=color,fmt=marker,label=label)
            else:
                self.ax.plot(years,fluxs, c=color,
                            label=label, marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = fluxs[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Flux Density [Jy]', fontsize=font_size_axis_title)

        return years,fluxs,fluxs_err

    def plot_pas(self, component_collection, color, label="",marker=".",snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        pas = []

        if component_collection.length() > 0:
            years = component_collection.year.flatten()
            pas = component_collection.posas.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            pas = pas[snrs >= snr_cut]

            self.ax.plot(years,pas, c=color,
                         label=label, marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = pas[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Position Angle [deg]', fontsize=font_size_axis_title)

        return years, pas

    def plot_theta(self, component_collection, color, label="", marker=".",plot_errors=False,snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        thetas = []
        thetas_err = []
        if component_collection.length() > 0:
            years = component_collection.year.flatten()
            thetas = component_collection.thetas.flatten()
            thetas_err = component_collection.thetas_err.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            thetas = thetas[snrs >= snr_cut]
            thetas_err = thetas_err[snrs >= snr_cut]

            if plot_errors:
                self.ax.errorbar(years,thetas,yerr=thetas_err,c=color,fmt=marker,label=label)
            else:
                self.ax.plot(years,thetas, c=color,
                             label=label,
                             marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = thetas[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Position Angle of the Component Position [°]', fontsize=font_size_axis_title)

        return years, thetas, thetas_err

    def plot_linpol(self, component_collection, color,label="", marker=".",snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        lin_pols = []
        if component_collection.length() > 0:
            years = component_collection.year.flatten()
            lin_pols = component_collection.lin_pols.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            lin_pols = lin_pols[snrs >= snr_cut]

            self.ax.plot(years,lin_pols, c=color,
                         label=label,
                         marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = lin_pols[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Linearly Polarized Flux Density [Jy]', fontsize=font_size_axis_title)

        return years, lin_pols

    def plot_fracpol(self, component_collection, color, marker=".",label="",snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        frac_pols = []
        if component_collection.length() > 0:
            years = component_collection.year.flatten()
            lin_pols = component_collection.lin_pols.flatten()
            fluxs = component_collection.fluxs.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            lin_pols = lin_pols[snrs>=snr_cut]
            fluxs = fluxs[snrs >= snr_cut]
            frac_pols = np.array(lin_pols) / np.array(fluxs) * 100

            self.ax.plot(years,
                         frac_pols,
                         c=color, label=label, marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = frac_pols[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Fractional Polarized Flux Density [%]', fontsize=font_size_axis_title)

        return years, frac_pols

    def plot_evpa(self, component_collection, color, label="",marker=".",snr_cut=1):
        evpas = component_collection.evpas.flatten()
        years = component_collection.year.flatten()

        snrs = component_collection.snrs.flatten()

        years = years[snrs >= snr_cut]
        evpas = evpas[snrs >= snr_cut]

        if self.pol_plot:

            plot_evpas = 2 * np.array(wrap_evpas(evpas)) / 180 * np.pi  # we will plot two times EVPA

            # interpolate EVPA for the line plot
            evpa_interp = interp1d(years, plot_evpas, kind="linear")
            years_interp = np.linspace(min(years), max(years), 10000)
            self.ax.plot(evpa_interp(years_interp), years_interp, color=color)

            # scatter plot the actual values
            self.ax.scatter(plot_evpas, years, color=color, label=label,marker=marker)

        else:
            if component_collection.length() > 0:
                self.ax.plot(years,evpas, c=color,
                             label=label,
                             marker=marker)
            self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
            self.ax.set_ylabel('EVPA [deg]', fontsize=font_size_axis_title)

        return years,evpas

    def plot_maj(self, component_collection, color, marker=".",label="",plot_errors=False,snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        majs = []
        majs_err = []
        if component_collection.length() > 0:
            years = component_collection.year.flatten()
            majs = component_collection.majs.flatten()
            majs_err = component_collection.majs_err.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            majs = majs[snrs >= snr_cut]
            majs_err = majs_err[snrs >= snr_cut]

            if plot_errors:
                self.ax.errorbar(years,majs,yerr=majs_err,c=color,fmt=marker,label=label)
            else:
                self.ax.plot(year,majs, c=color,label=label,
                            marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = majs[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Major Axis [deg]', fontsize=font_size_axis_title)

        return years, majs, majs_err

    def plot_min(self, component_collection, color, label="",marker=".",plot_errors=False,snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        mins = []
        mins_err = []
        if component_collection.length() > 0:

            years = component_collection.year.flatten()
            mins = component_collection.mins.flatten()
            mins_err = component_collection.mins_err.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            mins = mins[snrs >= snr_cut]
            mins_err = mins_err[snrs >= snr_cut]

            if plot_errors:
                self.ax.errorbar(years,mins,yerr=mins_err,c=color,fmt=marker,label=label)
            else:
                self.ax.plot(component_collection.year, component_collection.mins, c=color,label=label,
                             marker=marker)

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = mins[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('EVPA [deg]', fontsize=font_size_axis_title)

        return years,mins,mins_err

    def plot_tbs(self, component_collection, color, label="",marker=".",snr_cut=1,plot_evpa=False,evpa_len=200):
        years = []
        tbs = []
        if component_collection.length() > 0:
            years = component_collection.year.flatten()
            tbs_lower_limit = component_collection.tbs_lower_limit.flatten()
            tbs = component_collection.tbs.flatten()
            evpas = component_collection.evpas.flatten()

            snrs = component_collection.snrs.flatten()

            years = years[snrs >= snr_cut]
            tbs_lower_limit = tbs_lower_limit[snrs >= snr_cut]
            tbs = tbs[snrs >= snr_cut]

            lower_limit_inds = np.where(np.array(tbs_lower_limit))[0]
            tb_value_inds = np.where(np.array(tbs_lower_limit) == False)[0]
            self.ax.plot(np.array(years)[tb_value_inds],
                         np.array(tbs)[tb_value_inds], c=color,label=label,
                         marker=marker)
            self.ax.scatter(np.array(years)[lower_limit_inds],
                            np.array(tbs)[lower_limit_inds], c=color, marker="^")

            if plot_evpa:
                for i in range(len(years)):
                    year = years[i]
                    val = tbs[i]
                    evpa = evpas[i]

                    # make a markerstyle class instance and modify its transform prop
                    t = markers.MarkerStyle(marker="|")
                    t._transform = t.get_transform().rotate_deg(evpa)
                    self.ax.scatter(year, val, marker=t, s=evpa_len, c=color)

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Brightness Temperature [K]', fontsize=font_size_axis_title)
        self.ax.set_yscale("log")

        return years, tbs

    def plot_spectrum(self, component_collection, color, epochs="", marker="."):

        if epochs == "":
            epochs = component_collection.epochs_distinct
        elif isinstance(epochs, (float, int)):
            epochs = [epochs]
        elif not isinstance(epochs, list):
            raise Exception("Invalid input for 'epochs'.")

        for epoch in epochs:
            epoch_ind = closest_index(component_collection.epochs_distinct, epoch)

            freqs = component_collection.freqs[epoch_ind, :].flatten()
            fluxs = component_collection.fluxs[epoch_ind, :].flatten()

            if len(fluxs) > 0:
                self.ax.scatter(np.array(freqs) * 1e-9, fluxs,
                                c=color, label=component_collection.name, marker=marker)

        self.ax.set_xlabel("Frequency [GHz]", fontsize=font_size_axis_title)
        self.ax.set_ylabel("Flux Density [Jy]", fontsize=font_size_axis_title)

    def plot_chi_square(self, uvf_files, modelfit_files, difmap_path):

        # calculate chi-squares
        chi_squares = []
        dates = []

        for ind, uvf in enumerate(uvf_files):
            df = getComponentInfo(modelfit_files[ind])
            freq = get_freq(modelfit_files[ind])
            write_mod_file(df, "modelfit.mod", freq, adv=True)
            chi_square = get_model_chi_square_red(uvf, "modelfit.mod", difmap_path)
            chi_squares.append(chi_square)
            dates.append(get_date(modelfit_files[ind]))
            os.system("rm -rf modelfit.mod")

        chi_squares = np.array(chi_squares)

        for ind, dat in enumerate(dates):
            # calculate decimal year
            date = datetime.strptime(dat, "%Y-%m-%d")

            # Calculate the start of the year and the start of the next year
            start_of_year = datetime(date.year, 1, 1)
            start_of_next_year = datetime(date.year + 1, 1, 1)

            # Calculate the number of days since the start of the year and total days in the year
            days_since_start_of_year = (date - start_of_year).days
            total_days_in_year = (start_of_next_year - start_of_year).days

            # Calculate the decimal year
            decimal_year = date.year + days_since_start_of_year / total_days_in_year
            dates[ind] = float(decimal_year)

        # make plot
        self.ax.plot(dates, chi_squares, color="black")
        self.ax.scatter(dates, chi_squares, color="black")

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Reduced Chi-Square of Modelfits', fontsize=font_size_axis_title)

        try:
            self.set_limits([np.min(dates) - 0.5, np.max(dates) + 0.5],
                            [np.min(chi_squares) - 1, np.max(chi_squares) + 1])
        except:
            pass

    def set_limits(self, x, y):
        self.ax.set_xlim(x)
        self.ax.set_ylim(y)

    def set_scale(self, x, y):
        self.ax.set_xscale(x)
        self.ax.set_yscale(y)

    def plot_kinematic_fit(self, x_min, x_max, fit_params, color, t_mid=0, label=""):
        fit = np.poly1d(fit_params)
        x_values = np.linspace(x_min, x_max, 1000)
        self.ax.plot(x_values, fit(x_values - t_mid), color, label=label)

    def plot_kinematic_2d_fit(self, x_min, x_max, fit_params_x, fit_params_y, color, t_mid=0, label=""):

        fit_x = np.poly1d(fit_params_x)
        fit_y = np.poly1d(fit_params_y)
        x_values = np.linspace(x_min, x_max, 1000)
        x_cor = x_values - t_mid
        distance = np.sqrt(fit_x(x_cor) ** 2 + fit_y(x_cor) ** 2)
        self.ax.plot(x_values, distance, color=color, label=label)

    def plot_linear_fit(self, x_min, x_max, slope, y0, color, label=""):
        def y(x):
            return slope * x + y0

        self.ax.plot([x_min, x_max], [y(x_min), y(x_max)], color, label=label)

    def plot_coreshift_fit(self, fit_result):

        # read out fit_results
        k_r = fit_result["k_r"]
        r0 = fit_result["r0"]
        ref_freq = fit_result["ref_freq"]
        freqs = fit_result["freqs"]
        coreshifts = fit_result["coreshifts"]
        coreshift_err = fit_result["coreshift_err"]

        # define core shift function (Lobanov 1998)
        def delta_r(nu, k_r, r0, ref_freq):
            return r0 * ((nu / ref_freq) ** (-1 / k_r) - 1)

        # do plot
        plt.errorbar(freqs, coreshifts, yerr=coreshift_err, fmt=".", linetype=None, label='Data', color='red')
        nu_fine = np.linspace(min(freqs), max(freqs), 100)
        delta_r_fitted = delta_r(nu_fine, k_r, r0, ref_freq)
        plt.plot(nu_fine, delta_r_fitted, label='Fitted Curve', color='blue')
        plt.xlabel('Frequency [GHz]')
        plt.ylabel(f'Distance to {"{:.1f}".format(ref_freq)}GHz core [$\mu$as]')
        plt.legend()

    def plot_spectral_fit(self, fit_result, xr=np.arange(1, 300, 0.01), annotate_fit_results=True):
        """
        Input:
            fit_result: Dictionary with spectral fit results from "fit_comp_spectrum" of ComponentCollection object
            xr: numpy-array with x-values to use for plot
            annotate_fit_results: Boolean to choose whether to print fit functions and chi^2
        """
        props = dict(boxstyle='round', fc='w', alpha=0.5)
        exponent = -2
        ymin = float('1e{}'.format(exponent))

        if fit_result["fit"] == "PL":
            textstr = '\n'.join((
                r'$\alpha={:.2f}\pm{:.2f}$'.format(fit_result["alpha"], fit_result["alphaE"])
            ))
            if annotate_fit_results:
                self.ax.annotate(textstr, xy=(0.05, 0.1), xycoords='axes fraction', fontsize=8, bbox=props)
            self.ax.plot(xr, ff.powerlaw(fit_result["pl_p"], xr), 'k', lw=0.5)
            y1 = ff.powerlaw(fit_result["pl_p"] - fit_result["pl_sd"], xr)
            y2 = ff.powerlaw(fit_result["pl_p"] + fit_result["pl_sd"], xr)
            self.ax.fill_between(xr, y1, y2, alpha=0.3)

        elif fit_result["fit"] == "SN":
            if fit_result["fit_free_ssa"]:
                textstr = '\n'.join((
                    r'$\nu_m={:.2f}$'.format(fit_result["num"]),
                    r'$S_m={:.2f}$'.format(fit_result["Sm"]),
                    r'$\alpha_{{thin}}={:.2f}$'.format(fit_result["athin"]),
                    r'$\alpha_{{thick}}={:.2f}$'.format(fit_result["athick"]),
                    r'$\chi_\mathrm{{red}}^2={:.2f}$'.format(fit_result["chi2"])
                ))
            else:
                textstr = '\n'.join((
                    r'$\nu_m={:.2f}$'.format(fit_result["num"]),
                    r'$S_m={:.2f}$'.format(fit_result["Sm"]),
                    r'$\alpha_{{thin}}={:.2f}$'.format(fit_result["athin"]),
                    r'$\chi_\mathrm{{red}}^2={:.2f}$'.format(fit_result["chi2"])
                ))

            if annotate_fit_results:
                self.ax.annotate(textstr, xy=(0.05, 0.1), xycoords='axes fraction', fontsize=8, bbox=props)
                sn_low = fit_result["sn_p"] - fit_result["sn_sd"]
                sn_up = fit_result["sn_p"] + fit_result["sn_sd"]

            for jj, SNL in enumerate(sn_low[:2]):
                if SNL < 0:
                    logging.warning("Uncertainties for SN fit large, limit peak flux and freq \n")
                    if jj == 0:
                        sn_low[jj] = 0.1
                    if jj == 1:
                        sn_low[jj] = ymin

            if fit_result["fit_free_ssa"]:
                self.ax.plot(xf, ff.Snu(fit_result["sn_p"], xr), 'k', lw=0.5)
                y1 = ff.Snu(sn_low, xr)
                y2 = ff.Snu(sn_up, xr)
            else:
                self.ax.plot(xr, ff.Snu_real(fit_result["sn_p"], xr), 'k', lw=0.5)
                y1 = ff.Snu_real(sn_low, xr)
                y2 = ff.Snu_real(sn_up, xr)

            self.ax.fill_between(xr, y1, y2, alpha=0.2)
