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
from vcat.helpers import get_sigma_levs, getComponentInfo, convert_image_to_polar
import vcat.VLBI_map_analysis.modules.fit_functions as ff


#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
#mpl.rcParams['font.family'] = 'Quicksand'
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=13
font_size_axis_tick=12

class KinematicPlot(object):
    def __init__(self):

        super().__init__()
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)

    def plot_kinematics(self,component_collection,color):
        if component_collection.length()>0:
            self.ax.scatter(component_collection.year,component_collection.dist,c=color,marker=".")
        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Distance from Core [mas]', fontsize=font_size_axis_title)

    def plot_fluxs(self,component_collection,color):
        if component_collection.length() > 0:
            self.ax.plot(component_collection.year, component_collection.fluxs, c=color, label=component_collection.name,marker=".")
        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Flux Density [Jy]', fontsize=font_size_axis_title)

    def plot_tbs(self,component_collection,color):
        if component_collection.length() > 0:
            lower_limit_inds = np.where(np.array(component_collection.tbs_lower_limit))[0]
            tb_value_inds = np.where(np.array(component_collection.tbs_lower_limit)==False)[0]
            self.ax.plot(np.array(component_collection.year)[tb_value_inds],
                         np.array(component_collection.tbs)[tb_value_inds], c=color, label=component_collection.name,marker=".")
            self.ax.scatter(np.array(component_collection.year)[lower_limit_inds],
                            np.array(component_collection.tbs)[lower_limit_inds], c=color, marker="^")

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Brightness Temperature [K]', fontsize=font_size_axis_title)
        self.ax.set_yscale("log")
    
    def plot_spectrum(self,component_collection,color):
        if component_collection.length() > 0:
            self.ax.scatter(np.array(component_collection.freqs)*1e-9,component_collection.fluxs,
                    c=color,label=component_collection.name,marker=".")
        self.ax.set_xlabel("Frequency [GHz]",fontsize=font_size_axis_title)
        self.ax.set_ylabel("Flux Density [Jy]",fontsize=font_size_axis_title)

    def plot_chi_square(self,uvf_files,modelfit_files,difmap_path):

        #calculate chi-squares
        chi_squares=[]
        dates=[]

        for ind,uvf in enumerate(uvf_files):
            df=getComponentInfo(modelfit_files[ind],scale=self.clean_image.scale)
            freq=get_freq(modelfit_files[ind])
            write_mod_file(df,"modelfit.mod",freq,adv=True)
            chi_square=get_model_chi_square_red(uvf,"modelfit.mod",difmap_path)
            chi_squares.append(chi_square)
            dates.append(get_date(modelfit_files[ind]))
            os.system("rm -rf modelfit.mod")

        chi_squares=np.array(chi_squares)

        for ind,dat in enumerate(dates):
            #calculate decimal year
            date = datetime.strptime(dat, "%Y-%m-%d")

            # Calculate the start of the year and the start of the next year
            start_of_year = datetime(date.year, 1, 1)
            start_of_next_year = datetime(date.year + 1, 1, 1)

            # Calculate the number of days since the start of the year and total days in the year
            days_since_start_of_year = (date - start_of_year).days
            total_days_in_year = (start_of_next_year - start_of_year).days

            # Calculate the decimal year
            decimal_year = date.year + days_since_start_of_year / total_days_in_year
            dates[ind]=float(decimal_year)

        #make plot
        self.ax.plot(dates,chi_squares,color="black")
        self.ax.scatter(dates,chi_squares,color="black")

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Reduced Chi-Square of Modelfits', fontsize=font_size_axis_title)

        try:
            self.set_limits([np.min(dates)-0.5,np.max(dates)+0.5],
                            [np.min(chi_squares)-1,np.max(chi_squares)+1])
        except:
            pass

    def set_limits(self,x,y):
        self.ax.set_xlim(x)
        self.ax.set_ylim(y)
    
    def set_scale(self,x,y):
        self.ax.set_xscale(x)
        self.ax.set_yscale(y)

    def plot_linear_fit(self,x_min,x_max,slope,y0,color,label=""):
        def y(x):
            return slope*x+y0
        self.ax.plot([x_min,x_max],[y(x_min),y(x_max)],color,label=label)

    def plot_coreshift_fit(self,fit_result):

        #read out fit_results
        k_r = fit_result["k_r"]
        r0 = fit_result["r0"]
        ref_freq = fit_result["ref_freq"]
        freqs = fit_result["freqs"]
        coreshifts = fit_result["coreshifts"]
        coreshift_err = fit_result["coreshift_err"]

        #define core shift function (Lobanov 1998)
        def delta_r(nu,k_r,r0,ref_freq):
            return r0*((nu/ref_freq)**(-1/k_r)-1)

        #do plot
        plt.errorbar(freqs, coreshifts, yerr=coreshift_err,fmt=".",linetype=None,label='Data', color='red')
        nu_fine = np.linspace(min(freqs), max(freqs), 100)
        delta_r_fitted = delta_r(nu_fine, k_r, r0, ref_freq)
        plt.plot(nu_fine, delta_r_fitted, label='Fitted Curve', color='blue')
        plt.xlabel('Frequency [GHz]')
        plt.ylabel(f'Distance to {"{:.1f}".format(ref_freq)}GHz core [$\mu$as]')
        plt.legend()

    def plot_spectral_fit(self,fit_result,xr=np.arange(1,300,0.01),annotate_fit_results=True):
        """
        Input:
            fit_result: Dictionary with spectral fit results from "fit_comp_spectrum" of ComponentCollection object
            xr: numpy-array with x-values to use for plot
            annotate_fit_results: Boolean to choose whether to print fit functions and chi^2
        """
        props = dict(boxstyle='round',fc='w',alpha=0.5)
        exponent = -2
        ymin=float('1e{}'.format(exponent)) 


        if fit_result["fit"]=="PL":
            textstr = '\n'.join((
                r'$\alpha={:.2f}\pm{:.2f}$'.format(fit_result["alpha"],fit_result["alphaE"])
            ))
            if annotate_fit_results:
                self.ax.annotate(textstr, xy=(0.05,0.1),xycoords='axes fraction',fontsize=8,bbox=props)
            self.ax.plot(xr,ff.powerlaw(fit_result["pl_p"],xr),'k',lw=0.5)
            y1 = ff.powerlaw(fit_result["pl_p"]-fit_result["pl_sd"],xr)
            y2 = ff.powerlaw(fit_result["pl_p"]+fit_result["pl_sd"],xr)
            self.ax.fill_between(xr,y1,y2,alpha=0.3)

        elif fit_result["fit"]=="SN":
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
                self.ax.annotate(textstr, xy=(0.05,0.1),xycoords='axes fraction',fontsize=8,bbox=props)
                sn_low = fit_result["sn_p"]-fit_result["sn_sd"]
                sn_up = fit_result["sn_p"]+fit_result["sn_sd"]
            
            for jj, SNL in enumerate(sn_low[:2]):
                if SNL <0:
                    sys.stdout.write("Uncertainties for SN fit large, limit peak flux and freq \n")
                    if jj == 0:
                        sn_low[jj] = 0.1
                    if jj == 1:
                        sn_low[jj] = ymin

            if fit_result["fit_free_ssa"]:
                self.ax.plot(xf,ff.Snu(fit_result["sn_p"],xr),'k',lw=0.5)
                y1 = ff.Snu(sn_low,xr)
                y2 = ff.Snu(sn_up,xr)
            else:
                self.ax.plot(xr,ff.Snu_real(fit_result["sn_p"],xr),'k',lw=0.5)
                y1 = ff.Snu_real(sn_low,xr)
                y2 = ff.Snu_real(sn_up,xr)

            self.ax.fill_between(xr,y1,y2,alpha=0.2)

class EvolutionPlot(object):
    def __init__(self,xlabel="",ylabel="",font_size_axis_title=10):

        super().__init__()
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)
        self.ax.set_xlabel(xlabel, fontsize=font_size_axis_title)
        self.ax.set_ylabel(ylabel, fontsize=font_size_axis_title)

    def plotEvolution(self,mjds,value,c="black",marker=".",label=""):
        self.ax.scatter(mjds, value, c=c, marker=marker,label=label)


        
class FitsImage(object):
    """Class that generates Matplotlib graph for a VLBI image.
    
    Attributes:
        image_data: ImageData object which includes the VLBI image.
        stokes_i_sigma_cut: Select the sigma cut to apply to Stokes I
        plot_mode: Choose which parameter to plot (options: "stokes_i","lin_pol","frac_pol")
        im_colormap: Choose whether to do colormap or not
        contour: Choose whether to do contour plot or not
        contour_color: Choose contour color
        contour_cmap: Choose colormap for contours
        contour_alpha: Choose transparency for contours
        contour_width: Choose width of contours
        im_color: Choose colormap name
        plot_ridgeline: Choose to plot ridgeline
        ridgeline_color: Color for ridgeline
        plot_counter_ridgeline: Choose to plot counter ridgeline
        counter_ridgeline_color= Choose color for counter ridgeline
        plot_beam: Choose whether to plot the beam or not
        overplot_gauss: Choose whether to overplot modelfit components (if available in image_data)
        component_color: Choose color to plot components
        overplot_clean: Choose whether to overplot clean components (if available in image_data)
        xlim: Choose X-plot limits
        ylim: Choose Y-plot limits
        plot_evpa: Choose whether to plot EVPAs or not
        evpa_width: Choose EVPA width
        evpa_len: Choose EVPA len in pixels
        lin_pol_sigma_cut: Choose lowest sigma contour for lin pol
        evpa_distance: Choose the distance of EVPA vectors to plot in pixels
        rotate_evpa: rotate EVPAs by a given angle in degrees (North through East)
        evpa_color: Choose EVPA color
        title: Choose plot title
        rcParams: Put in matplotlib rcParams for more modification to the plots        
    """
    def __init__(self,
                 image_data, #ImageData object
                 stokes_i_sigma_cut=3, #sigma_cut for stokes_i_contours
                 plot_mode="stokes_i", #possible modes "stokes_i", "lin_pol", "frac_pol"
                 im_colormap=False, #Choose whether to do colormap or not
                 contour=True, #Choose whether to do contour plot or not
                 contour_color = 'grey',  # input: array of color-strings; if None, the contour-colormap (contour_cmap) will be used
                 contour_cmap = None,  # matplotlib colormap string
                 contour_alpha = 1,  # transparency
                 contour_width = 0.5,  # contour linewidth
                 im_color='', # string for matplotlib colormap
                 do_colorbar=False, #choose whether to display colorbar
                 plot_ridgeline=False, #choose whether to display the ridgeline
                 ridgeline_color="red", #choose ridgeline color
                 plot_counter_ridgeline= False,
                 counter_ridgeline_color= "red",
                 plot_line="", #Provide two points for plotting a line
                 line_color="black",
                 line_width=2, #width of the line
                 plot_beam=True, #choose whether to plot beam or not
                 overplot_gauss=False, #choose whether to plot modelfit components
                 component_color="black", # choose component color for Gauss component
                 plot_comp_ids=False, #plot component ids
                 overplot_clean=False, #choose whether to plot clean components
                 plot_mask=False, #choose whether to plot mask
                 xlim=[], #xplot limits, e.g. [5,-5]
                 ylim=[], #yplot limits
                 levs="", #predefined plot levels
                 levs1="", #predefined plot levels1
                 levs_linpol="", #predefined linpol levs
                 levs1_linpol="", #predefined linepol levs1
                 stokes_i_vmax="", #input vmax for plot
                 fracpol_vmax="", #input vmax for plot
                 linpol_vmax="", #input vmax for plot
                 plot_polar=False, #choose to plot image in polar coordinates
                 ###HERE STARTS POLARIZATION INPUT
                 plot_evpa=False, #decide whether to plot EVPA or not
                 evpa_width=2, #choose width of EVPA lines
                 evpa_len=8,  # choose length of EVPA in pixels
                 lin_pol_sigma_cut=3,  # choose lowest sigma contour for Lin Pol plot
                 evpa_distance=10,  # choose distance of EVPA vectors to draw in pixels
                 rotate_evpa=0, # rotate EVPAs by a given angle in degrees (North through East)
                 evpa_color="white", # set EVPA color for plot
                 title="", # plot title (default is date)
                 background_color="white", #background color
                 ax=None, #define custom matplotlib axes to plot on
                 fig=None, #define custom figure
                 font_size_axis_title=font_size_axis_title, #set fontsize for axis title
                 font_size_axis_tick=font_size_axis_tick, #set fontsize for axis ticks
                 rcparams={} # option to modify matplotlib look
                 ):

        super().__init__()

        mpl.rcParams.update(rcparams)
        
        #read image
        self.clean_image = image_data
        self.clean_image_file = self.clean_image.file_path
        self.model_image_file = self.clean_image.model_file_path

        #set parameters
        self.plot_mode=plot_mode
        self.name = self.clean_image.name
        self.freq = self.clean_image.freq
        X = self.clean_image.X
        Y = self.clean_image.Y
        Z = self.clean_image.Z
        self.Z=Z
        unit = self.clean_image.unit
        scale = self.clean_image.scale
        degpp = self.clean_image.degpp
        extent = self.clean_image.extent
        date=self.clean_image.date
        lin_pol=self.clean_image.lin_pol
        self.lin_pol=self.clean_image.lin_pol
        self.evpa_width=evpa_width
        # Set beam parameters
        beam_maj = self.clean_image.beam_maj
        beam_min = self.clean_image.beam_min
        beam_pa = self.clean_image.beam_pa
        self.evpa_color=evpa_color
        self.background_color=background_color
        self.noise_method=self.clean_image.noise_method
        self.do_colorbar=do_colorbar
        self.ridgeline_color=ridgeline_color
        self.counter_ridgeline_color=counter_ridgeline_color
        self.stokes_i_vmax=stokes_i_vmax
        self.linpol_vmax=linpol_vmax
        self.fracpol_vmax=fracpol_vmax
        self.col=""

        #modify these parameters if polar plot is selected
        if plot_polar:
            #currently only support colormap so turn off everything else:
            overplot_gauss=False
            overplot_clean=False
            plot_mask=False
            #Convert Stokes I
            R, Theta, Z_polar = convert_image_to_polar(X,Y, Z)
            extent=[Theta.min(),Theta.max(),R.min(),R.max()]
            Z=Z_polar.T
            self.Z=Z

            #Convert Lin Pol
            try:
                R, Theta, lin_pol = convert_image_to_polar(X,Y, lin_pol)
                lin_pol = lin_pol.T
                self.lin_pol = lin_pol
            except:
                pass



        #plot limits
        ra_max,ra_min,dec_min,dec_max=extent

        if len(xlim) == 2:
            ra_max, ra_min = xlim
        if len(ylim) == 2:
            dec_min, dec_max = ylim

        if ax==None and fig==None:
            self.fig, self.ax = plt.subplots(1, 1)
        else:
            if fig==None:
                self.fig = plt.figure()
            else:
                self.fig = fig
            self.ax = ax


        #set background color
        self.ax.set_facecolor(self.background_color)

        self.components=[]

        #component default color
        self.component_color = component_color

        fit_noise = True  # if True, the noise value and rms deviation will be fitted as described in the PhD-thesis of Moritz Böck (https://www.physik.uni-wuerzburg.de/fileadmin/11030400/Dissertation_Boeck.pdf); if False, the noise frome difmap will be used

        # Image colormap
        self.im_colormap = im_colormap  # if True, a image colormap will be done

        clean_alpha = 1  # float for sympol transparency

        #get sigma levs
        if not isinstance(levs,list) and not isinstance(levs1,list):
            levs, levs1 = get_sigma_levs(Z,stokes_i_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_noise)

        self.levs=levs
        self.levs1=levs1

        # Image colormap
        if self.im_colormap == True and plot_mode=="stokes_i":
            self.plotColormap(Z,im_color,levs,levs1,extent,do_colorbar=self.do_colorbar)
            contour_color="white"


        if (plot_mode=="lin_pol" or plot_mode=="frac_pol") and np.sum(lin_pol)!=0:

            if not isinstance(levs_linpol,list) and not isinstance(levs1_linpol,list):
                levs_linpol, levs1_linpol = get_sigma_levs(lin_pol, lin_pol_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_pol_noise)


            if plot_mode=="lin_pol":
                self.plotColormap(lin_pol,im_color,levs_linpol,levs1_linpol,extent,
                                  label="Linear Polarized Intensity [Jy/beam]",do_colorbar=self.do_colorbar)
            if plot_mode=="frac_pol":
                plot_lin_pol = np.array(lin_pol)
                plot_frac_pol = plot_lin_pol / np.array(self.clean_image.Z)
                plot_frac_pol = np.ma.masked_where((plot_lin_pol < levs1_linpol[0]) | (self.clean_image.Z<levs1[0]),
                                                  plot_frac_pol)

                self.plotColormap(plot_frac_pol,im_color,np.zeros(100),[0.00],extent,
                                  label="Fractional Linear Polarization",do_colorbar=self.do_colorbar)

        if plot_mode=="residual":
            if plot_polar:
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.residual_map)
                Z=Z.T
            else:
                Z=self.clean_image.residual_map
            self.plotColormap(Z,im_color,levs,levs1,extent,label="Residual Flux Density [Jy/beam]", do_colorbar=self.do_colorbar)
        if plot_mode=="spix":
            self.plotColormap(self.clean_image.spix,im_color,levs,levs1,extent,label="Spectral Index", do_colorbar=self.do_colorbar)

        if plot_mode=="rm":
            if plot_polar:
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.rm)
                Z=Z.T
            else:
                Z=self.clean_image.rm

            rm=np.ma.masked_where((abs(Z) > 20000),Z)
            self.plotColormap(rm, im_color, levs, levs1, extent, label="Rotation Measure [rad/m^2]",do_colorbar=self.do_colorbar)

        if plot_mode == "turnover_freq" or plot_mode=="turnover":
            if plot_polar:
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z=Z.T
            else:
                Z=self.clean_image.turnover

            to=np.ma.masked_where(Z==0,Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label="Turnover Frequency [GHz]", do_colorbar=self.do_colorbar)
        if plot_mode == "turnover_flux":
            if plot_polar:
                _, _, Z_filter = convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z_filter = Z_filter.T
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover_flux)
                Z=Z.T
            else:
                Z_filter=self.clean_image.turnover
                Z=self.clean_image.turnover_flux

            to=np.ma.masked_where(Z_filter==0,Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label="Turnover Flux [Jy/beam]",
                              do_colorbar=self.do_colorbar)
        if plot_mode == "turnover_error":
            if plot_polar:
                _, _, Z_filter = convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z_filter = Z_filter.T
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover_error)
                Z=Z.T
            else:
                Z_filter=self.clean_image.turnover
                Z=self.clean_image.turnover_error

            to=np.ma.masked_where(Z_filter==0,Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label="Turnover Error [GHz]",
                              do_colorbar=self.do_colorbar)
        if plot_mode == "turnover_chisquare":
            if plot_polar:
                _, _, Z_filter = convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z_filter = Z_filter.T
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover_chi_sq)
                Z=Z.T
            else:
                Z_filter=self.clean_image.turnover
                Z=self.clean_image.turnover_chi_sq

            to = np.ma.masked_where(Z_filter == 0, Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label=r"Turnover $\chi^2$",
                              do_colorbar=self.do_colorbar)

        if plot_evpa and np.sum(lin_pol)!=0:
            if not isinstance(levs_linpol,list) and not isinstance(levs1_linpol,list):
                levs_linpol, levs1_linpol = get_sigma_levs(lin_pol, lin_pol_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_pol_noise)
            self.plotEvpa(self.clean_image.evpa, rotate_evpa, evpa_len, evpa_distance, levs1_linpol, levs1)

        # Contour plot
        if contour == True:
            if contour_cmap=="" or contour_cmap==None:
                contour_cmap=None
            else:
                contour_color=None

            if plot_polar:
                self.ax.contour(Theta[:,0],R[0,:], Z, linewidths=contour_width, levels=levs, colors=contour_color,
                                alpha=contour_alpha,
                                cmap=contour_cmap)

            else:
                self.ax.contour(X, Y, self.Z, linewidths=contour_width, levels=levs, colors=contour_color,
                            alpha=contour_alpha,
                            cmap=contour_cmap)

        # Set beam ellipse, sourcename and observation date positions
        size_x = np.absolute(ra_max) + np.absolute(ra_min)
        size_y = np.absolute(dec_max) + np.absolute(dec_min)
        if size_x > size_y:
            ell_x = ra_max - beam_maj
            ell_y = dec_min + beam_maj
        else:
            ell_x = ra_max - beam_maj
            ell_y = dec_min + beam_maj

        if plot_beam:
            # Plot beam
            beam = Ellipse([ell_x, ell_y], beam_maj, beam_min,angle= -beam_pa + 90, fc='grey')
            self.ax.add_artist(beam)

        if title=="":
            self.ax.set_title(date + " " + "{:.0f}".format(self.freq/1e9)+" GHz", fontsize=font_size_axis_title)
        else:
            self.ax.set_title(title, fontsize=font_size_axis_title)

        #set x/y tick size
        self.ax.tick_params(axis="y",labelsize=font_size_axis_tick)
        self.ax.tick_params(axis="x",labelsize=font_size_axis_tick)

        if plot_mask:
            if plot_polar:
                _,_,mask_p=convert_image_to_polar(X,Y,self.clean_image.mask)
                mask=mask_p.T
            else:
                mask=self.clean_image.mask
            self.plotColormap(mask,"gray_r",np.zeros(100),[0.00],extent,label="Mask",do_colorbar=self.do_colorbar)


        # Read modelfit files in
        if (overplot_gauss == True) or (overplot_clean == True):
            model_df = getComponentInfo(self.model_image_file,scale=self.clean_image.scale)

            # sort in gauss and clean components
            model_gauss_df = model_df
            model_clean_df = model_df[model_df["Major_axis"] == 0.].reset_index()

            # Overplot clean components
            if overplot_clean == True:
                c_x = model_clean_df["Delta_x"]
                c_y = model_clean_df["Delta_y"]
                c_flux = model_clean_df["Flux"]

                for j in range(len(c_x)):
                    if c_flux[j] < 0.:
                        self.ax.plot(c_x[j] * scale, c_y[j] * scale, marker='+', color='red', alpha=clean_alpha,
                                     linewidth=0.2, zorder=2)
                    else:
                        self.ax.plot(c_x[j] * scale, c_y[j] * scale, marker='+', color='green', alpha=clean_alpha,
                                     linewidth=0.2, zorder=2)

            # Overplot Gaussian components
            if overplot_gauss == True:

                g_x = model_gauss_df["Delta_x"]
                g_y = model_gauss_df["Delta_y"]
                g_maj = model_gauss_df["Major_axis"]
                g_min = model_gauss_df["Minor_axis"]
                g_pos = model_gauss_df["PA"]
                g_flux = model_gauss_df["Flux"]
                g_date = model_gauss_df["Date"]
                g_mjd = model_gauss_df["mjd"]
                g_year = model_gauss_df["Year"]

                for j in range(len(g_x)):
                    # plot component
                    if plot_comp_ids:
                        for i,comp in enumerate(self.clean_image.components):
                            if comp.flux==g_flux[j]:
                                comp_id=comp.component_number
                        component_plot = self.plotComponent(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], scale, id=comp_id)
                    else:
                        component_plot = self.plotComponent(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], scale)


                    #calculate noise at the position of the component
                    try:
                        component_noise=get_noise_from_residual_map(self.clean_image.residual_map_path, g_x[j]*scale,g_y[j]*scale,np.max(X)/10,np.max(Y)/10,scale=scale)#TODO check if the /10 width works and make it changeable
                    except:
                        component_noise=self.clean_image.noise_3sigma

        if plot_ridgeline:
            #plot ridgeline in image
            self.ax.plot(self.clean_image.ridgeline.X_ridg,self.clean_image.ridgeline.Y_ridg,c=self.ridgeline_color,zorder=6)

        if plot_counter_ridgeline:
            #plot counterridgeline in image
            self.ax.plot(self.clean_image.counter_ridgeline.X_ridg, self.clean_image.counter_ridgeline.Y_ridg, c=self.counter_ridgeline_color,
                         zorder=6)

        if plot_line!="":
            self.ax.plot([plot_line[0][0],plot_line[1][0]],[plot_line[0][1],plot_line[1][1]],linewidth=line_width,c=line_color,zorder=7)

        self.xmin, self.xmax = ra_min, ra_max
        self.ymin, self.ymax = dec_min, dec_max

        self.levs_linpol = levs_linpol
        self.levs1_linpol = levs1_linpol

        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)

        # Plot look tuning
        if plot_polar:
            self.ax.set_xlim(np.min(Theta),np.max(Theta))
            self.ax.set_ylim(np.min(R),np.max(R))
            self.ax.invert_xaxis()
            self.ax.set_aspect('auto', adjustable='box', anchor='C')
            self.ax.set_xlabel("Position Angle [°]")
            self.ax.set_ylabel("Radius [mas]")
        else:

            self.ax.set_aspect('equal', adjustable='box', anchor='C')
            self.ax.set_xlim(ra_min, ra_max)
            self.ax.set_ylim(dec_min, dec_max)
            self.ax.invert_xaxis()
            self.ax.set_xlabel('Relative R.A. [' + unit + ']',fontsize=font_size_axis_title)
            self.ax.set_ylabel('Relative DEC. [' + unit + ']',fontsize=font_size_axis_title)
        self.fig.tight_layout()


    def plotColormap(self,
                     Z, #2d data array to plot
                     im_color, #colormap to use
                     levs, #sigma levs output
                     levs1, #sigma levs output
                     extent, #plot lims x_min,x_max,y_min,y_max
                     label="Flux Density [Jy/beam]", #label for colorbar
                     do_colorbar=False
                     ):

        #OPTIONS for fractional polarization plot
        if label=="Fractional Linear Polarization":
            vmin=0
            if self.fracpol_vmax=="":
                vmax = np.max([0.1, np.min([0.8, np.max(Z)*.8/.7])])
                self.fracpol_vmax=vmax
            else:
                vmax=self.fracpol_vmax
            if im_color == "":
                im_color = cmaps.neon_r

            if vmax > 0.4:
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               norm=colors.SymLogNorm(linthresh=0.4,
                                                       vmax=vmax, vmin=vmin), extent=extent)
            else:
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               vmax=vmax, vmin=vmin, extent=extent)
            if vmax >= 0.4:
                ticks = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
                ticklabels = ["0.0", "", "0.1", "", "0.2", "", "0.3", "", "0.4"]
                # add appropriate ticklabels up to 0.7.
                for tickval in ["0.5", "0.6", "0.7","0.8"]:
                    if vmax >= float(tickval):
                        ticks = np.append(ticks, float(tickval))
                        ticklabels.append(tickval)

                if do_colorbar:
                    divider = make_axes_locatable(self.ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax,ticks=ticks)
                    cbar.set_label(label,fontsize=self.ax.xaxis.label.get_size())
            elif vmax <=0.2:
                ticks = np.array([0.0, 0.025, 0.05, 0.75, 0.1, 0.125, 0.15, 0.175, 0.2])
                ticklabels = ["0.000", "0.025", "0.050", "0.075", "0.100", "0.125", "0.150", "0.175", "0.200"]
                final_labels=[]
                final_ticks=[]
                for tickval in ticks:
                    if vmax >= float(tickval):
                        final_ticks = np.append(final_ticks, float(tickval))
                        final_labels.append(tickval)
                if do_colorbar:
                    divider = make_axes_locatable(self.ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax,ticks=final_ticks)
                    cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())
            else:
                if do_colorbar:
                    divider = make_axes_locatable(self.ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                    cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())
            if do_colorbar:
                cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        elif label=="Linear Polarized Intensity [Jy/beam]":
            if im_color =="":
                im_color = "cubehelix_r"

            linthresh = 10.0 * levs1[0]
            if self.linpol_vmax=="":
                vmax = np.max([np.max(Z), 10.0 * levs1[0]])
                self.linpol_vmax=vmax
            else:
                vmax=self.linpol_vmax

            vmin = 0
            if linthresh < 0.5 * np.max([vmax, -vmin]):
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               norm=colors.SymLogNorm(linthresh=linthresh,
                                                       vmax=vmax, vmin=vmin),extent=extent)
            else:
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               vmax=vmax, vmin=vmin,extent=extent)

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())
        elif label=="Mask":
            if im_color=="":
                im_color="inferno"
            self.ax.imshow(Z, cmap=im_color, vmin=0, vmax=1, interpolation='none', alpha=Z.astype(float), extent=extent, origin="lower", zorder=10)

        elif label=="Residual Flux Density [Jy/beam]":
            if im_color=="":
                im_color="gray"
            self.col = self.ax.imshow(Z, cmap=im_color, extent=extent, origin="lower")

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        elif label=="Spectral Index":
            if im_color=="":
                im_color="hot_r"

            self.col = self.ax.imshow(Z, cmap=im_color, vmin=self.clean_image.spix_vmin, vmax=self.clean_image.spix_vmax, extent=extent, origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        elif label=="Rotation Measure [rad/m^2]":
            if im_color=="":
                im_color="coolwarm"

            if self.clean_image.rm_vmin!="" and self.clean_image.rm_vmax!="":
                self.col = self.ax.imshow(Z, cmap=im_color, vmin=self.clean_image.rm_vmin, vmax=self.clean_image.rm_vmax, extent=extent, origin='lower')
            else:
                #scale up and down equally
                vmax=np.max(abs(Z))
                vmin=-vmax
                self.col = self.ax.imshow(Z, cmap=im_color, vmin=vmin, vmax=vmax, extent=extent, origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        elif label=="Turnover Frequency [GHz]" or label=="Turnover Flux [Jy/beam]" or label=="Turnover Error [GHz]" or label=="Turnover $\Chi^2$":
            if im_color=="":
                im_color="inferno"

            self.col = self.ax.imshow(Z, cmap=im_color, extent=extent, origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        else:
            if im_color=="":
                im_color="inferno"

            if self.stokes_i_vmax=="":
                vmax = 0.5 * np.max(Z)
                self.stokes_i_vmax=vmax
            else:
                vmax=self.stokes_i_vmax

            self.col = self.ax.imshow(Z, cmap=im_color, norm=colors.SymLogNorm(linthresh=abs(levs1[0]), linscale=0.5, vmin=levs1[0],
                                                                        vmax=vmax, base=10.), extent=extent,
                                origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

    def plotComponent(self,x,y,maj,min,pos,scale,id=""):

        # Plotting ellipses
        comp = Ellipse([x * scale, y * scale], maj * scale, min * scale,angle= -pos + 90,
                       fill=False, zorder=2, color=self.component_color, lw=0.5)
        ellipse=self.ax.add_artist(comp)

        #deal with point like components
        if maj==0 and min==0:
            maj=0.1/scale
            min=0.1/scale

        # Plotting axes of the ellipses
        maj1_x = x - np.sin(-np.pi / 180 * pos) * maj * 0.5
        maj1_y = y + np.cos(-np.pi / 180 * pos) * maj * 0.5
        maj2_x = x + np.sin(-np.pi / 180 * pos) * maj * 0.5
        maj2_y = y - np.cos(-np.pi / 180 * pos) * maj * 0.5

        min1_x = x - np.sin(-np.pi / 180 * (pos + 90)) * min * 0.5
        min1_y = y + np.cos(-np.pi / 180 * (pos + 90)) * min * 0.5
        min2_x = x + np.sin(-np.pi / 180 * (pos + 90)) * min * 0.5
        min2_y = y - np.cos(-np.pi / 180 * (pos + 90)) * min * 0.5

        line1=self.ax.plot([maj1_x * scale, maj2_x * scale], [maj1_y * scale, maj2_y * scale], color=self.component_color, lw=0.5)
        line2=self.ax.plot([min1_x * scale, min2_x * scale], [min1_y * scale, min2_y * scale], color=self.component_color, lw=0.5)

        if id!="":
            self.ax.text(maj1_x*scale,maj1_y*scale,str(id),fontsize=10)
        return [ellipse,line1,line2]


    def change_plot_lim(self,x_min,x_max,y_min,y_max):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    def plotEvpa(self,evpa,rotate_evpa,evpa_len,evpa_distance,levs1_linpol,levs1_i):

        evpa_len=evpa_len*self.clean_image.degpp*self.clean_image.scale

        stokes_i=self.Z
        # plot EVPA
        evpa = evpa + rotate_evpa / 180 * np.pi

        # create mask where to plot EVPA (only where stokes i and lin pol have plotted contours)
        mask = np.zeros(np.shape(stokes_i), dtype=bool)
        mask[:] = (self.lin_pol > levs1_linpol[0]) * (stokes_i > levs1_i[0])
        YLoc, XLoc = np.where(mask)

        y_evpa = evpa_len * np.cos(evpa[mask])
        x_evpa = evpa_len * np.sin(evpa[mask])
        evpa=evpa[mask]

        SelPix = range(0, len(stokes_i), int(evpa_distance))

        lines = []
        for i in range(0, len(XLoc)):
            if XLoc[i] in SelPix and YLoc[i] in SelPix:
                Xpos = float(self.clean_image.X[XLoc[i]])
                Ypos = float(self.clean_image.Y[YLoc[i]])
                Y0 = float(Ypos - y_evpa[i] / 2.)
                Y1 = float(Ypos + y_evpa[i] / 2.)
                X0 = float(Xpos - x_evpa[i] / 2.)
                X1 = float(Xpos + x_evpa[i] / 2.)
                lines.append(((X0, Y0), (X1, Y1)))
        lines = tuple(lines)

        # plot the evpas
        evpa_lines = LineCollection(lines, colors=self.evpa_color, linewidths=self.evpa_width,zorder=5)
        self.ax.add_collection(evpa_lines)

    def export(self,name):
        #check if name is a directory, if so create generic filename in pdf and png format
        if os.path.isdir(name):
            if name[-1]!="/":
                name+="/"
            name+=self.name+"_"+"{:.2f}".format(self.freq/1e9)+"GHz_"+self.clean_image.date+"_"+self.plot_mode
            self.fig.savefig(name+".png", dpi=300, bbox_inches='tight', transparent=False)
            self.fig.savefig(name+".pdf", dpi=300, bbox_inches='tight', transparent=False)
        else:
            if name.split(".")[-1] in ("png","jpg","jpeg","pdf","gif"):
                self.fig.savefig(name, dpi=300, bbox_inches='tight', transparent=False)
            else:
                self.fig.savefig(name+".png",dpi=300,bbox_inches="tight", transparent=False)

class MultiFitsImage(object):

    def __init__(self,
                 image_cube,  # ImageData object
                 mode="individual", #Choose what effect the parameters have ('individual','freq','epoch','all')
                 swap_axis=False, #If True frequency will be plotted in x-direction and time in y
                 figsize="", #define figsize
                 shared_colormap="individual", #options are 'freq', 'epoch', 'all','individual'
                 shared_colorbar=False, #if true, will plot a shared colorbar according to share_colormap setting
                 shared_sigma="max", #select which common sigma to use options: 'max','min'
                 shared_colorbar_label="", #choose custom colorbar label
                 shared_colorbar_labelsize=10, #choose labelsize of custom colorbar
                 **kwargs #additional plot params
                 ):

        super().__init__()

        self.image_cube=image_cube
        if not swap_axis:
            self.nrows, self.ncols = self.image_cube.shape
        else:
            self.ncols, self.nrows = self.image_cube.shape
        if figsize=="":
            figsize=(3*self.ncols,3*self.nrows)
        self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=figsize)
        self.axes=np.atleast_2d(self.axes)

        if self.axes.shape[0]==self.ncols and self.axes.shape[1]==self.nrows:
            if not self.ncols==self.nrows and not swap_axis:
                self.axes=self.axes.T

        #read in input parameters for individual plots
        if mode=="all":
            #This means kwargs are just numbers
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])

                for i in range(len(self.image_cube.dates)):
                    for j in range(len(self.image_cube.freqs)):
                        kwargs[key][i, j] = value
        elif mode=="freq":
            #allow input parameters per frequency
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape,dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])

                if not isinstance(value,list) or (key in ["xlim","ylim"] and (len(value)==2 and isinstance(value[0],(float,int)) or len(value)==0)):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i,j] = value
                elif len(value)==len(self.image_cube.freqs):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i,j]=value[j]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")
        elif mode=="epoch":

            # allow input parameters per epoch
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])
                if not isinstance(value,list) or (key in ["xlim","ylim"] and (len(value)==2 and isinstance(value[0],(float,int)) or len(value)==0)):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.image_cube.dates):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i, j] = value[i]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")

        elif mode=="individual":
            # allow input parameters per frequency
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])
                if not isinstance(value,list) or (key in ["xlim","ylim"] and (len(value)==2 or len(value)==0)):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.image_cube.images) and len(value[0]) == len(self.image_cube.images[0]):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i,j] = value[i][j]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")
        else:
            raise Exception("Please select valid plot mode ('individual','freq','epoch','all'")

        #check if colormap is shared between plots:
        if shared_colormap == "all":
            noises=image_cube.noises

            if shared_sigma=="max":
                index = np.unravel_index(np.argmax(noises), noises.shape)
            else:
                index = np.unravel_index(np.argmin(noises), noises.shape)

            plot=image_cube.images[index].plot(plot_mode=kwargs["plot_mode"][0,0],im_colormap=True, im_color=kwargs["im_color"][0,0],show=False)

            for i in range(len(self.image_cube.dates)):
                for j in range(len(self.image_cube.freqs)):
                    #get levs:
                    kwargs["levs"][i,j]=plot.levs
                    kwargs["levs1"][i,j]=plot.levs1
                    kwargs["levs_linpol"][i,j]=plot.levs_linpol
                    kwargs["levs1_linpol"][i,j]=plot.levs1_linpol
                    kwargs["stokes_i_vmax"][i,j]=plot.stokes_i_vmax
                    kwargs["linpol_vmax"][i,j]=plot.linpol_vmax
                    kwargs["fracpol_vmax"][i,j]=plot.fracpol_vmax

            col=plot.col
            plt.close(plot.fig)

        elif shared_colormap=="epoch":
            col=[]
            for i in range(len(self.image_cube.dates)):
                images=image_cube.images[i,:].flatten()
                noises = image_cube.noises[i,:].flatten()

                if shared_sigma == "max":
                    index = np.argmax(noises)
                else:
                    index = np.argmin(noises)

                plot = images[index].plot(plot_mode=kwargs["plot_mode"][i, 0], im_colormap=True, im_color=kwargs["im_color"][i,0], show=False)

                for j in range(len(self.image_cube.freqs)):
                    # get levs:
                    kwargs["levs"][i, j] = plot.levs
                    kwargs["levs1"][i, j] = plot.levs1
                    kwargs["levs_linpol"][i, j] = plot.levs_linpol
                    kwargs["levs1_linpol"][i, j] = plot.levs1_linpol
                    kwargs["stokes_i_vmax"][i, j] = plot.stokes_i_vmax
                    kwargs["linpol_vmax"][i, j] = plot.linpol_vmax
                    kwargs["fracpol_vmax"][i, j] = plot.fracpol_vmax

                col.append(plot.col)
                plt.close(plot.fig)


        elif shared_colormap=="freq":
            col=[]
            for j in range(len(self.image_cube.freqs)):
                images=image_cube.images[:,j].flatten()
                noises = image_cube.noises[:,j].flatten()

                if shared_sigma == "max":
                    index = np.argmax(noises)
                else:
                    index = np.argmin(noises)

                plot = images[index].plot(plot_mode=kwargs["plot_mode"][0,j], im_colormap=True, im_color=kwargs["im_color"][0,j], show=False)

                for i in range(len(self.image_cube.dates)):
                    # get levs:
                    kwargs["levs"][i, j] = plot.levs
                    kwargs["levs1"][i, j] = plot.levs1
                    kwargs["levs_linpol"][i, j] = plot.levs_linpol
                    kwargs["levs1_linpol"][i, j] = plot.levs1_linpol
                    kwargs["stokes_i_vmax"][i, j] = plot.stokes_i_vmax
                    kwargs["linpol_vmax"][i, j] = plot.linpol_vmax
                    kwargs["fracpol_vmax"][i, j] = plot.fracpol_vmax

                col.append(plot.col)
                plt.close(plot.fig)

        elif shared_colormap=="individual":
            pass
        else:
            raise Exception("Please use valid share_colormap setting ('all','epoch','freq')")


        #create FitsImage for every image
        self.plots=np.empty((self.nrows,self.ncols),dtype=object)

        for i in range(self.nrows):
            for j in range(self.ncols):

                if swap_axis:
                    image_i=j
                    image_j=i
                else:
                    image_i=i
                    image_j=j
                if self.image_cube.images[image_i,image_j]==None:
                    #turn off the plot because no data is here
                    self.axes[i,j].axis("off")
                else:
                    self.plots[i,j]=FitsImage(image_data=self.image_cube.images[image_i,image_j],
                                        stokes_i_sigma_cut=kwargs["stokes_i_sigma_cut"][image_i,image_j],
                                        plot_mode=kwargs["plot_mode"][image_i,image_j],
                                        im_colormap=kwargs["im_colormap"][image_i,image_j],
                                        contour=kwargs["contour"][image_i,image_j],
                                        contour_color=kwargs["contour_color"][image_i,image_j],
                                        contour_cmap=kwargs["contour_cmap"][image_i,image_j],
                                        contour_alpha=kwargs["contour_alpha"][image_i,image_j],
                                        contour_width=kwargs["contour_width"][image_i,image_j],
                                        im_color=kwargs["im_color"][image_i,image_j],
                                        do_colorbar=kwargs["do_colorbar"][image_i,image_j],
                                        plot_ridgeline=kwargs["plot_ridgeline"][image_i,image_j],
                                        ridgeline_color=kwargs["ridgeline_color"][image_i,image_j],
                                        plot_counter_ridgeline=kwargs["plot_counter_ridgeline"][image_i,image_j],
                                        counter_ridgeline_color=kwargs["counter_ridgeline_color"][image_i,image_j],
                                        plot_line=kwargs["plot_line"][image_i,image_j],  # Provide two points for plotting a line
                                        line_color=kwargs["line_color"][image_i,image_j],
                                        line_width=kwargs["line_width"][image_i,image_j],  # width of the line
                                        plot_beam=kwargs["plot_beam"][image_i,image_j],
                                        overplot_gauss=kwargs["overplot_gauss"][image_i,image_j],
                                        component_color=kwargs["component_color"][image_i,image_j],
                                        plot_comp_ids=kwargs["plot_comp_ids"][image_i,image_j],
                                        overplot_clean=kwargs["overplot_clean"][image_i,image_j],
                                        plot_mask=kwargs["plot_mask"][image_i,image_j],
                                        xlim=kwargs["xlim"][image_i,image_j],
                                        ylim=kwargs["ylim"][image_i,image_j],
                                        levs=kwargs["levs"][image_i,image_j],  # predefined plot levels
                                        levs1=kwargs["levs1"][image_i,image_j],  # predefined plot levels1
                                        levs_linpol=kwargs["levs_linpol"][image_i,image_j],  # predefined linpol levs
                                        levs1_linpol=kwargs["levs1_linpol"][image_i,image_j],  # predefined linepol levs1
                                        stokes_i_vmax=kwargs["stokes_i_vmax"][image_i,image_j],  # input vmax for plot
                                        fracpol_vmax=kwargs["fracpol_vmax"][image_i,image_j],  # input vmax for plot
                                        linpol_vmax=kwargs["linpol_vmax"][image_i,image_j],  # input vmax for plot
                                        plot_evpa=kwargs["plot_evpa"][image_i,image_j],
                                        evpa_width=kwargs["evpa_width"][image_i,image_j],
                                        evpa_len=kwargs["evpa_len"][image_i,image_j],
                                        lin_pol_sigma_cut=kwargs["lin_pol_sigma_cut"][image_i,image_j],
                                        evpa_distance=kwargs["evpa_distance"][image_i,image_j],
                                        rotate_evpa=kwargs["rotate_evpa"][image_i,image_j],
                                        evpa_color=kwargs["evpa_color"][image_i,image_j],
                                        title=kwargs["title"][image_i,image_j],
                                        background_color=kwargs["background_color"][image_i,image_j],
                                        fig=self.fig,
                                        ax=self.axes[i,j],
                                        font_size_axis_title=kwargs["font_size_axis_title"][image_i,image_j],
                                        font_size_axis_tick=kwargs["font_size_axis_tick"][image_i,image_j],
                                        rcparams=kwargs["rcparams"][image_i,image_j])

        #get colorbar label:
        if shared_colorbar_label == "":
            if kwargs["plot_mode"][0, 0] == "stokes_i":
                shared_colorbar_label="Flux Density [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "lin_pol":
                shared_colorbar_label="Linear Polarization [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "frac_pol":
                shared_colorbar_label="Linear Polarization Fraction"
            elif kwargs["plot_mode"][0, 0] == "spix":
                shared_colorbar_label="Spectral Index"
            elif kwargs["plot_mode"][0, 0] == "rm":
                shared_colorbar_label="Rotation Measure [rad/m^2]"
            elif kwargs["plot_mode"][0, 0] == "residual":
                shared_colorbar_label="Residual Flux Density [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "turnover":
                shared_colorbar_label="Turnover Frequency [GHz]"
            elif kwargs["plot_mode"][0, 0] == "turnover_flux":
                shared_colorbar_label = "Turnover Flux [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "turnover_error":
                shared_colorbar_label = "Turnover Error [GHz]"
            elif kwargs["plot_mode"][0, 0] == "turnover_chisquare":
                shared_colorbar_label = r"Turnover $\chi^2$"
            else:
                shared_colorbar_label = "Flux Density [Jy/beam]"



        # check if colorbar should be plotted and is shared between plots:
        if shared_colormap == "all" and shared_colorbar:
            cbar = self.fig.colorbar(col, ax=self.axes, orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label(shared_colorbar_label,fontsize=shared_colorbar_labelsize)
        elif shared_colormap=="epoch" and shared_colorbar:
            for i in range(len(self.image_cube.dates)):
                if swap_axis:
                    cbar = self.fig.colorbar(col[i], ax=self.axes[:, i], orientation="horizontal", fraction=0.05, pad=0.05)
                else:
                    cbar = self.fig.colorbar(col[i], ax=self.axes[i, :], orientation="vertical", fraction=0.05, pad=0.1)
                cbar.set_label(shared_colorbar_label,fontsize=shared_colorbar_labelsize)
        elif shared_colormap=="freq" and shared_colorbar:
            for j in range(len(self.image_cube.freqs)):
                if swap_axis:
                    cbar = self.fig.colorbar(col[j], ax=self.axes[j, :], orientation="vertical", fraction=0.05, pad=0.05)
                else:
                    cbar = self.fig.colorbar(col[j], ax=self.axes[:, j], orientation="horizontal", fraction=0.05, pad=0.1)
                cbar.set_label(shared_colorbar_label,fontsize=shared_colorbar_labelsize)


    def export(self,name):
        if name.split(".")[-1] in ("png","jpg","jpeg","pdf","gif"):
            self.fig.savefig(name, dpi=300, bbox_inches='tight', transparent=False)
        else:
            self.fig.savefig(name+".png",dpi=300,bbox_inches="tight", transparent=False)



