import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
import pandas as pd
from sympy import Ellipse, Point, Line
import vcat.fit_functions as ff
import sys
from vcat.helpers import closest_index, get_comp_peak_rms, calculate_dist_with_err, coreshift_fit, get_resolution_limit
from scipy.interpolate import interp1d

#initialize logger
from vcat.config import logger, uvw, difmap_path, mfit_err_method, res_lim_method, H0, Om0

class Component():
    def __init__(self, x, y, maj, min, pos, flux, date, mjd, year, delta_x_est=0, delta_y_est=0,
                 component_number=-1, is_core=False, redshift=0, scale=60 * 60 * 10 ** 3,freq=15e9,noise=0,
                 beam_maj=0, beam_min=0, beam_pa=0, lin_pol=0, evpa=0, lin_pol_err=0, evpa_err=0,
                 snr=1, gain_err=0.05, error_method=mfit_err_method,res_lim_method=res_lim_method):
        self.x = x
        self.y = y
        self.mjd = mjd
        self.maj = maj
        self.min = min
        self.pos = pos
        self.flux = flux
        self.date = date
        self.year = year
        self.component_number = component_number
        self.is_core = is_core
        self.noise = noise #image noise at the position of the component
        self.beam_maj = beam_maj
        self.beam_min = beam_min
        self.beam_pa = beam_pa
        self.delta_x_est = self.x #TODO check this!
        self.delta_y_est = self.y #TODO check this!
        self.distance_to_core = np.sqrt(self.delta_x_est ** 2 + self.delta_y_est ** 2)
        self.distance_to_core_err = 0
        self.redshift = redshift
        self.freq=freq
        self.lin_pol=lin_pol
        self.evpa=evpa
        self.snr=snr
        self.gain_err=gain_err
        self.scale = scale
        self.lin_pol_err=lin_pol_err
        self.evpa_err=evpa_err

        
        def calculate_theta():
            if (self.delta_y_est > 0 and self.delta_x_est > 0) or (self.delta_y_est > 0 and self.delta_x_est < 0):
                return np.arctan(self.delta_x_est / self.delta_y_est) / np.pi * 180
            elif self.delta_y_est < 0 and self.delta_x_est > 0:
                return np.arctan(self.delta_x_est / self.delta_y_est) / np.pi * 180 + 180
            elif self.delta_y_est < 0 and self.delta_x_est < 0:
                return np.arctan(self.delta_x_est / self.delta_y_est) / np.pi * 180 - 180
            else:
                return 0

        # Calculate radius
        self.radius = np.sqrt(self.delta_x_est ** 2 + self.delta_y_est ** 2)*scale

        # Calculate theta
        self.theta = calculate_theta()

        # Calculate ratio
        self.ratio = self.min / self.maj if self.maj > 0 else 0

        self.size=self.maj*scale
        skip_tb=False
        is_circular=False
        if noise==0:
            self.res_lim_min=0
            self.res_lim_maj=0
        else:
            if self.beam_maj==0 and self.beam_min==0:
                skip_tb=True
                self.res_lim_maj=0
                self.res_lim_min=0
            #check for circular components:
            elif self.maj == self.min:
                self.res_lim_maj, dummy = get_resolution_limit(beam_maj,beam_min,beam_pa,beam_pa,snr,method=res_lim_method)
                self.res_lim_maj=self.res_lim_maj/self.scale
                self.res_lim_min=self.res_lim_maj
                is_circular=True
            else:
                self.res_lim_maj, self.res_lim_min=get_resolution_limit(beam_maj,beam_min,beam_pa,pos,snr,method=res_lim_method)
                self.res_lim_maj=self.res_lim_maj/self.scale
                self.res_lim_min=self.res_lim_min/self.scale

        #check if component is resolved or not:
        if (self.res_lim_min>self.min) or (self.res_lim_maj>self.maj):
            if is_circular:
                maj_for_tb = self.res_lim_maj
                min_for_tb = self.res_lim_maj
            else:
                maj_for_tb = np.max(np.array([self.res_lim_maj, self.maj]))
                min_for_tb = np.max(np.array([self.res_lim_min, self.min]))
            self.tb_lower_limit=True
            self.resolved=False
        else:
            self.tb_lower_limit=False
            self.resolved=True
            maj_for_tb = self.maj
            min_for_tb = self.min

        if skip_tb:
            self.tb = 0
        else:
            self.tb = 1.22e12/(self.freq*1e-9)**2 * self.flux * (1 + self.redshift) / (maj_for_tb*self.scale) / (min_for_tb*self.scale)   #Kovalev et al. 2005


        # determine errors
        self.get_errors(method=error_method,gain_err=self.gain_err)

    def __str__(self):
        line1 = f"Component with ID {self.component_number} at frequency {self.freq * 1e-9:.1f} GHz\n"
        if self.scale == 1.:
            line2 = f"x: {self.x * self.scale:.2e} deg, y:{self.y * self.scale:.2e} deg\n"
            line3 = f"Maj: {self.maj * self.scale:.2e} deg, Min: {self.min * self.scale:.2e} deg, PA: {self.pos} °\n"
            line4 = f"Flux: {self.flux*1e3} mJy, Distance to Core: {self.distance_to_core * self.scale:.2e} deg\n"
        elif self.scale == 60.:
            line2 = f"x: {self.x * self.scale:.2e} arcmin, y:{self.y * self.scale:.2e} arcmin\n"
            line3 = f"Maj: {self.maj * self.scale:.2e} arcmin, Min: {self.min * self.scale:.2e} arcmin, PA: {self.pos} °\n"
            line4 = f"Flux: {self.flux*1e3} mJy, Distance to Core: {self.distance_to_core * self.scale:.2e} arcmin\n"
        elif self.scale == 360.:
            line2 = f"x: {self.x * self.scale:.2e} arcsec, y:{self.y * self.scale:.2e} arcsec\n"
            line3 = f"Maj: {self.maj * self.scale:.2e} arcsec, Min: {self.min * self.scale:.2e} arcsec, PA: {self.pos} °\n"
            line4 = f"Flux: {self.flux*1e3} mJy, Distance to Core: {self.distance_to_core * self.scale:.2e} arcsec\n"
        else:
            line2 = f"x: {self.x * self.scale:.2f} mas, y:{self.y * self.scale:.2f} mas\n"
            line3 = f"Maj: {self.maj * self.scale:.2f} mas, Min: {self.min * self.scale:.2f} mas, PA: {self.pos} °\n"
            line4 = f"Flux: {self.flux*1e3} mJy, Distance to Core: {self.distance_to_core * self.scale:.2f} mas\n"
        
        if self.lin_pol!=0:
            line5=f"Lin Pol: {self.lin_pol*1e3} mJy ({self.lin_pol/self.flux*1e2}%), EVPA: {self.evpa}°\n"
        else:
            line5=""

        return line1 + line2 + line3 + line4 + line5

    def get_errors(self, method=mfit_err_method, gain_err=0.1):
        if method == 'flat':
            self.x_err = 0.1 * self.maj
            self.y_err = 0.1 * self.maj
            self.maj_err = 0.1 * self.maj
            self.min_err = 0.1 * self.min
            self.flux_err = 0.1 * self.flux
            self.radius_err = 0.1 * self.maj * self.scale
            self.theta_err = 20

        elif method == 'Schinzel12':

            if self.snr < 0:
                logger.debug('! Component {0} peak flux density is negative; something must have gone wrong. Set to rms level'.format(self.component_number))
                self.snr=1

            ### Calculate errors ###
            S_p = self.snr*self.noise
            if S_p ==0:
                S_p=0.00001
            sigma_p = self.noise*np.sqrt(1 + self.snr)
            SNR_p = S_p/sigma_p

            sigma_t = sigma_p*np.sqrt(1 + (self.flux**2/S_p**2))
            self.flux_err = np.sqrt(sigma_t**2 + (gain_err*self.flux)**2)    # Add gain error in
                # quadrature to reflect individual telescopes' fundamental gain uncertainty
                # this was added to the description in Schinzel et al. 2012. TODO: evaluate kepping it.
            
            ### Get resolution limit ###
            res_lim_maj, res_lim_min = self.res_lim_maj, self.res_lim_min
            
            size_maj = np.maximum(res_lim_maj, self.maj)
            size_min = np.maximum(res_lim_min, self.min)
            if self.maj > res_lim_maj:
                self.maj_err = self.maj/SNR_p
            else:
                self.maj_err = np.nan
            if self.min > res_lim_min:
                self.min_err = self.min/SNR_p
            else:
                self.min_err = np.nan
            
            ### Calculate other errors ###
            self.radius_err = np.sqrt(self.beam_maj*self.beam_min + size_maj**2*self.scale**2)/SNR_p
            if self.radius==0:
                self.theta_err=180
            else:
                self.theta_err = np.arctan(self.radius_err/self.radius) * 180/np.pi

            # NOTE: this does not take into account the covariance of the radial coordinates!
            # TODO: implement that
            self.x_err = np.sqrt(  (self.radius_err/self.scale*np.sin(self.theta/180*np.pi))**2
                                 + (self.radius/self.scale*self.theta_err/180*np.pi*np.cos(self.theta/180*np.pi))**2)
            self.y_err = np.sqrt(  (self.radius_err/self.scale*np.cos(self.theta/180*np.pi))**2
                                 + (self.radius/self.scale*self.theta_err/180*np.pi*np.sin(self.theta/180*np.pi))**2)
        elif method == 'Weaver22':

            ### Get peak flux density and rms around component position ###
            S_p = self.snr * self.noise

            if S_p < 0:
                logger.debug('! Component {0} peak flux density is negative; something must have gone wrong. Set to rms level'.format(self.component_number))
                S_p = rms

            ### Calculate errors ###
            sigma_p = self.noise*np.sqrt(1 + self.snr)
            SNR_p = S_p/sigma_p

            ### Calculate resolution limit ###
            res_lim_maj, res_lim_min = self.res_lim_maj, self.res_lim_min
            
            # TODO: check that limiting size is handled like this in Weaver et al. 2022
            size_maj = np.maximum(res_lim_maj, self.maj)
            size_min = np.maximum(res_lim_min, self.min)
            if self.maj > res_lim_maj:
                self.maj_err = self.maj/SNR_p
            else:
                self.maj_err = np.nan
            if self.min > res_lim_min:
                self.min_err = self.min/SNR_p
            else:
                self.min_err = np.nan
            
            ### Calculate other errors ###
            Tb_obs = 7.5E8*self.flux/(size_maj*size_min*self.scale**2)    # in K, adjusted from Weaver+'22 to include elliptical Gaussian components
            self.x_err = np.sqrt((1.3*1E4*Tb_obs**(-0.6))**2 + 0.005**2)/self.scale    # in deg
            self.y_err = 2*self.x_err    # in deg, taken from Weaver+'22.
                # this assumes that the beam is ~ 2 times more elongated in North-South direction. TODO: generalize this.
            self.flux_err = np.sqrt((0.09*Tb_obs**(-0.1))**2 + (self.gain_err*self.flux**2))    # in Jy, assumes gain error added in quadrature

            self.maj_err = 6.5*Tb_obs**(-0.25)/self.scale    # in deg
            self.min_err = 6.5*Tb_obs**(-0.25)/self.scale    # in deg

            self.radius_err = np.sqrt( (self.x_err*self.x/np.sqrt(self.x**2+self.y**2))**2
                                      +(self.y_err*self.y/np.sqrt(self.x**2+self.y**2))**2)*self.scale    # in mas
            self.theta_err = np.arctan(self.radius_err/self.radius)*180/np.pi    # in deg

        #regardless of the error method, calculate error for tb
        rel_flux_err=self.flux_err/self.flux

        # check if component is resolved or not:
        if (self.res_lim_min > self.min) or (self.res_lim_maj > self.maj):
            rel_maj_err = 0
            rel_min_err = 0
        else:
            rel_maj_err = self.maj_err/self.maj
            rel_min_err = self.min_err/self.min

        self.tb_err = self.tb * ((rel_flux_err) ** 2 + (rel_maj_err) ** 2 + (rel_min_err) ** 2) ** 0.5

    def set_distance_to_core(self, core_x, core_y,core_x_err=0,core_y_err=0):
        self.delta_x_est = self.x - core_x
        self.delta_y_est = self.y - core_y
        self.distance_to_core, self.distance_to_core_err = calculate_dist_with_err(self.x,self.y,core_x,core_y,
                                                                                   self.x_err,self.y_err,core_x_err,core_y_err)

    def assign_component_number(self, number):
        self.component_number = number

    def get_info(self):
        return {"x": self.x, "y": self.y, "mjd": self.mjd, "maj": self.maj, "min": self.min,
                "radius": self.radius, "theta": self.theta, "size": self.size, "ratio": self.ratio,
                "pos": self.pos, "flux": self.flux, "date": self.date,"year": self.year,
                "component_number": self.component_number, "is_core": self.is_core,
                "delta_x_est": self.delta_x_est, "delta_y_est": self.delta_y_est,
                "distance_to_core": self.distance_to_core, "redshift": self.redshift,
                "freq": self.freq, "tb": self.tb, "scale": self.scale, "lin_pol": self.lin_pol, "evpa": self.evpa}
    
    def print_errors(self):
        print(f'! Info on component {self.component_number}')
        print(f'flux: {self.flux:.5f} +/- {self.flux_err:.5f} Jy')
        if self.scale == 1.:
            unit="deg"
        elif self.scale == 60.:
            unit="arcmin"
        elif self.scale == 360.:
            unit="arcsec"
        else:
            unit="mas"

        print(f'radius: {self.radius:.3f} +/- {self.radius_err:.3f} {unit}')
        print(f'theta: {self.theta:.1f} +/- {self.theta_err:.1f} °')
        print(f'RA: {self.x*self.scale:.2f} +/- {self.x_err*self.scale:.2f} {unit}')
        print(f'Dec: {self.y*self.scale:.2f} +/- {self.y_err*self.scale:.2f} {unit}')
        print(f'Major axis: {self.maj*self.scale:.3f} +/- {self.maj_err*self.scale:.3f} {unit}')
        print(f'Minor axis: {self.min*self.scale:.3f} +/- {self.min_err*self.scale:.3f} {unit}')
        print('\n')

class ComponentCollection():
    def __init__(self, components=[], name="",date_tolerance=1,freq_tolerance=1):

        #set redshift and scale (Assumes this is the same for all components)
        if len(components) > 0:
            self.redshift = components[0].redshift
            self.scale = components[0].scale
        else:
            self.redshift = 0
            self.scale = 60*60*1000

        self.name = name

        years=np.array([])
        for comp in components:
            years=np.append(years,comp.year)

        sort_inds=np.argsort(years)
        #sort components by date
        components = np.array(components)[sort_inds]

        year_prev=0
        freqs=[]
        epochs=[]
        for comp in components:
            if not any(abs(num - comp.freq) <= freq_tolerance * 1e9 for num in freqs):
                freqs.append(comp.freq)
            if abs(comp.year-year_prev)>=date_tolerance/365.25:
                year_prev = comp.year
                epochs.append(year_prev)

        freqs=np.sort(freqs)
        epochs=np.sort(epochs)

        self.n_epochs=len(epochs)
        self.n_freqs=len(freqs)
        self.epochs_distinct=epochs
        self.freqs_distinct=freqs

        #create empty component grids
        self.components=np.empty((self.n_epochs,self.n_freqs),dtype=object)
        self.mjds = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.year = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.dist = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.dist_err = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.xs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.x_errs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.ys = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.y_errs = np.empty((self.n_epochs, self.n_freqs), dtype=float)
        self.fluxs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.fluxs_err = np.empty((self.n_epochs, self.n_freqs), dtype=float)
        self.tbs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.tbs_lower_limit= np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.tbs_err = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.resolved = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.freqs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.ids = np.empty((self.n_epochs,self.n_freqs),dtype=int)
        self.majs =  np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.majs_err =  np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.mins = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.mins_err = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.posas = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.thetas = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.thetas_err = np.empty((self.n_epochs, self.n_freqs), dtype=float)
        self.delta_x_ests = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.delta_y_ests = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.lin_pols = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.evpas = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.snrs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.lin_pols_err = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.evpas_err = np.empty((self.n_epochs,self.n_freqs),dtype=float)

        for i, year in enumerate(epochs):
            for j, freq in enumerate(freqs):
                for comp in components:
                    if comp.year-year >=0 and (comp.year-year)<=date_tolerance/365.25 and (abs(comp.freq-freq)<=freq_tolerance*1e9):
                        self.components[i,j]=comp
                        self.year[i,j]=comp.year
                        self.mjds[i,j]=comp.mjd
                        self.dist[i,j]=comp.distance_to_core * self.scale
                        self.dist_err[i,j]=comp.distance_to_core_err * self.scale
                        self.xs[i,j]=comp.x
                        self.x_errs[i,j]=comp.x_err
                        self.ys[i,j]=comp.y
                        self.y_errs[i,j]=comp.y_err
                        self.fluxs[i,j]=comp.flux
                        self.fluxs_err[i,j]=comp.flux_err
                        self.tbs[i,j]=comp.tb
                        self.tbs_err[i,j]=comp.tb_err
                        self.tbs_lower_limit[i,j]=comp.tb_lower_limit
                        self.resolved[i,j]=comp.resolved
                        self.freqs[i,j]=comp.freq
                        self.ids[i,j]=comp.component_number
                        self.majs[i,j]=comp.maj
                        self.majs_err[i,j]=comp.maj_err
                        self.mins[i,j]=comp.min
                        self.mins_err[i,j]=comp.min_err
                        self.posas[i,j]=comp.pos
                        self.thetas[i,j]=comp.theta
                        self.thetas_err[i,j]=comp.theta_err
                        self.delta_x_ests[i,j]=comp.delta_x_est
                        self.delta_y_ests[i,j]=comp.delta_y_est
                        self.lin_pols[i,j]=comp.lin_pol
                        self.evpas[i,j]=comp.evpa
                        self.snrs[i,j]=comp.snr
                        self.lin_pols_err[i,j]=comp.lin_pol_err
                        self.evpas_err[i,j]=comp.evpa_err

        try:
            self.id=np.unique(self.ids.flatten())[0]
        except:
            self.id=-1


    def __str__(self):
        line1=f"Component Collection of ID {self.id} with {len(self.year.flatten())} components.\n"
        line2=f"{len(self.ids[0,:].flatten())} Frequencies and {len(self.ids[:,0].flatten())} epochs.\n"
        return line1+line2

    def length(self):
        return len(self.components)

    def get_speed2d(self,freqs="",order=1,cosmo=FlatLambdaCDM(H0=H0, Om0=Om0),snr_cut=1,weighted_fit=True):

        #we use the one dimensional function for x and y separately
        dist=self.dist

        #do x_fit
        self.dist=self.delta_x_ests*self.scale
        x_fits=self.get_speed(freqs=freqs,order=order,cosmo=cosmo,snr_cut=snr_cut,weighted_fit=weighted_fit)

        #do y_fit
        self.dist=self.delta_y_ests*self.scale
        y_fits=self.get_speed(freqs=freqs,order=order,cosmo=cosmo,snr_cut=snr_cut,weighted_fit=weighted_fit)

        #reset dist
        self.dist=dist

        return x_fits, y_fits

    def get_speed(self,freqs="",order=1,weighted_fit=True, cosmo=FlatLambdaCDM(H0=H0, Om0=Om0),snr_cut=1,
                  t0_error_method="Gauss"):


        if freqs=="":
            freqs=self.freqs_distinct*1e-9
        elif isinstance(freqs,(float,int)):
            freqs=[freqs]
        elif not isinstance(freqs, list):
            try:
                freqs = freqs.tolist()
            except:
                raise Exception("Invalid input for 'freqs'.")

        results=[]
        for freq in freqs:
            freq_ind=closest_index(self.freqs_distinct,freq*1e9)
            year = self.year[:, freq_ind].flatten()
            dist = self.dist[:, freq_ind].flatten()
            dist_err = self.dist_err[:, freq_ind].flatten()
            snrs = self.snrs[:, freq_ind].flatten()

            year = year[snrs >= snr_cut]
            dist = dist[snrs >= snr_cut]
            dist_err = dist_err[snrs >= snr_cut]

            #check if there is enough data to perform a fit
            if len(year) > 2:

                def reduced_chi2(fit, x, y, yerr, N, n):
                    return 1. / (N - n) * np.sum(((y - fit) / yerr) ** 2.)

                t_mid = (np.min(year) + np.max(year)) / 2.
                time = np.array(year) - t_mid

                if weighted_fit:
                    try:
                        linear_fit, cov_matrix = np.polyfit(time, dist, order, cov='scaled', w=1./dist_err)
                    except:
                        logger.warning(f"Could not perform weighted fit for Component {self.name}, will do unweighted.")
                        weighted_fit=False
                if not weighted_fit:
                    linear_fit, cov_matrix = np.polyfit(time, dist, order, cov='scaled')

                #TODO check and update those parameters for order>1!!!!
                speed = linear_fit[0]
                speed_err = np.sqrt(cov_matrix[0, 0])
                y0 = linear_fit[-1] - t_mid*speed
                y0_err = np.sqrt(cov_matrix[-1, -1])
                beta_app = speed * (np.pi / (180 * self.scale * u.yr)) * (
                        cosmo.luminosity_distance(self.redshift) / (const.c.to('pc/yr') * (1 + self.redshift)))
                beta_app_err = speed_err * (np.pi / (180 * self.scale * u.yr)) * (
                        cosmo.luminosity_distance(self.redshift) / (const.c.to('pc/yr') * (1 + self.redshift)))
                d_crit = np.sqrt(1 + beta_app ** 2)
                d_crit_err = (1 + beta_app) ** (-0.5) * beta_app * beta_app_err
                dist_0_est = linear_fit[-1] - speed * t_mid
                t_0 = - linear_fit[-1] / speed + t_mid
                if t0_error_method=="Gauss":
                    t_0_err = np.sqrt((cov_matrix[-1, -1] / speed ** 2) + (linear_fit[-1] ** 2 * cov_matrix[0, 0] / speed ** 4))
                elif t0_error_method=="Rösch":
                    #see Master Thesis F. Rösch 2019 https://www.physik.uni-wuerzburg.de/fileadmin/11030400/2019/Masterarbeit_Roesch.pdf
                    sum_x = time / np.array(dist_err) ** 2
                    sum_x2 = time ** 2 / np.array(dist_err) ** 2
                    sum_err = 1. / np.array(dist_err) ** 2
                    Delta = np.sum(sum_err) * np.sum(sum_x2) - (np.sum(sum_x)) ** 2
                    t_0_err = np.sqrt(
                        (cov_matrix[-1, -1] / speed ** 2) + (linear_fit[-1] ** 2 * cov_matrix[0, 0] / speed ** 4) +
                        2 * linear_fit[-1] / speed ** 3 * np.sum(sum_x) / Delta)
                red_chi_sqr = reduced_chi2(linear_fit[0] * time + linear_fit[-1], time, dist, dist_err, len(time),
                                           len(linear_fit))

            else:
                speed = 0
                speed_err = 0
                y0 = 0
                y0_err = 0
                beta_app = 0
                beta_app_err = 0
                d_crit = 0
                d_crit_err = 0
                dist_0_est = 0
                t_0 = 0
                t_0_err = 0
                red_chi_sqr = 0
                linear_fit=0
                cov_matrix=0
                t_mid=0

            results.append({"name": self.name, "speed": float(speed), "speed_err": float(speed_err), "y0": y0, "y0_err": y0_err,
                    "beta_app": float(beta_app), "beta_app_err": float(beta_app_err), "d_crit": float(d_crit), "d_crit_err": float(d_crit_err),
                    "dist_0_est": dist_0_est, "t_0": t_0, "t_0_err": t_0_err, "red_chi_sqr": red_chi_sqr,
                    "t_mid": t_mid, "linear_fit": linear_fit, "cov_matrix": cov_matrix})

        return results

    def get_fluxes(self):
        return [comp.flux for comp in self.components]

    def get_average_component(self,freq="",epochs="",weighted=True,filter_unresolved=True,snr_cut=0):

        data=self.get_model_profile(freq=freq,epochs=epochs,filter_unresolved=filter_unresolved,snr_cut=snr_cut)

        if weighted:
            new_x = np.average(data["x"],weights=1/np.array(data["x_err"])**2)
            new_x_err = np.sqrt(1 / np.sum(1/np.array(data["x_err"])**2))
            new_y = np.average(data["y"],weights=1/np.array(data["y_err"])**2)
            new_y_err = np.sqrt(1 / np.sum(1 / np.array(data["y_err"]) ** 2))
            new_maj = np.average(data["maj"],weights=1/np.array(data["maj_err"])**2)
            new_maj_err = np.sqrt(1 / np.sum(1 / np.array(data["maj_err"]) ** 2))
            new_min = np.average(data["min"], weights=1 / np.array(data["min_err"]) ** 2)
            new_min_err = np.sqrt(1 / np.sum(1 / np.array(data["min_err"]) ** 2))
            new_flux = np.average(data["flux"],weights=1/ np.array(data["flux_err"]) ** 2)
            new_flux_err = np.sqrt(1 / np.sum(1 / np.array(data["flux_err"]) ** 2))
        else:
            new_x = np.average(data["x"])
            new_x_err = np.std(data["x"])
            new_y = np.average(data["y"])
            new_y_err = np.std(data["y"])
            new_maj = np.average(data["maj"])
            new_maj_err = np.std(data["maj"])
            new_min = np.average(data["min"])
            new_min_err = np.std(data["min"])
            new_flux = np.average(data["flux"])
            new_flux_err = np.std(data["flux"])



        new_lin_pol = np.average(data["lin_pol"])
        new_evpa = np.average(data["evpa"])
        new_pa =  np.average(data["PA"])
        new_mjd = 60000
        new_year = 2020
        new_date = "1900-01-01"

        newComp=Component(new_x/self.scale,new_y/self.scale,new_maj/self.scale,new_min/self.scale,new_pa,flux=new_flux,
                          date=new_date,mjd=new_mjd,year=new_year,lin_pol=new_lin_pol,evpa=new_evpa,scale=self.scale)

        newComp.x_err=new_x_err/self.scale
        newComp.y_err=new_y_err/self.scale
        newComp.maj_err=new_maj_err/self.scale
        newComp.min_err=new_min_err/self.scale
        newComp.flux_err=new_flux_err

        return newComp

    def get_coreshift(self, epochs="",k_r=""):

        if epochs=="":
            epochs=self.epochs_distinct
        elif isinstance(epochs,(float,int)):
            epochs=[epochs]
        elif not isinstance(epochs, list):
            try:
                epochs = epochs.tolist()
            except:
                raise Exception("Invalid input for 'epochs'.")

        results=[]
        for epoch in epochs:
            epoch_ind=closest_index(self.epochs_distinct,epoch)

            freqs=self.freqs[epoch_ind,:].flatten()
            components=self.components[epoch_ind,:].flatten()
            dist=self.dist[epoch_ind,:].flatten()
            dist_err=self.dist_err[epoch_ind,:].flatten()

            max_i=0
            max_freq=0
            for i in range(len(freqs)):
                if freqs[i]>max_freq:
                    max_i=i
                    max_freq=freqs[max_i]
            max_freq=max_freq*1e-9
            freqs = np.array(freqs)*1e-9

            #calculate core shifts:
            coreshifts=[]
            coreshift_err=[]
            for i,comp in enumerate(components):
                coreshifts.append((dist[max_i]-dist[i])*1e3)#in uas
                coreshift_err.append(np.sqrt(dist_err[max_i]**2+dist_err[i]**2)*1e3)
            result=coreshift_fit(freqs,coreshifts,coreshift_err,max_freq,k_r=k_r)
            results.append(result)

        return results

    def fit_comp_spectrum(self,epochs="",fluxerr=False,fit_free_ssa=False):
        """
        This function only makes sense on a component collection with multiple components on the same date at different frequencies

        Inputs:
            fluxerr: Fractional Errors (dictionary with {'error': [], 'freq':[]})
        """

        if not isinstance(epochs, list) and epochs=="":
            epochs=self.epochs_distinct
        elif isinstance(epochs,(float,int)):
            epochs=[epochs]
        elif not isinstance(epochs, list):
            try:
                epochs = epochs.tolist()
            except:
                raise Exception("Invalid input for 'epochs'.")

        results=[]
        for epoch in epochs:
            logger.info(epoch)
            epoch_ind=closest_index(self.epochs_distinct,epoch)

            fluxs=self.fluxs[epoch_ind,:].flatten()
            freqs=self.freqs[epoch_ind,:].flatten()
            ids=self.ids[epoch_ind,:].flatten()

            logger.info("Fit component spectrum\n")

            cflux = np.array(fluxs)
            if fluxerr:
                cfluxerr = fluxerr['error']*cflux.copy()
                cfreq = fluxerr['freq']
                cfluxerr = fluxerr['error']
            else:
                cfluxerr = 0.15*cflux.copy() #default of 15% error
                cfreq = np.array(freqs)*1e-9 #convert to GHz

            cid = ids

            logger.info("Fit Powerlaw to Comp" + str(cid[0]))
            pl_x0 = np.array([np.mean(cflux),-1])
            pl_p,pl_sd,pl_ch2,pl_out = ff.odr_fit(ff.powerlaw,[cfreq,cflux,cfluxerr],pl_x0,verbose=1)

            #fit Snu
            logger.info("Fit SSA to Comp " + str(cid[0]))
            if fit_free_ssa:
                sn_x0 = np.array([20,np.max(cflux),2.5,-1])
                sn_p,sn_sd,sn_ch2,sn_out = ff.odr_fit(ff.Snu,[cfreq,cflux,cfluxerr],sn_x0,verbose=1)
            else:
                sn_x0 = np.array([20,np.max(cflux),-1])
                sn_p,sn_sd,sn_ch2,sn_out = ff.odr_fit(ff.Snu_real,[cfreq,cflux,cfluxerr],sn_x0,verbose=1)

            if np.logical_and(sn_ch2>pl_ch2,pl_out.info<5):
               logger.info("Power law fits better\n")
               CompPL = cid[0]
               alpha = pl_p[1]
               alphaE = pl_sd[1]
               chi2PL = pl_ch2
               fit = "PL"
            elif np.logical_and(pl_ch2>sn_ch2,sn_out.info<5):
                logger.info('ssa spectrum fits better\n')
                CompSN = cid[0]
                num = sn_p[0]
                Sm = sn_p[1]
                chi2SN = sn_ch2
                SmE = sn_sd[1]
                numE = sn_sd[0]
                fit = "SN"
                if fit_free_ssa:
                    athin = sn_p[3]
                    athinE = sn_sd[3]
                    athick = sn_p[2]
                    athickE = sn_sd[2]
                else:
                    athin = sn_p[2]
                    athinE = sn_sd[2]
                    athick = 2.5
                    athickE = 0.0

            else:
                logger.info('NO FIT WORKED, use power law\n')
                CompPL = cid[0]
                alpha = pl_p[1]
                alphaE = pl_sd[1]
                chi2PL = pl_ch2
                fit = "PL"

            #return fit results
            if fit=="PL":
                results.append({"fit":"PL","alpha":alpha,"alphaE":alphaE,"chi2":chi2PL,"pl_p":pl_p,"pl_sd":pl_sd})

            if fit=="SN":
                results.append({"fit":"SN","athin":athin,"athinE":athinE,
                    "athick":athick,"athickE":athickE,"num":num,"Sm":Sm,
                    "chi2":chi2SN,"SmE":SmE,"numE":numE,"fit_free_ssa":fit_free_ssa,
                    "sn_p":sn_p,"sn_sd":sn_sd})

        return results

    def get_model_profile(self,freq="",epochs="",core_position=[0,0],filter_unresolved=False,snr_cut=1):

        # read in settings
        if freq == "":
            freq = self.freqs_distinct*1e-9
        elif isinstance(freq, (int, float)):
            freq = np.array([freq])
        elif isinstance(freq, list):
            freq = np.array(freq)
        else:
            raise Exception("Invalid input for 'freq'.")

        if epochs == "":
            epochs = self.epochs_distinct
        elif isinstance(epochs, (float, int)):
            epochs = [epochs]
        elif not isinstance(epochs, list):
            try:
                epochs = epochs.tolist()
            except:
                raise Exception("Invalid input for 'epochs'.")

        # export comp info
        majs = []
        majs_err = []
        mins = []
        mins_err = []
        fluxs = []
        thetas = []
        thetas_err = []
        tbs = []
        tbs_lower_limit = []
        dists = []
        dist_errs = []
        fluxs_err = []
        xs = []
        x_errs = []
        ys = []
        y_errs = []
        pas = []
        lin_pols = []
        lin_pols_err = []
        evpas = []
        evpas_err = []

        for i, freq in enumerate(freq):
            for j, epoch in enumerate(epochs):
                ind_f = closest_index(self.freqs_distinct, freq * 1e9)
                ind_e = closest_index(self.epochs_distinct, epoch)
                if not filter_unresolved or (filter_unresolved and self.resolved[ind_e, ind_f]):
                    if self.snrs[ind_e,ind_f] >= snr_cut:
                        majs.append(self.majs[ind_e, ind_f]*self.scale)
                        majs_err.append(self.majs_err[ind_e,ind_f]*self.scale)
                        mins.append(self.majs[ind_e, ind_f]*self.scale)
                        mins_err.append(self.mins_err[ind_e,ind_f]*self.scale)
                        fluxs.append(self.fluxs[ind_e, ind_f])
                        fluxs_err.append(self.fluxs_err[ind_e,ind_f])
                        tbs.append(self.tbs[ind_e, ind_f])
                        tbs_lower_limit.append(self.tbs_lower_limit[ind_e, ind_f])
                        xs.append(self.xs[ind_e,ind_f]*self.scale)
                        x_errs.append(self.x_errs[ind_e,ind_f]*self.scale)
                        ys.append(self.ys[ind_e,ind_f]*self.scale)
                        y_errs.append(self.y_errs[ind_e,ind_f]*self.scale)
                        pas.append(self.posas[ind_e,ind_f])
                        thetas.append(self.thetas[ind_e,ind_f])
                        thetas_err.append(self.thetas_err[ind_e, ind_f])
                        lin_pols.append(self.lin_pols[ind_e,ind_f])
                        lin_pols_err.append(self.lin_pols_err[ind_e,ind_f])
                        evpas.append(self.evpas[ind_e,ind_f])
                        evpas_err.append(self.evpas_err[ind_e,ind_f])
                        dist,dist_err=calculate_dist_with_err(self.xs[ind_e,ind_f]*self.scale,self.ys[ind_e,ind_f]*self.scale,
                                                              core_position[0],core_position[1],self.x_errs[ind_e,ind_f]*self.scale,
                                                              self.y_errs[ind_e,ind_f]*self.scale,0,0)
                        dists.append(dist)
                        dist_errs.append(dist_err)

        return {"maj": majs,"maj_err": majs_err, "min": mins, "min_err": mins_err, "flux": fluxs, "flux_err": fluxs_err,
                "tb": tbs, "tb_lower_limit":tbs_lower_limit,"dist":dists,"dist_err":dist_errs,
                "x": xs, "x_err":x_errs,"theta":thetas, "theta_err": thetas_err,"y":ys,"y_err":y_errs,"PA": pas,
                "lin_pol": lin_pols,"lin_pol_err": lin_pols_err, "evpa": evpas,"evpa_err": evpas_err}

    def interpolate(self, mjd, freq):
        freq_ind=closest_index(self.freqs_distinct,freq*1e9)

        #obtain values to interpolate
        mjds=self.mjds[:,freq_ind].flatten()
        if mjd<np.min(mjds) or mjd>np.max(mjds):
            return None
        year=interp1d(mjds,self.year[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        maj=interp1d(mjds,self.majs[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        min=interp1d(mjds,self.mins[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        pos=interp1d(mjds,self.posas[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        x=interp1d(mjds,self.xs[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        y=interp1d(mjds,self.ys[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        flux=interp1d(mjds,self.fluxs[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        delta_x_est=interp1d(mjds,self.delta_x_ests[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)
        delta_y_est=interp1d(mjds,self.delta_y_ests[:,freq_ind].flatten(),kind="linear",fill_value="extrapolate")(mjd)


        return Component(x, y, maj, min, pos, flux, "", mjd, year, delta_x_est=delta_x_est, delta_y_est=delta_y_est,
                         component_number=self.components[:,freq_ind].flatten()[0].component_number,
                         is_core=self.components[:,freq_ind].flatten()[0].is_core, redshift=0,
                         scale=self.components[:,freq_ind].flatten()[0].scale,freq=self.freqs_distinct[freq_ind],noise=0,
                         beam_maj=self.components[:,freq_ind].flatten()[0].beam_maj, beam_min=self.components[:,freq_ind].flatten()[0].beam_min,
                         beam_pa=self.components[:,freq_ind].flatten()[0].beam_pa)
