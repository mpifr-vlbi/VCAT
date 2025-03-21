import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
import pandas as pd
from sympy import Ellipse, Point, Line
import vcat.VLBI_map_analysis.modules.fit_functions as ff
import sys
from scipy.optimize import curve_fit
from vcat.helpers import closest_index
from scipy.interpolate import interp1d


class Component():
    def __init__(self, x, y, maj, min, pos, flux, date, mjd, year, delta_x_est=0, delta_y_est=0,
                 component_number=-1, is_core=False, redshift=0, scale=60 * 60 * 10 ** 3,freq=15e9,noise=0,
                 beam_maj=0, beam_min=0, beam_pa=0):
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
        self.redshift = redshift
        self.freq=freq

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
        self.radius = np.sqrt(self.delta_x_est ** 2 + self.delta_y_est ** 2) * scale

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
            if self.maj==0 and self.min==0:
                skip_tb=True
                self.res_lim_maj=0
                self.res_lim_min=0
            #check for circular components:
            elif self.maj == self.min:
                self.res_lim_maj, dummy = get_resolution_limit(beam_maj,beam_min,beam_pa,beam_pa,flux,noise)
                self.res_lim_min=self.res_lim_maj
                is_circular=True
            else:
                self.res_lim_maj, self.res_lim_min=get_resolution_limit(beam_maj,beam_min,beam_pa,pos,flux,noise) #Kovalev et al. 2005

        #check if component is resolved or not:
        if (self.res_lim_min>self.min) or (self.res_lim_maj>self.maj):
            if is_circular:
                maj_for_tb = self.res_lim_maj
                min_for_tb = self.res_lim_maj
            else:
                maj_for_tb = np.max(np.array([self.res_lim_maj, self.maj]))
                min_for_tb = np.max(np.array([self.res_lim_min, self.min]))
            self.tb_lower_limit=True
        else:
            self.tb_lower_limit=False
            maj_for_tb = self.maj
            min_for_tb = self.min

        maj_for_tb=np.max(np.array([self.res_lim_maj,self.maj]))
        min_for_tb=np.max(np.array([self.res_lim_min,self.min]))

        if skip_tb:
            self.tb = 0
        else:
            self.tb = 1.22e12/(self.freq*1e-9)**2 * self.flux * (1 + self.redshift) / maj_for_tb / min_for_tb   #Kovalev et al. 2005
        self.scale = scale

    def __str__(self):
        line1=f"Component with ID {self.component_number} at frequency {self.freq*1e-9:.1f} GHz\n"
        line2=f"x: {self.x*self.scale:.2f}mas, y:{self.y*self.scale:.2f}mas\n"
        line3=f"Maj: {self.maj*self.scale:.2f}mas, Min: {self.min*self.scale:.2f}, PA: {self.pos}°\n"
        line4=f"Flux: {self.flux} Jy, Distance to Core: {self.distance_to_core*self.scale:.2f} mas\n"

        return line1+line2+line3+line4

    def set_distance_to_core(self, core_x, core_y):
        self.delta_x_est = self.x - core_x
        self.delta_y_est = self.y - core_y
        self.distance_to_core = np.sqrt(self.delta_x_est ** 2 + self.delta_y_est ** 2)

    def assign_component_number(self, number):
        self.component_number = number

    def get_info(self):
        return {"x": self.x, "y": self.y, "mjd": self.mjd, "maj": self.maj, "min": self.min,
                "radius": self.radius, "theta": self.theta, "size": self.size, "ratio": self.ratio,
                "pos": self.pos, "flux": self.flux, "date": self.date,"year": self.year,
                "component_number": self.component_number, "is_core": self.is_core,
                "delta_x_est": self.delta_x_est, "delta_y_est": self.delta_y_est,
                "distance_to_core": self.distance_to_core, "redshift": self.redshift,
                "freq": self.freq, "tb": self.tb, "scale": self.scale}

class ComponentCollection():
    def __init__(self, components=[], name="",date_tolerance=1,freq_tolerance=1):

        #set redshift and scale (Assumes this is the same for all components)
        if len(components) > 0:
            self.redshift = components[0].redshift
            self.scale = components[0].scale
        else:
            self.redshift = 0
            self.scale = 1

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
        self.ys = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.fluxs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.tbs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.tbs_lower_limit= np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.freqs = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.ids = np.empty((self.n_epochs,self.n_freqs),dtype=int)
        self.majs =  np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.mins = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.posas = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.delta_x_ests = np.empty((self.n_epochs,self.n_freqs),dtype=float)
        self.delta_y_ests = np.empty((self.n_epochs,self.n_freqs),dtype=float)

        for i, year in enumerate(epochs):
            for j, freq in enumerate(freqs):
                for comp in components:
                    if comp.year-year >=0 and (comp.year-year)<=date_tolerance/365.25 and (abs(comp.freq-freq)<=freq_tolerance*1e9):
                        self.components[i,j]=comp
                        self.year[i,j]=comp.year
                        self.mjds[i,j]=comp.mjd
                        self.dist[i,j]=comp.distance_to_core * self.scale
                        self.dist_err[i,j]=comp.maj*0.1 * self.scale #TODO fix this!
                        self.xs[i,j]=comp.delta_x_est
                        self.ys[i,j]=comp.delta_y_est
                        self.fluxs[i,j]=comp.flux
                        self.tbs[i,j]=comp.tb
                        self.tbs_lower_limit[i,j]=comp.tb_lower_limit
                        self.freqs[i,j]=comp.freq
                        self.ids[i,j]=comp.component_number
                        self.majs[i,j]=comp.maj
                        self.mins[i,j]=comp.min
                        self.posas[i,j]=comp.pos
                        self.delta_x_ests[i,j]=comp.delta_x_est
                        self.delta_y_ests[i,j]=comp.delta_y_est

        if len(np.unique(self.ids.flatten()))>1:
            numbers=np.unique(self.ids.flatten())
            raise Exception(f"Used components with different ID numbers ({numbers}) as component collection!")
        elif len(np.unique(self.ids.flatten()))==1:
            self.id=np.unique(self.ids.flatten())[0]
        else:
            self.id=-1


    def __str__(self):
        line1=f"Component Collection of ID {self.id} with {len(self.year.flatten())} components.\n"
        line2=f"{len(self.ids[0,:].flatten())} Frequencies and {len(self.ids[:,0].flatten())} epochs.\n"
        return line1+line2

    def length(self):
        return len(self.components)

    def get_speed2d(self,freqs="",order=1,cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):

        #we use the one dimensional function for x and y separately
        dist=self.dist

        #do x_fit
        self.dist=self.delta_x_ests*self.scale
        x_fits=self.get_speed(freqs=freqs,order=order,cosmo=cosmo)

        #do y_fit
        self.dist=self.delta_y_ests*self.scale
        y_fits=self.get_speed(freqs=freqs,order=order,cosmo=cosmo)

        #reset dist
        self.dist=dist

        return x_fits, y_fits

    def get_speed(self,freqs="",order=1,weighted_fit=False, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):


        if freqs=="":
            freqs=self.freqs_distinct
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

            #check if there is enough data to perform a fit
            if len(self.year[:,freq_ind].flatten()) > 2:

                year=self.year[:,freq_ind].flatten()
                dist=self.dist[:,freq_ind].flatten()
                dist_err=self.dist_err[:,freq_ind].flatten()

                def reduced_chi2(fit, x, y, yerr, N, n):
                    return 1. / (N - n) * np.sum(((y - fit) / yerr) ** 2.)

                t_mid = (np.min(year) + np.max(year)) / 2.
                time = np.array(year) - t_mid

                if weighted_fit:
                    linear_fit, cov_matrix = np.polyfit(time, dist, order, cov='scaled', w=1./dist_err)
                else:
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
                sum_x = time / np.array(dist_err) ** 2
                sum_x2 = time ** 2 / np.array(dist_err) ** 2
                sum_err = 1. / np.array(dist_err) ** 2
                Delta = np.sum(sum_err) * np.sum(sum_x2) - (np.sum(sum_x)) ** 2
                t_0_err = np.sqrt((cov_matrix[-1, -1] / speed ** 2) + (linear_fit[-1] ** 2 * cov_matrix[0, 0] / speed ** 4) +
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

    def get_coreshift(self, epochs=""):

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
            dist_err=self.dist[epoch_ind,:].flatten()

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

            #define core shift function (Lobanov 1998)
            def delta_r(nu,k_r,r0,ref_freq):
                return r0*((nu/ref_freq)**(-1/k_r)-1)

            params, covariance = curve_fit(lambda nu, k_r, r0: delta_r(nu,k_r,r0,max_freq),freqs,coreshifts,p0=[1,1],sigma=coreshift_err)

            k_r_fitted, r0_fitted = params

            print(f"Fitted k_r: {k_r_fitted}")
            print(f"Fitted r0: {r0_fitted}")

            results.append({"k_r":k_r_fitted,"r0":r0_fitted,"ref_freq":max_freq,"freqs":freqs,"coreshifts":coreshifts,"coreshift_err":coreshift_err})

        return results

    def fit_comp_spectrum(self,epochs="",add_data=False,plot_areas=False,plot_all_components=False,comps=False,
            exclude_comps=False,ccolor=False,out=True,fluxerr=False,fit_free_ssa=False,plot_fit_summary=False,
            annotate_fit_results=True):
        """
        This function only makes sense on a component collection with multiple components on the same date at different frequencies
        Inputs:
            fluxerr: Fractional Errors (dictionary with {'error': [], 'freq':[]})
        """

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
            print(epoch)
            epoch_ind=closest_index(self.epochs_distinct,epoch)

            fluxs=self.fluxs[epoch_ind,:].flatten()
            freqs=self.freqs[epoch_ind,:].flatten()
            ids=self.ids[epoch_ind,:].flatten()

            sys.stdout.write("Fit component spectrum\n")

            cflux = np.array(fluxs)
            if fluxerr:
                cfluxerr = fluxerr['error']*cflux.copy()
                cfreq = fluxerr['freq']
                cfluxerr = fluxerr['error']
            else:
                cfluxerr = 0.15*cflux.copy() #default of 15% error
                cfreq = np.array(freqs)*1e-9 #convert to GHz

            cid = ids

            print("Fit Powerlaw to Comp" + str(cid[0]))
            pl_x0 = np.array([np.mean(cflux),-1])
            pl_p,pl_sd,pl_ch2,pl_out = ff.odr_fit(ff.powerlaw,[cfreq,cflux,cfluxerr],pl_x0,verbose=1)

            #fit Snu
            print("Fit SSA to Comp " + str(cid[0]))
            if fit_free_ssa:
                sn_x0 = np.array([120,np.max(cflux),2.5,-3])
                beta,sd_beta,chi2,sn_out = ff.odr_fit(ff.Snu,[cfreq,cflux,cfluxerr],sn_x0,verbose=1)
            else:
                sn_x0 = np.array([20,np.max(cflux),-1])
                sn_p,sn_sd,sn_ch2,sn_out = ff.odr_fit(ff.Snu_real,[cfreq,cflux,cfluxerr],sn_x0,verbose=1)

            if np.logical_and(sn_ch2>pl_ch2,pl_out.info<5):
               sys.stdout.write("Power law fits better\n")
               CompPL = cid[0]
               alpha = pl_p[1]
               alphaE = pl_sd[1]
               chi2PL = pl_ch2
               fit = "PL"
            elif np.logical_and(pl_ch2>sn_ch2,sn_out.info<5):
                sys.stdout.write('ssa spectrum fits better\n')
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
                sys.stdout.write('NO FIT WORKED, use power law\n')
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


def get_resolution_limit(beam_maj,beam_min,beam_pos,comp_pos,flux,noise):
    # TODO check the resolution limits, if they make sense and are reasonable (it looks okay though...)!!!!
    #here we need to check if the component is resolved or not!
    factor=np.sqrt(4*np.log(2)/np.pi*np.log(abs(flux/noise)/(abs(flux/noise)-1))) #following Kovalev et al. 2005

    #rotate the beam to the x-axis
    new_pos=beam_pos-comp_pos

    #TODO double check the angles and check that new_pos and pos are both in degree!
    #We use SymPy to intersect the beam with the component maj/min directions
    beam=Ellipse(Point(0,0),hradius=beam_maj/2,vradius=beam_min/2)
    line_maj=Line(Point(0,0),Point(np.cos(new_pos/180*np.pi),np.sin(new_pos/180*np.pi)))
    line_min=Line(Point(0,0),Point(np.cos((new_pos+90)/180*np.pi),np.sin((new_pos+90)/180*np.pi)))
    p1,p2=beam.intersect(line_maj)
    b_phi_maj=float(p1.distance(p2)) #as in Kovalev et al. 2005
    p1,p2=beam.intersect(line_min)
    b_phi_min=float(p1.distance(p2)) #as in Kovalev et al. 2005
    theta_min = b_phi_min*factor
    theta_maj = b_phi_maj*factor
    return theta_maj,theta_min
