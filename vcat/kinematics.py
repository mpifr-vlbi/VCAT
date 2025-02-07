import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
import pandas as pd
from sympy import Ellipse, Point, Line


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

        is_circular=False
        if noise==0:
            self.res_lim_min=0
            self.res_lim_maj=0
        else:
            #check for circular components:
            if self.maj == self.min:
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

        self.tb = 1.22e12/(self.freq*1e-9)**2 * self.flux * (1 + self.redshift) / maj_for_tb / min_for_tb   #Kovalev et al. 2005
        self.scale = scale

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
    def __init__(self, components=[], name=""):
        self.components = components
        if len(components) > 0:
            self.redshift = components[0].redshift
            self.scale = components[0].scale
        else:
            self.redshift = 0
            self.scale = 1
        self.name = name
        self.mjds = []
        self.year = []
        self.dist = []
        self.dist_err = []
        self.time = []
        self.xs = []
        self.ys = []
        self.fluxs = []
        self.tbs = []
        self.tbs_lower_limit= []

        for comp in components:
            self.year.append(comp.year)
            self.dist.append(comp.distance_to_core * self.scale)
            self.dist_err.append(comp.maj / 2 * self.scale)
            self.xs.append(comp.delta_x_est)
            self.ys.append(comp.delta_y_est)
            self.fluxs.append(comp.flux)
            self.tbs.append(comp.tb)
            self.tbs_lower_limit.append(comp.tb_lower_limit)

    def length(self):
        return len(self.components)

    def get_speed(self):
        if self.length() > 2:
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

            def reduced_chi2(fit, x, y, yerr, N, n):
                return 1. / (N - n) * np.sum(((y - fit) / yerr) ** 2.)

            t_mid = (np.min(self.year) + np.max(self.year)) / 2.
            time = np.array(self.year) - t_mid

            linear_fit, cov_matrix = np.polyfit(time, self.dist, 1, cov='scaled')  # ,w=1./dist_err)

            speed = linear_fit[0]
            speed_err = np.sqrt(cov_matrix[0, 0])
            y0 = linear_fit[1] - t_mid*speed
            y0_err = np.sqrt(cov_matrix[1, 1])
            beta_app = speed * (np.pi / (180 * self.scale * u.yr)) * (
                    cosmo.luminosity_distance(self.redshift) / (const.c.to('pc/yr') * (1 + self.redshift)))
            beta_app_err = speed_err * (np.pi / (180 * self.scale * u.yr)) * (
                    cosmo.luminosity_distance(self.redshift) / (const.c.to('pc/yr') * (1 + self.redshift)))
            d_crit = np.sqrt(1 + beta_app ** 2)
            d_crit_err = (1 + beta_app) ** (-0.5) * beta_app * beta_app_err
            dist_0_est = linear_fit[1] - speed * t_mid
            t_0 = - linear_fit[1] / speed + t_mid
            sum_x = time / np.array(self.dist_err) ** 2
            sum_x2 = time ** 2 / np.array(self.dist_err) ** 2
            sum_err = 1. / np.array(self.dist_err) ** 2
            Delta = np.sum(sum_err) * np.sum(sum_x2) - (np.sum(sum_x)) ** 2
            t_0_err = np.sqrt((cov_matrix[1, 1] / speed ** 2) + (linear_fit[1] ** 2 * cov_matrix[0, 0] / speed ** 4) +
                              2 * linear_fit[1] / speed ** 3 * np.sum(sum_x) / Delta)
            red_chi_sqr = reduced_chi2(linear_fit[0] * time + linear_fit[1], time, self.dist, self.dist_err, len(time),
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

        return {"name": self.name, "speed": float(speed), "speed_err": float(speed_err), "y0": y0, "y0_err": y0_err,
                "beta_app": float(beta_app), "beta_app_err": float(beta_app_err), "d_crit": float(d_crit), "d_crit_err": float(d_crit_err),
                "dist_0_est": dist_0_est, "t_0": t_0, "t_0_err": t_0_err, "red_chi_sqr": red_chi_sqr}

def get_resolution_limit(beam_maj,beam_min,beam_pos,comp_pos,flux,noise):
    # TODO check the resolution limits, if they make sense and are reasonable (it looks okay though...)!!!!
    #here we need to check if the component is resolved or not!
    factor=np.sqrt(4*np.log(2)/np.pi*np.log(flux/noise/(flux/noise-1))) #following Kovalev et al. 2005

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
