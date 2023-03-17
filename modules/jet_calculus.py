from astropy.io import fits
import astropy.constants as const
import numpy as np
from scipy.ndimage import fourier_shift
from itertools import groupby
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

#for drawing ellipses/circles
import math
from skimage.draw import (circle_perimeter,ellipse,ellipse_perimeter)
def keyfunc(s):
    return [int(''.join(g)) if k else ''.join(g) for k, g in groupby(s, str.isdigit)]

def read_fits(file):
    with fits.open(file) as hdulist:
        comps   = hdulist[1].data
        header  = hdulist[0].header
        img     = hdulist[0].data
        img         = img.reshape(img.shape[2],img.shape[3])
        #img            = img[0][0]
    return comps,header,img

def read_header(file):
    with fits.open(file) as hdulist:
        header  = hdulist[0].header
    return header

def px_to_mas(data,head):
    #convert pixel to mas
    xx,yy=data
    mas=3.6e6
    return round((xx - head['crpix1'])*head['cdelt2']*mas,2),round((yy - head['crpix2'])*head['cdelt2']*mas,2)

def PXPERBEAM(b_maj,b_min,px_inc):
    beam_area = np.pi/(4*np.log(2))*b_min*b_maj
    PXPERBEAM = beam_area/(px_inc**2)
    return PXPERBEAM
#

def JyPerBeam2Jy(jpb,b_maj,b_min,px_inc):
    return jpb/PXPERBEAM(b_maj,b_min,px_inc)

def axis_along_pa(major,minor,ellang,pa):
    '''
    Evaluates the axis of the beam along a given angle in polar coordinates.
    '''
    t =  (pa - ellang)*np.pi/180
    return (major*minor)/np.sqrt((minor*np.cos(t))**2+(major*np.sin(t))**2)
#
def mod_peak_flux(S,amaj,amin,bmaj,bmin):
    '''
    A way to derive the Model component peak flux
    '''
    return S/np.sqrt((amaj**2+bmaj**2)*(amin**2+bmin**2))*(bmaj*bmin)

def resLim(amaj,amin,snr,beta):
    return 2.**(2-(beta/2.))/np.pi*np.sqrt(np.pi*amaj*amin*np.log(2)*np.log(snr/(snr-1)))

def derive_positional_error(beam,compsize,snr_c,snr_m):
    '''
    resolution depend on beam size and component size as well as clean snr and model component snr.
    snr_c: S/N of map peak to post-fit rms noise
    snr_m: S/N of component peak flux density to post-fit rms noise
    '''
    return np.sqrt(np.square(beam/2./snr_c)+np.square(compsize/2./snr_m))
#
def derive_comp_width_error(W,snr):
    '''
    W: component FWHM
    snr: S/N of component peak flux density to post-fit rms noise
    '''
    return W/snr
#
def derive_tb(S,f,z,major,ratio=1,**kwargs):
    '''
    Derives brightness temperature (see Kovalev et al. 2005.
    S           : Flux density of component/Region [Jy]
    f           : Observing Frequency [GHz]
    major : Major axis ofcomponent/region[mas]
    minor : Minor axis of component/region[mas]
    '''
#   Const = 2*np.log(2)*np.square(const.c)*1e-26/(np.pi*.k*np.square(np.pi/180/3.6e6))
    Const = 1.22e12
#   tb = Const*S*(1+z)/(major**2*ratio*np.square(f*1e9))
    tb = Const*S*(1+z)/(major**2*ratio*f**2)

    if 'Serr' in kwargs.keys():
        Serr = kwargs.get('Serr')
        if 'majorerr' in kwargs.keys():
            majorerr = kwargs.get('majorerr')
        else:
            raise Exception('Please give a value for majorerr as well as Serr to derive error for tb.\n')
        dtb = Const*(1+z)/(ratio*np.square(f*1e9))*np.sqrt(np.square(Serr/np.square(major))+np.square(2*S*majorerr/major**3))
        print('derived error as well.\n')
        return (tb,dtb)
    else:
        return tb

def Bssa(f,Tb,z):
    return 4.57e19*f/Tb**2*(1+z)

def draw_circ(img,x,y,radius):
    rr,cc = circle_perimeter(x,y,radius)
    img[rr,cc] = 1
    return img

def draw_ell(img,x,y,rmin,rmaj,posang):
    #returns rr,cc: index px that belong to the ellipse perimeter.
    rr,cc =ellipse_perimeter(x,y,rmin,rmaj,orientation=posang) #(x,y) center coord., posang major axis orientation clockwise [rad]
    img[rr,cc] = 1
    return img
#

def apply_shift(img,shift):
    offset_image = fourier_shift(np.fft.fftn(img), shift)
    imgalign = np.fft.ifftn(offset_image)
    img2 = imgalign.real

    return img2

def Rs(m,return_pc=True):
    R   = 2*const.G*m*const.M_sun/const.c**2
    if return_pc:
        R=R.to(u.parsec)
    return R

def mastopc(z=None,d=None):
    cosmo=FlatLambdaCDM(H0=71,Om0=0.27) #Komatsu+09
    if d:
        D=d*1e6*u.parsec
    else:
        D   = cosmo.angular_diameter_distance(z)
    return (D*np.pi/180/3.6e6).to(u.parsec)
#   return 1/(60*60*1e3)*D

def resinRs(z,f,M):
    mtp = mastopc(z)
    return const.c/f*1e9*u.s/8600e3/u.m*180/np.pi*3.5e6*mtp*u.pc/((2*const.G*10**(M)*const.M_sun/const.c**2).to(u.pc))/2.

def mas2Rs(x,M=10**8.2,z=0.005037,D=False):
    rs =Rs(M)
    if D:
        mtp = mastopc(d=D)
    else:
        mtp=mastopc(z)
    return x*mtp/rs
def Rs2mas(x,M=10**8.2,z=0.005037,D=False):
    rs =Rs(M)
    if D:
        mtp = mastopc(d=D)
    else:
        mtp=mastopc(z)
    return x*rs/mtp

def app_ang(width,z,sd=False):
    ang = 2*np.arctan(width/(2*z))*180/np.pi
    if len(sd)==1:
        sw = sd[0]
        angErr = 2*180/np.pi*(1/(1+(width/(2*z))**2)*sw/(2*z))
    elif len(sd)==2:
        (sw,sz) = sd
        angErr = 2*180/np.pi*((1/(1+(width/(2*z))**2)*sw/(2*z))-((1/(width/(2*z))**2)*width/(2*z**2)*sz))
    else:
        angErr = False
    #print(angErr)
    return ang,angErr

############
#Code below to derive enclosing ellipse taken from isisscripts.sl
def r_e(a,b,p,t,y=1.):
    #distance from the origin to the point on ellipse at phase t
    return np.sqrt(np.cos(t)**2*a**2*(np.cos(p)**2+y**2*np.sin(p)**2)+np.sin(t)**2*b**2*(y**2*np.cos(p)**2+np.sin(p)**2)+np.sin(t)*np.cos(t)*2*a*b*(y**2-1)*np.cos(p)*np.sin(p))

def t_m(a,b,p,y):
    # phase parameter of ellipse where r_e is maximal (minimal?)
    return 0.5*np.arctan( 2*a*b*np.cos(p)*np.sin(p)*(y**2-1.)/(a**2*(np.cos(p)**2+y**2*np.sin(p)**2)-b**2*(y**2*np.cos(p)**2+np.sin(p)**2)))

def enclosing_ellipse(beam1,beam2):
    bmaj1,bmin1,bpa1 = beam1
    bmaj2,bmin2,bpa2 = beam2

    if bmaj1 <= bmin2:
        return (bmaj2,bmin2,bpa2)
    if bmaj2 <= bmin1:
        return (bmaj1,bmin1,bpa1)
    if np.logical_and.reduce((bmaj1==bmaj2,bmin1==bmin2,((bpa1-bpa2) % np.pi) == 0)):
        return (bmaj1,bmin1,bpa1)

    if bmaj2 < bmaj1:
        tmp     = bmaj2
        bmaj2 = bmaj1
        bmaj1   = tmp
        tmp     = bmin2
        bmin2 = bmin1
        bmin1   = tmp
        tmp     = bpa2
        bpa2 = bpa1
        bpa1    = tmp

    dpa = bpa1

    bpa1 -= dpa
    bpa2 -= dpa

    Y = bmaj1/bmin1
    A = r_e(bmaj2,bmin2,bpa2,t_m(bmaj2,bmin2,bpa2,Y),y=Y)
    B = max([r_e(bmaj2,bmin2,bpa2,t_m(bmaj2,bmin2,bpa2,Y)+0.5*np.pi,y=Y),bmaj1])

  #(mx,my)= ellipse (bmaj2,bmin2,bpa2,t_m(bmaj2,bmin2,bpa2,Y);y=Y)
    (mx,my) = ellipse_perimeter(0,0,bmin2,bmaj2,orientation=bpa2)
    P = np.arctan2(my,mx)

    print(bmin2,bmaj2,bpa2)
    print(A,B,P)
    a = r_e(A,B,P,t_m(A,B,P,1./Y),y=1./Y)
    b = r_e(A,B,P,t_m(A,B,P,1./Y)+0.5*np.pi,y=1./Y)
 # (mx,my)= ellipse (A,B,P,t_m(A,B,P,1./Y),y=1./Y)
    (mx,my) = ellipse_perimeter(0,0,A,B,orientation=P)
    p = np.arctan2 (my,mx)
    p += dpa

    if b > a:
        (a,b,p) = (b,a, p+0.5*np.pi) # make sure that 'a' is the semimajor axis
    return (a,b,p % np.pi)
