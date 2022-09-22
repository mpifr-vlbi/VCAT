import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimization
from itertools import groupby
from scipy import stats
from scipy.odr import *
from astropy.io import fits

def fractdiff(y,ym):
    return (y/ym)-1

def resid(y,ym,sd=None):
    '''
    Returns the residues of data an dmodel taking standard deviations sd into account if desired
    '''
    if sd is None:
        res = y-ym
    else:
        res = (y-ym)/sd
    return np.array(res)

def chisq(y,ym,sd=None):
    '''
    Derives the Chi-square based on data y and model ym at the same x points.
    If standard deviations are to be taken into account specify sd
    '''
    #res =resid(y,ym)**2/sd
    res = resid(y,ym,sd=sd)**2
    return np.sum([r for r in res])

def deltachi(y,ym,sd):
    return resid(y,ym)**2/sd

def redchisq(y,ym,fpars=1,sd=None):
    '''
    Calculates the reduced chi-squared making use of the function chisq.
    To evaluate correctly please specify the free model parameters as fpars
    '''
    csq = chisq(y,ym,sd=sd)
    return csq/(y.size-fpars)

def linar_fit(p,x):
    a,b=p
    return a+b*x

def powerlaw(p,x):
    a,b=p
    return a*(x**b)

def powerlawVar(p,x):
    a,b,c=p
    return a*(x**(-1/b))+c

def broken_powerlaw(p,x):
    w0,au,ad,xb=p
    s=10
    return w0*2**((au-ad)/s)*(x/xb)**au*(1+(x/xb)**s)**((ad-au)/s)

def Snu(p,nu):
    num,Sm,athick,athin = p
    taum =3/2*(np.sqrt(1-(8*athin/3/athick))-1)
    return Sm*(nu/num)**athick*(1-np.exp(-taum*(nu/num)**(athin-athick)))/(1-np.exp(-taum))

def scatter(p,x):
    ws,wi = p
    xx= 1
    return np.sqrt(np.square(ws*x**-2.2) + np.square(wi*x*xx))

def odr_fit(func,data,x0,fit_type=2,verbose=False,maxit=1e4):
    model=Model(func)
    if len(data)==3:
        x,y,dy=data
    elif len(data)==4:
        x,y,dy,dx=data
    else:
        print ("Syntax: odr_log(func,[x,y,dy[,dx]],x0...)")
    #x0=x0
    if 'dx' in vars() or 'dx' in globals():
        print('fitting for x-errors')
        fitdata=RealData(x,y,sx=dx,sy=dy)
        fit_type = 0
        #fitdata=RealData(x,y,wd=1./np.power(dx,1),we=1./np.power(dy,1))
    else:
        fitdata=RealData(x,y,sy=dy)
        fit_type=2
        #fitdata=Data(x,y,we=1./np.power(dy,2)) #this is what is done in RealData to convert standard deviation dy to weights wd
    print('fit_type='+str(fit_type))
    myodr=ODR(fitdata,model,beta0=x0,maxit=int(maxit))
    myodr.set_job(fit_type=fit_type)
    if verbose == 2:
        myodr.set_iprint(final=2)
    out=myodr.run()
    out.pprint()
    if out.stopreason[0] == 'Iteration limit reached':
        print ('(WWW) poly_lsq: Iteration limit reached, result not reliable!')
    #return out.beta[0],out.beta[1],out.sd_beta[0],out.sd_beta[1],out
    return out.beta,out.sd_beta,out.res_var ,out

def fit_func(x,y,sd,function,x0=False):
    if type(x)==list:
        x = np.concatenate(np.abs(x))
        y = np.concatenate(np.abs(y))
        sd = np.concatenate(sd)
    if x0 is False:
        x0 = np.array([0.1,1])
    beta,sd_beta,chi2fit,out = odr_fit(function,[x,y,sd],x0,verbose=1)
    return beta,sd_beta,chi2fit,out

def fit_pl(x,y,sd,x0=False):
    if type(x)==list:
        x = np.concatenate(np.abs(x))
        y = np.concatenate(np.abs(y))
        sd = np.concatenate(sd)
    if x0 is False:
        x0 = np.array([0.1,1])
    beta,sd_beta,chi2fit,out = odr_fit(powerlaw,[x,y,sd],x0,verbose=1)
    return beta,sd_beta,chi2fit,out

def fit_scatter(x,y,sd,x0=False):
    if type(x)==list:
        x = np.concatenate(np.abs(x))
        y = np.concatenate(np.abs(y))
        sd = np.concatenate(sd)
    if x0 is False:
        x0 = np.array([0.1,1])
    beta,sd_beta,chi2fit,out = odr_fit(scatter,[x,y,sd],x0,verbose=1)
    return beta,sd_beta,chi2fit,out

def fit_bpl(x,y,sd,sx=False,x0=False):
    if type(x)==list:
        x = np.concatenate(np.abs(x))
        y = np.concatenate(np.abs(y))
        sd = np.concatenate(sd)
    if x0 is False:
        x0=np.array([min(np.concatenate(y)),0,1,2])
    if sx is False:
        print('only use y-error')
        #beta,sd_beta,chi2fit,out = odr_fit(broken_powerlaw,[np.concatenate(x),np.concatenate(y),np.concatenate(sd)],x0,verbose=1)
        beta,sd_beta,chi2fit,out = odr_fit(broken_powerlaw,[x,y,sd],x0,verbose=1)
    else:
        if type(sx)==list:
            sx = np.concatenate(sx)
        print('include x error\n')
        beta,sd_beta,chi2fit,out = odr_fit(broken_powerlaw,[x,y,sd,sx],x0,verbose=1)
    return beta,sd_beta,chi2fit,out


def read_fits(file,col1,col2,dcol1,dcol2):
    hdulist=fits.open(file)
    data=hdulist[1].data
    x=data[col1]
    y=data[col2]
    dx=data[dcol1]
    dy=data[dcol2]
    hdulist.close()
    return x,y,dx,dy

def log_error(x,dx):
    return 1./np.log(10)*dx/x

def keyfunc(s):
      return [int(''.join(g)) if k else ''.join(g) for k, g in groupby(s, str.isdigit)]


