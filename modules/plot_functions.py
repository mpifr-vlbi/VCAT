#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import sys
from itertools import cycle
from matplotlib.rcsetup import cycler
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,AutoLocator,ScalarFormatter,FuncFormatter,MaxNLocator)
from matplotlib.patches import Ellipse
from math import log10,floor,ceil
from matplotlib.colors import LogNorm
#
from VLBIana.modules.jet_calculus import *
import VLBIana.modules.fit_functions as ff
from VLBIana.modules.plotSet import *
###################################
#plt.style.use('talkstyle')

default_cmap = 'gist_earth'
colormap = 'inferno'
mpl.rcParams['image.cmap'] = default_cmap
cmap = cm.get_cmap(default_cmap)
bbox_props = dict(boxstyle="square,pad=0.3",color='k',fill=None, lw=0.5)
n=8
colors = cmap(np.linspace(0,0.95,n))
asize = 8

mm = ['x','>','<','+','d','*','p','o','2']
markers = cycle(mm)

mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)


def plotHist(data,ax=None,**kwargs):
    '''Possible keywords and default values:
    'plot_norm':False
    'color':'k'
    'xlabel':'Data'
    'ylabel':'Proportion',
    'fsize':8
    'acoord':(0.6,0.9)
    '''
    args = {'xlabel':'Data','ylabel':'Proportion','asize':asize,'acoord':(0.8,0.9)}
    diff = set(kwargs.keys()) - set(args.keys())

    args.update(kwargs)

    ax = ax or plt.gca()
    ax.hist(data, alpha=0.8,density=True, bins=50)
    ax.set_xlabel = args['xlabel']
    ax.set_ylabel = args['ylabel']

    if 'plot_norm' in args:
        xhmin,xhmax = plt.xlim()
        mu,std = stats.norm.fit(data)
        xhist = np.linspace(xhmin,xhmax,200)
        ax.plot(np.sort(data),stats.norm.pdf(np.sort(data),mu,std))
        ax.annotate('$\mu ={:.1f}$ \n$std={:.1f}$'.format(mu,std),xy=args['acoord'],xycoords='axes fraction',size=args['asize'],horizontalalignment='left',verticalalignment='top')

def axesWidthPlot (ax, **kwargs):
    args ={'xlabel':r'Distance from the core \small{[mas]}','ylabel':r'De-convolved jet width \small{[mas]}','xscale':'log','yscale':'log'}
    args.update(kwargs)

    xlabel = args['xlabel']
    ylabel = args['ylabel']

    ax.set_xscale(args['xscale'])
    ax.set_yscale(args['yscale'])
    if args['xlabel']:
        ax.set_xlabel(xlabel)
    if args['ylabel']:
        ax.set_ylabel(ylabel)
    if 'secxax' in args:
        secx = ax.secondary_xaxis('top', functions=(mas2Rs, Rs2mas))
        secx.set_xlabel(args['secxax'])
    if 'secyax' in args:
        secy = ax.secondary_yaxis('right', functions=(mas2Rs, Rs2mas))
        secy.set_ylabel(args['secyax'])


def plot_fit(x,fitfunc,beta,betaerr,chi2,ax=None,**kwargs):
    args = {'color':'k', 'annotate':False,'asize':asize,'annox':0.6,'annoy':0.05,'lw':1}
    args.update(kwargs)
    ax = ax or plt.gca()
    if fitfunc == 'scatter':
        function = ff.scatter(beta,x)
        text = '$theta_s={:.2f}\pm{:.2f}$\n $theta_i={:.2f}\pm{:.2f}$\n$\chi_\mathrm{{red}}^2={:.2f}$'.format(beta[0],betaerr[1],beta[1],betaerr[1],chi2)

    if fitfunc == 'Powerlaw':
        function = ff.powerlaw(beta,x)
        text = '$k={:.2f}\pm{:.2f}$\n $\chi_\mathrm{{red}}^2={:.2f}$'.format(beta[1],betaerr[1],chi2)
    elif fitfunc == 'brokenPowerlaw':
        function = ff.broken_powerlaw(beta,x)
        #text = '$W_\mathrm{{0}}={:.2f}\pm{:.2f}$\n$k_\mathrm{{u}}={:.2f}\pm{:.2f}$\n$k_\mathrm{{d}}={:.2f}\pm{:.2f}$\n$\chi_\mathrm{{red}}^2={:.2f}$\nBreak at {:.1f} mas'.format(beta[0],betaerr[0],beta[1],betaerr[1],beta[2],betaerr[2],chi2,beta[-1])
        text = '$W_\mathrm{{0}}={:.2f}\pm{:.2f}$\n$k_\mathrm{{u}}={:.2f}\pm{:.2f}$\n$k_\mathrm{{d}}={:.2f}\pm{:.2f}$\n$z_\mathrm{{B}} = {:.1f}\pm {:.1f}$ mas'.format(beta[0],betaerr[0],beta[1],betaerr[1],beta[2],betaerr[2],beta[-1],betaerr[-1])

    ax.plot(x,function,'-'+args['color'],lw=args['lw'])
    if args['annotate']:
        ax.annotate(text, xy=(args['annox'],args['annoy']),xycoords='axes fraction',size=args['asize'],horizontalalignment='left',verticalalignment='bottom',bbox=bbox_props)

def add_subplot_unshare(ax):
    ''' based on an answer from stacked overflow 
    '''
    ax._shared_x_axes.remove(ax)
    ax.xaxis.major = mpl.axis.Ticker()
    xloc = AutoLocator()
    xfmt = ScalarFormatter()
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(xfmt)
    ax.yaxis.set_tick_params(which='both', labelleft=True)

#########
# I changed order bmaj, bmin !!!! check all plots!
#########
def plotBeam(bmaj,bmin,bpos,ramax,decmin,ax=None,color='grey'):
    ax = ax or plt.gca()
    ell_dist = 2
    ell_x = ramax-bmaj*ell_dist
    ell_y = decmin+bmaj*ell_dist
    if color:
        e = Ellipse([ell_x,ell_y],bmaj,bmin,-bpos, fc=color,zorder=2)
    else:
        e = Ellipse([ell_x,ell_y],bmaj,bmin,-bpos, fc=color,zorder=2)
    ax.add_artist(e)

def privImshow(img,noise,extent,ax=None,**kwargs):
    args = {'cmap':colormap,'sigma':1}
    args.update(kwargs)
    ax = ax or plt.gca()

    level0  = noise*args['sigma']
    lev=[]
    for i in range(0,10):
        lev.append(level0*2**i)

    col = ax.imshow(img,cmap=args['cmap'],norm=mpl.colors.SymLogNorm(linthresh=level0,linscale=0.5,vmin=lev[0],vmax=0.5*ma.amax(img)),extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad='3%')
    cbar = plt.colorbar(col,cax=cax)
    cbar.set_label('Flux Density [Jy/beam]')

