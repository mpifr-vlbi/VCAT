import matplotlib as mpl
import matplotlib.pyplot as plt
from vcat.helpers import set_figsize, scatter, mas2pc
from vcat.config import logger,font
from vcat.fit_functions import powerlaw, broken_powerlaw, powerlaw_withr0, broken_powerlaw_withr0
from functools import partial

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = font
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=13
font_size_axis_tick=12

class JetProfilePlot(object):

    def __init__(self,jet="Jet",fig_size="aanda*",xlabel="Distance from core [mas]",ylabel="De-convolved width [mas]",
                 xscale="log",yscale="log",secxax=r"De-projected distance from core [pc]",
                 secyax=r'De-convolved width [pc]',xlim=None,ylim=None,redshift=0):

        super().__init__()

        self.jet=jet

        if jet=="Jet" or jet=="CJet":
            figsize = (set_figsize(fig_size, subplots=(1, 1)))
            self.fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0},
                                 figsize=figsize)
            self.axes = [ax]
        elif jet=="Twin":
            figsize = (set_figsize(fig_size, subplots=(1, 2)))
            self.fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0},
                                        figsize=figsize)
            self.axes = ax.flatten()
        else:
            raise Exception("Please specify valid 'jet' parameter ('Jet','CJet','Twin')")

        def mas_to_pc(x):
            return x * float(mas2pc(redshift))

        def pc_to_mas(x):
            return x / float(mas2pc(redshift))

        for ax in self.axes:
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(ylim[0],ylim[1])
            if xlim:
                ax.set_xlim(xlim[0],xlim[1])
            ax.minorticks_on()
            ax.tick_params(which='both', direction='inout')
            ax.label_outer()



            secx = ax.secondary_xaxis('top', functions=(mas_to_pc, pc_to_mas))
            secx.set_xlabel(secxax)

        if jet=="Twin":
            self.axes[0].invert_xaxis()

            if secyax:
                secy = self.axes[1].secondary_yaxis('right', functions=(mas_to_pc, pc_to_mas))
                secy.set_ylabel(secyax)

    def plot_profile(self,dist,width,width_err,color,marker,label):

        if self.jet=="Twin":
            logger.debug("Plotting twin jet.")
            #this means dist, width and width_err need to be 2d-lists
            #plot jet
            self.axes[0].scatter(dist[0], width[0], s=4, marker=marker, color=color, label='{}'.format(label))
            self.axes[0].errorbar(dist[0], width[0], yerr=width_err[0], fmt=marker, ms=0, color=color,
                                 linewidth=0, elinewidth=0.4, errorevery=1, label=label, alpha=0.3)
            #plot counterjet
            self.axes[1].scatter(dist[1], width[1], s=4, marker=marker, color=color)
            self.axes[1].errorbar(dist[1], width[1], yerr=width_err[1], fmt=marker, ms=0, color=color,
                                  linewidth=0, elinewidth=0.4, errorevery=1, label=label, alpha=0.3)
        else:
            logger.debug("Plotting only one jet.")
            self.axes[0].scatter(dist, width, s=4, marker=marker, color=color, label='{}'.format(label))
            self.axes[0].errorbar(dist, width, yerr=width_err, fmt=marker, ms=0, color=color, linewidth=0,
                        elinewidth=0.4, errorevery=1, label=label, alpha=0.3)

    def plot_fit(self, x, fitfunc, beta, betaerr, chi2, jet="Jet",color="k",annotate=False,asize=8,annox=0.6,annoy=0.05,lw=1,ls="-",label=None,fit_r0=True,s=100):

        if self.jet=="Twin":
            if jet=="Jet":
                ax=self.axes[1]
            elif jet=="CJet":
                ax=self.axes[0]
            else:
                raise Exception("Please provide valid 'jet' parameter ('Jet','CJet')")
        else:
            ax=self.axes[0]


        if fitfunc == 'scatter':
            function = scatter(beta, x)
            text = '$theta_s={:.2f}\\pm{:.2f}$\n $theta_i={:.2f}\\pm{:.2f}$\n$\\chi_\\mathrm{{red}}^2={:.2f}$'.format(
                beta[0], betaerr[1], beta[1], betaerr[1], chi2)

        if fitfunc == 'Powerlaw':
            if fit_r0:
                function = powerlaw_withr0(beta, x)
                text = '{}\n$k={:.2f}\\pm{:.2f}$\n r0={:.2f}\\pm{:.2f}$mas\n $\\chi_\\mathrm{{red}}^2={:.2f}$'.format(label, beta[1],
                                                                                            betaerr[1],beta[2],betaerr[2], chi2)
            else:
                function = powerlaw(beta, x)
                text = '{}\n$k={:.2f}\\pm{:.2f}$\n $\\chi_\\mathrm{{red}}^2={:.2f}$'.format(label, beta[1],
                                                                                        betaerr[1], chi2)
        elif fitfunc == 'brokenPowerlaw':
            if fit_r0:
                function = broken_powerlaw_withr0(beta,x,s=s)
                text = '{}\n$W_\\mathrm{{0}}={:.2f}\\pm{:.2f}$\n$k_\\mathrm{{u}}={:.2f}\\pm{:.2f}$\n$k_\\mathrm{{d}}={:.2f}\\pm{:.2f}$\n$z_\\mathrm{{B}} = {:.1f}\\pm {:.1f}$ mas'.format(
                    label, beta[0], betaerr[0], beta[1], betaerr[1], beta[2], betaerr[2], beta[3], betaerr[3])
            else:
                function = broken_powerlaw(beta, x,s=s)
                text = '{}\n$W_\\mathrm{{0}}={:.2f}\\pm{:.2f}$\n$k_\\mathrm{{u}}={:.2f}\\pm{:.2f}$\n$k_\\mathrm{{d}}={:.2f}\\pm{:.2f}$\n$z_\\mathrm{{B}} = {:.1f}\\pm {:.1f}$ mas \n r0={:2.f}\\pm{:.2f}$mas\n'.format(
                    label, beta[0], betaerr[0], beta[1], betaerr[1], beta[2], betaerr[2], beta[3], betaerr[3],beta[4],betaerr[4])

        if label:
            ax.plot(x, function, color=color, lw=lw, ls=ls, label=label, zorder=1)
        else:
            ax.plot(x, function, color=color, lw=lw, ls=ls, zorder=1)
        if annotate:
            ax.annotate(text, xy=(annox, annoy), xycoords='axes fraction', size=asize,
                        horizontalalignment='left', verticalalignment='bottom', bbox=bbox_props)

    def plot_legend(self):
        handles, labels = self.axes[0].get_legend_handles_labels()

        if self.jet=="Twin":
            self.axes[0].legend(handles, labels, loc='lower left', ncol=5, markerscale=1, labelspacing=0.1,
                           bbox_to_anchor=(0.0, 1.3, 2., .4), mode='expand', borderaxespad=0., handletextpad=0.1)
        else:
            self.axes[0].legend(handles, labels, loc='lower left', ncol=2, markerscale=1, labelspacing=0.1,
                           bbox_to_anchor=(0.0, 1.15, 2., .4), borderaxespad=0., handletextpad=0.1)

    def savefig(self,savefig):
        plt.savefig(savefig,bbox_inches="tight")
        logger.info(f"Saved plot to file {savefig}.")