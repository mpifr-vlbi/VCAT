import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse
class JetCalc(object):
    def __init__(self,theta,ba,R,block=False,jet='jet'):
        self.block=block
        self.jet=jet
        self.ba=ba
        self.R=[(x if x>=1. else 1.+1e-10) for x in R]
        self.theta = np.deg2rad(theta)
        self.theta_deg=theta

    def beta_jet(self,b_app=1):
        bapp=(self.ba if b_app==1 else b_app)
        ''' Caclulates intrinsic beta as dependent of apparent beta for jet on a range of theta_LOS'''
        return [x/(np.sin(self.theta)+x*np.cos(self.theta)) for x in bapp]
    def beta_cjet(self,b_app=1):
        ''' Caclulates intrinsic beta as dependent of apparent beta for counterjet on a range of theta_LOS'''
        bapp=(self.ba if b_app==1 else b_app)
        return [x/(np.sin(self.theta)-x*np.cos(self.theta)) for x in bapp]
    def DopplerFactor_jet(self,beta):
        gamma = 1/np.sqrt(1-beta**2)
        return 1/(gamma*(1-(beta*np.cos(self.theta))))
    def DopplerFactor_cjet(self,beta):
        gamma = 1/np.sqrt(1-beta**2)
        return 1/(gamma*(1+(beta*np.cos(self.theta))))
    def flux_ratio_symm(self,beta,alpha):
        return ((1+beta*np.cos(self.theta))/(1-beta*np.cos(self.theta)))**(2-alpha)

    def beta_R_symm(self,p=2,alpha=-1):
        '''Caclulates intrinsic beta as dependent of flux density ratio R on a range of theta_LOS'''
        b_R_s=[]
        for i in range(len(self.R)):
            b_R_s.append((1-2./(1+self.R[i]**(1./(p-alpha))))/np.cos(self.theta))
        return b_R_s
    def get_intersection(self,a1,a2):
        '''
        To get intersection of beta_jet and beta_R_symm
        use as:
        get_intersection(b_j,b_R)
        '''
        ip=lambda f1,f2: np.argwhere(np.diff(np.sign(f1-f2))).flatten()
        idx=[]
        idx.append(ip(a1[0],a2[0]))
        idx.append(ip(a1[2],a2[0]))
        idx.append(ip(a1[0],a2[2]))
        idx.append(ip(a1[2],a2[2]))
        return idx

    def plot_theta_beta(self,savefig=True,beta_j=False,beta_R=False,plot_ip=False):
        '''
        To plot beta-theta as dependent from beta_app and the flux density Ratio R.
        Uses self value of jet to calculate curves either for Jet or counterjet
        '''
        mpl.rcParams['legend.numpoints']=1
        if not beta_j:
            if self.jet=='jet':
                b_j=self.beta_jet()
            elif self.jet=='cjet':
                b_j=self.beta_cjet()
            else:
                sys.stdout("please specifiy jet='jet' or 'cjet'")
        else:
            b_j=beta_j
        if not beta_R:
            b_R=self.beta_R_symm()
        else:
            b_R=beta_R
        idx=self.get_intersection(b_j,b_R)
        x1=np.array(map(max,zip(b_R[0],b_j[0])))
        x2=np.array(map(min,zip(b_R[2],b_j[2])))
        fig,ax=plt.subplots(figsize=(12,6))
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\theta_{LOS}\,[^\circ]$')
        ls=['--','-','-.']
        c1=('darkorange' if self.jet=='jet' else 'darkred')
        c2='blue'
        for i in range(len(b_j)):
            plt.plot(b_j[i],self.theta_deg,'%s' %ls[i],color=c1,label=r'$\theta_{\beta_{app}=%.2f}$' %self.ba[i])
        for i in range(len(b_j)):
            plt.plot(b_R[i],self.theta_deg,'%s' %ls[i],color=c2,label=r'$\theta_R=%.1f$' %self.R[i])
        plt.fill_betweenx(self.theta_deg,x1,x2,where=x2>=x1,facecolor='gray',alpha=0.5)
        if plot_ip:
            plt.plot(b_j[0][idx[0]],self.theta_deg[idx[0]],'k*',markersize=5)
            plt.plot(b_j[2][idx[1]],self.theta_deg[idx[1]],'k*',markersize=5)
            plt.plot(b_j[0][idx[2]],self.theta_deg[idx[2]],'k*',markersize=5)
            plt.plot(b_j[2][idx[3]],self.theta_deg[idx[3]],'k*',markersize=5)
        axes = plt.gca()
        axes.set_xlim([0,1])
        axes.set_ylim([35,90])
        axes.minorticks_on()
        #axes.tick_params(which='major', length=10, width=2, direction='inout')
        #axes.tick_params(which='minor', length=5, width=2, direction='in')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,loc=0,fontsize=11,numpoints=1)
        ax.set_title(r'Parameter space of $\theta$ and $\beta$ for %s' %self.jet)
        if savefig==True:
            if self.block:
                plt.savefig('{0}_block{1}_max_beta_theta.pdf'.format(self.jet,self.block),dpi=300,bbox_inches='tight')
            else:
                plt.savefig('{0}_beta_theta.pdf'.format(self.jet),dpi=300,bbox_inches='tight')
        else:
            return (fig,ax)

class BothJets(JetCalc):
    def __init__(self,theta,ba,R,ba_c,block=False,jet='jet',cjet='cjet'):
        JetCalc.__init__(self,theta,ba,R,block)
        self.ba_c=ba_c
        self.cjet=cjet
    def plot_intersection(self):
        b_j=self.beta_jet(b_app=self.ba)
        b_cj=self.beta_cjet(b_app=self.ba_c)
        b_R=self.beta_R_symm()
        x1=np.array(map(max,zip(b_R[0],b_j[0])))
        x2=np.array(map(min,zip(b_R[2],b_j[2])))
        x3=np.array(map(max,zip(b_R[0],b_cj[0])))
        x4=np.array(map(min,zip(b_R[2],b_cj[2])))

        fig,ax=plt.subplots(figsize=(12,6))
        if self.block:
            ax.set_title(r'Parameter space of $\theta$ and $\beta$ for jet and counter-jet block%s' %self.block)
        else:
            ax.set_title(r'Parameter space of $\theta$ and $\beta$ for jet and counter-jet')
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\theta_{LOS}\,[^\circ]$')

        c1=['darkorange','darkred']
        c2='blue'
        ls=['--','-','-.']
        bb=[(0.9,0.8),(0.5,0.5)]
        ll=['WJet','EJet']
        pj,pcj,pR=[],[],[]
        for i in range(len(b_j)):
            pR+=plt.plot(b_R[i],self.theta_deg,'%s' %ls[i],color=c2,label=r'$\theta_R$')
        for i in range(len(b_j)):
            pj+=plt.plot(b_j[i],self.theta_deg,'%s' %ls[i],color=c1[0],label=r'$\theta_{\beta_{app}}$(%s)' %ll[0])
        for i in range(len(b_cj)):
            pcj+=plt.plot(b_cj[i],self.theta_deg,'%s' %ls[i],color=c1[1],label=r'$\theta_{\beta_{app}}$(%s)' %ll[1])

        pfj=plt.fill_betweenx(self.theta_deg,x1,x2,where=x2>=x1,facecolor=c1[0],alpha=0.5)
        pfcj=plt.fill_betweenx(self.theta_deg,x3,x4,where=x4>=x3,facecolor=c1[1],alpha=0.5)
        p1 = plt.fill(np.NaN, np.NaN, c1[0], alpha=0.5,label=ll[0])
        p2 = plt.fill(np.NaN, np.NaN, c1[1], alpha=0.5,label=ll[1])

        axes = plt.gca()
        axes.set_xlim([0,1])
        axes.set_ylim([35,90])
        axes.minorticks_on()
        display=(1,4,7,9,10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handle for i,handle in enumerate(handles) if i in display],[handle for i,handle in enumerate(labels) if i in display],loc=5,fontsize=11,numpoints=1)
        if self.block:
            plt.savefig('Intersection_Block{0}_max_beta_theta.pdf'.format(self.block),dpi=300,bbox_inches='tight')
        else:
            plt.savefig('Intersection_beta_theta.pdf',dpi=300,bbox_inches='tight')

def plot_intersection(theta,R,bapp_j,bapp_cj,block=False):
    jets=BothJets(theta,ba=bapp_j,R=R,ba_c=bapp_cj,block=block)
    jets.plot_intersection()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="If run as '__main__': Produce beta_theta plot of given beta and flux ratio. Function assumes theta(beta_app) for approaching jet. If formular for receding jet is required set --jet='cjet'. In case there are different velocities for approaching and receding jet it is possible to plot both graphs into one figure. For this set --beta_r to beta_apparent for receding jet.")
    parser.add_argument('-r','--ratio',help='Flux density ratio, give [R, R_err]',nargs=2,type=float)
    parser.add_argument('-b','--beta',help='Apparent beta of jet, give [beta,beta_err]',nargs=2,type=float)
    parser.add_argument('--jet',help="Specify if jet or counterjet. Valid 'jet'/'cjet'",default='jet')
    parser.add_argument('-br','--beta_r',help='Apparent beta of an receding jet, give [beta,beta_err]',nargs=2,type=float)
    parser.add_argument('-t','--theta',help='Define the theta_range to plot. Default [35,90]',default=[35,90],nargs=2,type=float)
    args=parser.parse_args()
    print ('Inputs: {}'.format(args))
    if args.ratio!=None:
        theta=np.linspace(args.theta[0],args.theta[1],1e4)
        R=[args.ratio[0]-args.ratio[1],args.ratio[0],args.ratio[0]+args.ratio[1]]
        if args.beta_r==None:
            beta_app=[args.beta[0]-args.beta[1],args.beta[0],args.beta[0]+args.beta[1]]
            jet=JetCalc(theta,beta_app,R,jet=args.jet)
            jet.plot_theta_beta()
        else:
            bapp_1=[args.beta[0]-args.beta[1],args.beta[0],args.beta[0]+args.beta[1]]
            bapp_2=[args.beta_r[0]-args.beta_r[1],args.beta_r[0],args.beta_r[0]+args.beta_r[1]]
            plot_intersection(theta,R,bapp_1,bapp_2)
    else:
        ''' Assumes to use max values'''
        beta_app_wj=[0.651,0.355]
        beta_app_wj_error=[0.042,0.017]
        beta_app_ej=[0.652,0.517]
        beta_app_ej_error=[0.017,0.029]

        Ratio=[1.1,1.7,1.6]
        R_error=[0.2,0.3,0.2]
        theta=np.linspace(0,90,1e4)

        bapp_w=[[x-y,x,x+y] for x,y in zip(beta_app_wj,beta_app_wj_error)]
        bapp_e=[[x-y,x,x+y] for x,y in zip(beta_app_ej,beta_app_ej_error)]
        R=[[x-y,x,x+y] for x,y in zip(Ratio,R_error)]

        bb=[1,3]
        cjet,jet,cjet_is,jet_is,b_j,b_R,b_cj=[],[],[],[],[],[],[]
        for i in range(2):
            cjet.append(JetCalc(theta,bapp_e[i],R[i],block=str(bb[i]),jet='cjet'))
            b_R.append(cjet[i].beta_R_symm())
            b_cj.append(cjet[i].beta_cjet())
            cjet_is.append(cjet[i].get_intersection(b_cj[i],b_R[i]))
            cjet[i].plot_theta_beta(beta_j=b_cj[i],beta_R=b_R[i])

            jet.append(JetCalc(theta,bapp_w[i],R[i],block=str(bb[i]),jet='jet'))
            b_j.append(jet[i].beta_jet())
            jet_is.append(jet[i].get_intersection(b_j[i],b_R[i]))
            jet[i].plot_theta_beta(beta_j=b_j[i],beta_R=b_R[i])
            plot_intersection(theta,R[i],bapp_w[i],bapp_e[i],block=str(bb[i]))

