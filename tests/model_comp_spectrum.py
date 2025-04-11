from vcat import ImageData
from vcat.kinematics import ComponentCollection
from vcat.plots.fits_image import KinematicPlot
import matplotlib.pyplot as plt
import numpy as np

#Import Data
dataC=ImageData(model="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.modelfits")
dataQ=ImageData(model="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.modelfits")
dataU=ImageData(model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.modelfits")
dataW=ImageData(model="../dataset_example/3C111_W_2014_05_08/3C111_W_2014_05_08.modelfits")
dataX=ImageData(model="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.modelfits")

data=[dataC,dataQ,dataU,dataW,dataX]

comp_list=[]
for dat in data:
    print(dat.freq)
    #for test purposes we will just use the first component in the list
    comp=dat.components[0]
    comp.component_number=1
    comp_list.append(comp)

#Now we create a ComponentCollection for the same component across frequencies
cc=ComponentCollection(comp_list)

#fit the spectrum
fit=cc.fit_comp_spectrum()[0]

#plot the spectrum
plot=KinematicPlot()

plot.plot_spectrum(cc,"black")
plot.plot_spectral_fit(fit)
plot.set_scale("log","log")
plt.show()



#test core-shift with modelcomponent

#first simulate core distances
k_r=1
r0=50
ref_freq=86
dist_0=1

max_i=np.argmax(cc.freqs)

for i in range(len(cc.components)):
    coreshift=r0*((cc.freqs[i]*1e-9/ref_freq)**(-1/k_r)-1)*1e-3
    cc.dist[i]=cc.dist[max_i]-coreshift
    cc.components[i].distance_to_core=cc.components[max_i].distance_to_core-coreshift/cc.scale
 
#now calculate core shift
fit_result=cc.get_coreshift()
plot=KinematicPlot()

plot.plot_coreshift_fit(fit_result)
plt.show()
