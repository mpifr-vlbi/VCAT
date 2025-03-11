from vcat import ImageData,ImageCube
from vcat.graph_generator import KinematicPlot
import matplotlib.pyplot as plt
import numpy as np

#Import Data
dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits",
        model="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.mfit")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits",
        model="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.mfit")
dataU=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.fits",
        model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.mfit")
dataW=ImageData("../dataset_example/3C111_W_2014_05_08/3C111_W_2014_05_08.fits",
        model="../dataset_example/3C111_W_2014_05_08/3C111_W_2014_05_08.mfit")
dataX=ImageData("../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.fits",
        model="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.mfit")


data=[dataC,dataQ,dataU,dataW,dataX]

im_cube=ImageCube(data)

im_cube.plot(overplot_gauss=True,plot_comp_ids=True)

#choose the first component collection as a test
cc=im_cube.get_comp_collection(1)
print(cc)

for comp in cc.components:
    print(comp)

#fit the spectrum
fit=cc.fit_comp_spectrum()

#plot the spectrum
plot=KinematicPlot()

plot.plot_spectrum(cc,"black")
plot.plot_spectral_fit(fit)
plot.set_scale("log","log")
plt.show()


"""
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
"""


#now calculate core shift
fit_result=cc.get_coreshift()
plot=KinematicPlot()

plot.plot_coreshift_fit(fit_result)
plt.show()

