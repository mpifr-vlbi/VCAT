from vcat import ImageData
from vcat.kinematics import ComponentCollection
from vcat.graph_generator import KinematicPlot
import matplotlib.pyplot as plt

#Import Data
dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.modelfits")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.modelfits")
dataU=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.modelfits")
dataW=ImageData("../dataset_example/3C111_W_2014_05_08/3C111_W_2014_05_08.modelfits")
dataX=ImageData("../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.modelfits")

data=[dataC,dataQ,dataU,dataW,dataX]
comp_list=[]
for dat in data:
    print(dat.freq)
    #for test purposes we will just use the first component in the list
    comp=dat.components[0]
    comp.component_number=1
    comp_list.append(comp)
    print(comp.freq)
#Now we create a ComponentCollection for the same component across frequencies
cc=ComponentCollection(comp_list)

#fit the spectrum
fit=cc.fit_comp_spectrum()

#plot the spectrum
plot=KinematicPlot()

plot.plot_spectrum(cc,"black")
plot.plot_spectral_fit(fit)
plot.set_scale("log","log")
plt.show()
