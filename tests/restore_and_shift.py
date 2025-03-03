from vcat import ImageData
from vcat.graph_generator import FitsImage
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        model="../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/Ku/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/Ku/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/Ku/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")


FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()
print(data.integrated_pol_flux_clean)


"""
#Test convolution
print("Convolution without DIFMAP, only blurring with Gaussian")
data_convolved=data.restore(10,2,-10,useDIFMAP=False)
FitsImage(data_convolved,plot_mode="lin_pol",plot_evpa=True,plot_mask=True)
print(data_convolved.integrated_flux_image)
plt.show()
"""


#Test convolution with DIFMAP
print("Convolution with DIFMAP")
data_convolved=data.restore(10,2,-10,useDIFMAP=True)
FitsImage(data_convolved,plot_mode="lin_pol",plot_evpa=True,plot_mask=True)
print(data_convolved.integrated_pol_flux_clean)
print(data_convolved.integrated_flux_image)
plt.show()


#Test shift
data_shifted=data.restore(2,2,90,useDIFMAP=True)
FitsImage(data_shifted,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

data2=data_shifted.restore(0.5,0.5,90,useDIFMAP=True)
data_shifted.plot()
plt.show()

