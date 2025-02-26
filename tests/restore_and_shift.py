from vcat import ImageData
from vcat.graph_generator import FitsImage
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/0215_pol/0215+015.icn.fits",
        model="../dataset_example/0215_pol/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")


FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()
print(data.integrated_pol_flux_clean)


#Test convolution
data_convolved=data.restore(10,2,-10,useDIFMAP=False)
FitsImage(data_convolved,plot_mode="lin_pol",plot_evpa=True,plot_mask=True)
print(data_convolved.integrated_pol_flux_clean)
plt.show()


#Test shift
data_shifted=data.shift(10,5)
FitsImage(data_shifted,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

