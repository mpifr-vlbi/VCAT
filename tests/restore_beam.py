from vcat import ImageData
from vcat.graph_generator import FitsImage
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/0215_pol/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")


FitsImage(data,plot_mode="lin_pol",plot_evpa=True)
plt.show()
print(data.integrated_pol_flux_clean)

data_convolved=data.restore(2,2,180)
FitsImage(data_convolved,plot_mode="lin_pol",plot_evpa=True)
print(data_convolved.integrated_pol_flux_clean)
plt.show()

#test shift
data_shifted=data.shift(30,30)
FitsImage(data_shifted,plot_mode="lin_pol",plot_evpa=True)
print(data_convolved.integrated_pol_flux_clean)
plt.show()

