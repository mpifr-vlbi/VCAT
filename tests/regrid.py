from vcat import ImageData
from vcat.graph_generator import FitsImage
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        model="../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/Ku/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/Ku/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/Ku/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

data.plot(plot_mode="lin_pol",overplot_clean=True,plot_mask=True,plot_polar=False)

#Test regridding w/o difmap
print(data.integrated_flux_image)
print("Regridding without DIFMAP")
data_regrid=data.regrid(2048,0.1,useDIFMAP=False)
print(data_regrid.integrated_flux_image)
FitsImage(data_regrid,plot_mode="lin_pol",plot_evpa=True,plot_mask=True)
plt.show()



#Test regridding difmap
print("Regridding with difmap")
data_regrid=data.regrid(2048,0.1)
print(data_regrid.integrated_flux_image)
FitsImage(data_regrid,plot_mode="lin_pol",plot_evpa=True,plot_mask=True)
plt.show()
