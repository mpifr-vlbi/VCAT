from vcat import ImageData,FitsImage
import matplotlib.pyplot as plt

useDIFMAP=False

#Import Data
data=ImageData("../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/Ku/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/Ku/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/Ku/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

#Import copy of the same data (note: must be seperate fits files!!!)
data2=ImageData("../dataset_example/0215_pol/K/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/K/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/K/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/K/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")


#shift and regrid
data_shifted=data2.shift(5,5,useDIFMAP=useDIFMAP)
data=data.regrid(1024,0.05,useDIFMAP=useDIFMAP)

print("Plotting Image")
FitsImage(data,plot_mode="stokes_i")
plt.show()

print("Plotting Shifted Image")
FitsImage(data_shifted,plot_mode="stokes_i",plot_evpa=True)
plt.show()


print("Aligning shifted image with image before using cross-correlation")
data_aligned=data_shifted.align(data,auto_regrid=True,useDIFMAP=useDIFMAP)
data_aligned.plot()
plt.show()

#Import Data
data=ImageData("../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/Ku/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/Ku/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/Ku/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

#Import copy of the same data (note: must be seperate fits files!!!)
data2=ImageData("../dataset_example/0215_pol/K/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/K/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/K/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/K/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

#shift and regrid
data_shifted=data2.shift(5,5,useDIFMAP=useDIFMAP).restore(1,1,180,useDIFMAP=useDIFMAP)
data=data.restore(0.5,0.5,90,useDIFMAP=useDIFMAP)

print("Aligning shifted image with image before using brightest pixel")
data_aligned=data_shifted.align(data,auto_regrid=True,method="brightest",useDIFMAP=useDIFMAP)
FitsImage(data_aligned,plot_mode="stokes_i")
plt.show()

