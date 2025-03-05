from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt
import glob
useDIFMAP=True

fits_files=glob.glob("../dataset_example/0235+164/**/*.icn.fits",recursive=True)
uvf_files=glob.glob("../dataset_example/0235+164/**/*.uvf",recursive=True)
stokes_q_files=glob.glob("../dataset_example/0235+164/**/*.qcn.fits",recursive=True)
stokes_u_files=glob.glob("../dataset_example/0235+164/**/*.ucn.fits",recursive=True)

"""
#import data
EKu=ImageData("../dataset_example/0235+164/Ku/0235+164E.icn.fits",
        stokes_q="../dataset_example/0235+164/Ku/0235+164E.qcn.fits",
        stokes_u="../dataset_example/0235+164/Ku/0235+164E.ucn.fits",
        uvf_file="../dataset_example/0235+164/Ku/0235+164E.uvf")
FKu=ImageData("../dataset_example/0235+164/Ku/0235+164F.icn.fits",
        stokes_q="../dataset_example/0235+164/Ku/0235+164F.qcn.fits",
        stokes_u="../dataset_example/0235+164/Ku/0235+164F.ucn.fits",
        uvf_file="../dataset_example/0235+164/Ku/0235+164F.uvf")
EK=ImageData("../dataset_example/0235+164/K/0235+164E.icn.fits",
        stokes_q="../dataset_example/0235+164/K/0235+164E.qcn.fits",
        stokes_u="../dataset_example/0235+164/K/0235+164E.ucn.fits",
        uvf_file="../dataset_example/0235+164/K/0235+164E.uvf")
FK=ImageData("../dataset_example/0235+164/K/0235+164F.icn.fits",
        stokes_q="../dataset_example/0235+164/K/0235+164F.qcn.fits",
        stokes_u="../dataset_example/0235+164/K/0235+164F.ucn.fits",
        uvf_file="../dataset_example/0235+164/K/0235+164F.uvf")
EQ=ImageData("../dataset_example/0235+164/Q/0235+164E.icn.fits",
        stokes_q="../dataset_example/0235+164/Q/0235+164E.qcn.fits",
        stokes_u="../dataset_example/0235+164/Q/0235+164E.ucn.fits",
        uvf_file="../dataset_example/0235+164/Q/0235+164E.uvf")
FQ=ImageData("../dataset_example/0235+164/Q/0235+164F.icn.fits",
        stokes_q="../dataset_example/0235+164/Q/0235+164F.qcn.fits",
        stokes_u="../dataset_example/0235+164/Q/0235+164F.ucn.fits",
        uvf_file="../dataset_example/0235+164/Q/0235+164F.uvf")

images=[EKu,FKu,EK,FK,EQ,FQ]
"""

im_cube=ImageCube().import_files(fits_files=fits_files,uvf_files=uvf_files,stokes_q_files=stokes_q_files,stokes_u_files=stokes_u_files)

#plot it
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

im_cube=im_cube.align(mode="freq",useDIFMAP=False)
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

im_cube=im_cube.stack(mode="freq")
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)
