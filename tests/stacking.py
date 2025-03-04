from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt


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

im_cube=ImageCube(images)

#plot it
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

im_cube=im_cube.regrid(1024,0.03)

stack=im_cube.stack(mode="freq")

stack.plot(plot_mode="lin_pol",plot_evpa=True)
