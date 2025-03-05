from vcat import ImageData

data=ImageData("../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        model="../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/Ku/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/Ku/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/Ku/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

data.plot()

data_new=data.rotate(-45,useDIFMAP=False)

data_new.plot(plot_mode="lin_pol",overplot_clean=True,plot_mask=True)
