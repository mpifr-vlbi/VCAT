from vcat import ImageData

data=ImageData("../dataset_example/ehtim_test.fits",stokes_q="../dataset_example/ehtim_testQ.fits",stokes_u="../dataset_example/ehtim_testU.fits")
data.plot(plot_mode="lin_pol",plot_evpa=True)
