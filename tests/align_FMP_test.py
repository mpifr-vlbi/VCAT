from vcat import ImageData

im1=ImageData("../dataset_example/3C345/3C345_Ub_I.fits")
im2=ImageData("../dataset_example/3C345/3C345_Qb_I.fits")

im1.align(im2,auto_regrid=True)

