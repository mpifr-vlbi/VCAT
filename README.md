VCAT
====

VCAT is a project to collect analysis scripts for VLBI analysis. It is part of the ERC advanced grant M2FINDERS.

Currently it combines the following analysis methods:

1. VLBI image alignment using 2D crosscorrelation and modelfit components
2. Pixel based image analysis of the continuum spectrum
3. Ridgeline fitting

Requirements
------------
There are a couple of python packages required for the software package:

* astropy
* scipy
* numpy
* scikit-image
* ehtim

Furthermore it is recommended to install the following astronomical packages for more advanced routines:

* CASA
* difmap

Information for contributors
----------------------------

Info for yall:  
the py3_VIMAP... file is thedefault VIMAP but Petra moved it to py3  
the mod_VIMAP... Jan will start modifying using ehtim and other nifty stuff he finds on StackOverflow 

Acknowledgement
---------------
This publication/presentation is part of the M2FINDERS project which has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 Research and Innovation Programme (grant agreement No 101018682).



