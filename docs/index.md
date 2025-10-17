# VCAT — VLBI Comprehensive Analysis Toolkit

**VCAT (VLBI Comprehensive Analysis Toolkit)** is a Python package that provides a unified framework for the analysis of Very Long Baseline Interferometry (VLBI) data. It was developed in collaboration with several colleagues from the VLBI community (A. Baczko, V. Bartolini, F. Eppel, F. Pötzl, L. Ricci, J. Röder, F. Rösch) to combine existing analysis scripts into a single, coherent package that supports a wide range of VLBI analysis methods.

Many of the implemented analysis techniques have been used for a long time, while others were specifically developed for recent scientific projects. VCAT is still evolving — we’d love your help! Contributions, feedback, and ideas are always welcome.
All included methods are applicable to any source with standard VLBI data available in .fits, .uvf format.

---

## Table of Contents

This documentation includes detailed information about the code base, but also several example application as jupyter-notebooks:

1. [Tutorials](tutorials.md)
2. [Code Documentation](imagedata.md)

Quickly find what you're looking for depending on
your use case by looking at the different pages.

---

## Installation

The package can be installed directly from PyPI:

```bash
pip install vcat-vlbi
```

!!! Important note
    Many tasks run **DIFMAP** in the background.  
    If DIFMAP is already in your `PATH`, VCAT will automatically find it.  
    If you don't have DIFMAP yet, see below for a quick installation instruction for Ubuntu-based systems.
    t is also possible to use many functions without DIFMAP by specifying `use_difmap=False`, but this mode is **not yet fully supported** and may have limited functionality.

### Installing DIFMAP

VCAT relies on **DIFMAP** for several imaging and model-fitting tasks.  
To install it on Linux, first install the required libraries:  
```bash
sudo apt-get install pgplot5 fort77 libx11-dev libncurses-dev gawk cvs gfortran make
```
Then download and compile DIFMAP from the [Caltech FTP server](ftp://ftp.astro.caltech.edu/pub/difmap/difmap2.5e.tar.gz):  
```bash
tar -xvf difmap2.5e.tar.gz
cd uvf_difmap
./configure linux-i486-gcc
./makeall
```
Finally, add it to your PATH (e.g., in `~/.bashrc`):  
```bash
alias difmap=~/uvf_difmap/difmap
```
After reloading your shell (`source ~/.bashrc`), you can run `difmap` from anywhere.

---

## Citation

If you use **VCAT** in your research, please cite:  
**INSERT VCAT REFERENCE HERE**

---

## License

VCAT is released under the **GNU General Public License v3.0 (GPL-3.0)**.  
You are free to use, modify, and distribute this software under the terms of the GPL-3.0 license.  
For more details, see the [LICENSE](LICENSE) file included with this repository.

---

## Acknowledgements

The VCAT project is part of the M2FIDNERS project. For an overview on the project please visit:
<a href="https://www.mpifr-bonn.mpg.de/m2finders" target="_blank">M2FINDERS</a>

!!! info

    M2FINDERS project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 101018682)

**VCAT – VLBI Comprehensive Analysis Toolkit**  
A unified framework for the analysis and visualization of VLBI data.
