# VCAT — VLBI Comprehensive Analysis Toolkit

**VCAT (VLBI Comprehensive Analysis Toolkit)** is a Python package that provides a unified framework for the analysis of Very Long Baseline Interferometry (VLBI) data. It was developed in collaboration with several colleagues from the VLBI community (A. Baczko, F. Pötzl, J. Röder, F. Rösch, V. Bartolini, L. Ricci) to combine existing analysis scripts into a single, coherent package that supports a wide range of VLBI analysis methods.

Many of the implemented analysis techniques—such as kinematic analysis and stacking—have been used for a long time, while others, like spine–sheath fitting and model component polarization analysis, were developed specifically for recent scientific projects.  
All included methods are applicable to any source with standard VLBI data available.

---

## Installation

The package can be installed directly from PyPI:

```bash
pip install vcat-vlbi
```

---

## Overview

VCAT consists of two main classes: **`ImageData`** and **`ImageCube`**.

### `ImageData`

`ImageData` represents a single observation of a radio source at a given frequency and epoch.  
It accepts as input:

- **FITS** files (full polarization images)  
- **UVFITS** (`.uvf`) visibility data  
- **Modelfit** files (`.mod` or `.fits`)

Upon loading, key parameters such as image noise, integrated flux density, and fractional polarization are automatically calculated and stored as attributes.  
If a modelfit file is provided, VCAT also calculates the signal-to-noise ratio (SNR), brightness temperature, and polarization errors following *Schinzel et al. (2012)*.

`ImageData` objects can be modified or analyzed using various tasks:

- `restore()`, `regrid()`, `shift()`, `center()` – run DIFMAP in the background to modify full-polarization images  
- `get_ridgeline()` – perform a ridgeline fit  
- `align()` – align one image to another  
- `plot()` – generate customizable plots of total intensity or polarization

---

### `ImageCube`

`ImageCube` is designed to handle multi-frequency, multi-epoch data sets.  
It manages a collection of `ImageData` objects representing different observations of the same source and provides high-level analysis methods, including:

- **Spectral analysis**  
  - `get_turnover_map()` — pixel-based turnover frequency fitting  
  - `fit_comp_spectrum()` — component-based spectral fitting  

- **Kinematic analysis**  
  - `get_speed()` and `get_speed2d()` — estimate apparent jet speeds  

- **Structural analysis**  
  - `stack()` — stack multiple images  
  - `fit_collimation_profile()` — fit collimation profiles (ridgeline or component-based)  

- **Spectral & polarization mapping**  
  - `get_spectral_index_map()` and `get_rm_map()` — compute spectral index and rotation measure maps  

- **Temporal evolution & visualization**  
  - `plot_evolution()` and `plot_component_evolution()` — track flux density, polarization, and EVPA evolution  
  - `movie()` — generate interpolated movies of jet evolution in total intensity or full polarization  

---

## Configuration

Usually, the standard settings are sufficient, but some parameters can be customized via a `config.yml` file.  
An example configuration file is provided in the repository.

To use a custom configuration file, define the environment variable `VCAT_CONFIG` to point to it:

```bash
export VCAT_CONFIG=/path/to/config.yml
```

### Important Note

Many tasks run **DIFMAP** in the background.  
If DIFMAP is already in your `PATH`, VCAT will automatically find it.  
It is also possible to use many functions without DIFMAP by specifying `use_difmap=False`, but this mode is **not yet fully supported** and may have limited functionality.

---

### Example `config.yml`

```yaml
# General settings:
difmap_path: "/usr/local/difmap/uvf_difmap_2.5g/"
uvw: [0, -1]

# Method selection
noise_method: "Histogram Fit"
mfit_err_method: "Schinzel12"
res_lim_method: "Kovalev05"

# Plot settings
font: "Quicksand"
plot_colors: ["#023743FF", "#FED789FF", "#72874EFF", "#476F84FF", "#A4BED5FF", "#453947FF"]
plot_markers: [".", ".", ".", ".", ".", "."]

# Cosmology
# Values from Planck Collaboration et al. 2020
H0: 67.4
Om0: 0.315

logging:
  level: "INFO"       # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: ""         # Optional: specify path to redirect log output to a file
```

---

## Example Scripts

Example scripts demonstrating the different functionalities of **VCAT** are available on the [VCAT website](INSERT LINK).  
These examples showcase how to load data, perform analysis, and generate publication-quality plots.

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

**VCAT – VLBI Comprehensive Analysis Toolkit**  
A unified framework for the analysis and visualization of VLBI data.
