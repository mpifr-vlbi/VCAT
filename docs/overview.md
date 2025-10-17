# Overview about Code Base

VCAT consists of the two core classes: **`ImageData`** and **`ImageCube`**.
Additionally, there are several helper classes that mostly run in the background. They can also be run independently, especially the vcat.helpers submodule includes several useful functions.

## ImageData

[`ImageData`](imagedata.md) represents a single observation of a radio source at a given frequency and epoch.  
It accepts as input:

- **FITS** files (full polarization images)  
- **UVFITS** (`.uvf`) visibility data  
- **Modelfit** files (`.mod` or `.fits`)

For example, loading and plotting an image can quickly be done with

```python
from vcat import ImageData

image=ImageData("image.fits", uvf_file="data.uvf", model="model.mod")
image.plot()
```

Upon loading, key parameters such as image noise, integrated flux density, and fractional polarization are automatically calculated and stored as attributes.  
If a modelfit file is provided, VCAT also calculates the signal-to-noise ratio (SNR), brightness temperature, and component errors automatically.

`ImageData` objects can be modified or analyzed using various tasks:

- `restore()`, `regrid()`, `shift()`, `center()` – runs DIFMAP in the background to modify full-polarization images  
- `get_ridgeline()` – perform a ridgeline fit  
- `align()` – align one image to another  
- `plot()` – generate customizable plots of total intensity or polarization

---

## ImageCube

[`ImageCube`](imagecube.md) is designed to handle multi-frequency, multi-epoch data sets.  
It manages a collection of `ImageData` objects representing different observations of the same source and provides high-level analysis methods, including:

- **Spectral analysis**  
  - `get_turnover_map()` — pixel-based turnover frequency fitting  
  - `fit_comp_spectrum()` — component-based spectral fitting  

- **Kinematic analysis**  
  - `get_speed()` and `get_speed2d()` — estimate apparent jet speeds and perform kinematic analysis

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
