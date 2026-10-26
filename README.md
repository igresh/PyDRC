# PyDRC: Passive Daytime Radiative Cooling & Atmospheric Water Capture Simulator

**PyDRC** is a physics-based Python modeling package for simulating the transient thermal dynamics, steady-state subcooling equilibrium, and atmospheric dew condensation yields of **Passive Daytime Radiative Cooling (PDRC)** materials under real-world meteorological conditions.

Developed to accompany research on nanoengineered porous radiative cooling coatings and atmospheric water harvesters, `PyDRC` couples spectral long-wave radiative heat transfer, angle- and sky-resolved solar irradiance, size- and tilt-dependent natural/forced convection, substrate conduction, and Chilton–Colburn mass transfer.

---

## Key Features

* **Spectral Long-Wave Radiative Transfer (`ThermalHeatTransfer`)**
  * Full Planck blackbody integration over arbitrary wavelength grids (e.g., $2.5\text{--}100\,\mu\text{m}$).
  * Supports experimental UV-Vis-NIR / FTIR spectral absorptivity profiles or a two-band step emissivity model (`emissivity_model`).
  * **Vapor-Pressure-Parameterized Atmospheric Emissivity (`spectral_humidCorr`):** Calibrated against `libRadtran` radiative transfer spectra across the $2.7\,\mu\text{m}$ $\text{H}_2\text{O}/\text{CO}_2$ band, $3.8\text{--}7.8\,\mu\text{m}$ vibronic plateau, $8.3\text{--}12.5\,\mu\text{m}$ atmospheric transparency window (including the water-vapor continuum and $9.6\,\mu\text{m}$ stratospheric $\text{O}_3$ band), $15\,\mu\text{m}$ $\text{CO}_2$ band, and $>16.5\,\mu\text{m}$ rotational water-vapor bands.
  * **Irradiance-Driven Ground Temperature Balance:** Accounts for daytime solar heating and nocturnal subcooling of surrounding ground/rooftop infrastructure seen by tilted panels.
  * Optional direct forcing or diagnostic benchmarking against ERA5 reanalysis or pyrgeometer downwelling long-wave radiation (`strd`).

* **Solar Decomposition, Anisotropic Transposition & Angular Reflection**
  * **Low-Turbidity Modified Erbs Decomposition (`erbs_australia`):** Smooth $C^1$-continuous clearness-index decomposition tailored for low-aerosol southern-hemisphere skies ($k_{d,\text{min}} = 0.085$), alongside standard `erbs` (1982).
  * **Plane-of-Array (POA) Transposition:** Built on `pvlib`, supporting anisotropic diffuse sky models (`haydavies`, `reindl`, `perez-driesse`, `isotropic`).
  * **Angle-of-Incidence Modifiers (IAM) & Bezel Shading:** Martin & Ruiz Fresnel angular loss modifiers (`iam_ar`) for direct, sky-diffuse, and ground-reflected radiation, plus geometric mounting-bezel cutoff shading (`bezel_cutoff_angle`).

* **Size- and Orientation-Aware Convective Heat & Mass Transfer**
  * **Empirical & Hybrid TARP Models:** Includes `Maestre-Valero`, `Richards`, and `Sharples` correlations, as well as their combined Thermal Analysis Research Program (`Combined-TARP-*`) formulations.
  * **Characteristic-Length Scaling (`char_length`):** Couples McAdams/ASHRAE Rayleigh-number natural convection (smoothly blended across laminar and turbulent regimes for horizontal and inclined plates) with boundary-layer-scaled forced wind convection (`boundary_layer='laminar'` or `'turbulent'`).
  * **Wind Direction & Wake Sheltering (`calculate_effective_wind_speed`):** Accounts for windward impingement versus leeward bluff-body wake separation behind tilted panels.
  * **Dew Condensation (`deriv_m`):** Chilton–Colburn heat and mass transfer analogy driven by Buck saturation vapor pressure and ambient dew point.

---

## Dependencies & Installation

`PyDRC` requires Python $\ge 3.9$ and the following scientific packages:

```bash
pip install numpy scipy pandas matplotlib pvlib
```

Clone the repository and import `Model` directly in your scripts or Jupyter notebooks:

```bash
git clone https://github.com/<your-org>/PyDRC.git
cd PyDRC
```

---

## Quick Start

### 1. Define a Radiative Cooling Surface

```python
import Model

# Define a 500 mm x 500 mm PDRC panel tilted at 30° facing South (azimuth = 180°)
surface = Model.Surface(
    name='PDRC_Coating',
    solar_reflectivity=0.96,               # 96% solar reflectance (alpha_solar = 0.04)
    thermal_emissivity=0.95,               # 95% LWIR thermal emissivity
    tilt=30.0,                             # Panel tilt angle from horizontal (deg)
    azimuth=180.0,                         # Panel facing direction (180° = South)
    latitude=-33.8688,                     # Sydney, Australia
    longitude=151.2093,
    timezone='Australia/Sydney',
    convection_model='Combined-TARP-Maestre',
    char_length=0.125,                     # L = Area / Perimeter = 0.5 / 4 = 0.125 m
    boundary_layer='laminar',              # 'laminar' (n=0.5) or 'turbulent' (n=0.2)
    insulation_conductivity=0.035,         # Backing foam k [W/(m·K)]
    insulation_thickness=0.050,            # Backing foam thickness [m]
    ground_h_eff=30.0,                     # Rooftop effective heat transfer coeff [W/(m²·K)]
    surrounds_albedo=0.15,
    transposition_model='haydavies',
    decomposition_model='erbs_australia',
    use_iam=True,
    iam_ar=0.16,
    bezel_cutoff_angle=85.0,
)
```

### 2. Initialize the Spectral Thermal Radiation Model

```python
radht = Model.ThermalHeatTransfer(
    low_wavelength=2.5e-6,                 # 2.5 µm
    high_wavelength=50.0e-6,               # 50 µm
    num_wavelengths=200,
    surface=surface,
    humidity_model=Model.spectral_humidCorr,
    cloud_temperature_delTamb=-8.0,        # Cloud-base lapse offset (K)
)
```

### 3. Run a Transient Time-Series Simulation

Given a meteorological `pandas.DataFrame` (`df_raw`) indexed by timestamp:

```python
# Prepare aligned weather and plane-of-array solar irradiance DataFrame
exp_df = Model.make_dataframe(df_raw, surface)

# Solve coupled ODEs for surface temperature and condensed water mass
sim = Model.SurfaceSimulator(surface, radht, exp_df, use_measured_strd=False)
results_df = sim.run_simulation()

# Plot experimental vs. simulated surface temperature and cumulative dew yield
Model.plot_simulation(results_df, relT=True)
```

### 4. Solve for Steady-State Equilibrium Subcooling

To evaluate the steady-state temperature difference $\Delta T = T_{\text{surf}} - T_{\text{amb}}$ as a function of solar irradiance ($\text{GHI}$) and optical properties:

```python
eq = Model.equil_solver(
    surface=surface,
    radht=radht,
    Tamb=20.0,
    humidity=60.0,
    windspeed=1.5,
    cloud_cover=0.0,
)

# Subcooling (K) at GHI = 800 W/m² and R_solar = 0.96
delta_T = eq.T_ghi(ghi=800.0, reflectivity=0.96)
print(f"Equilibrium T_surf - T_amb: {delta_T:+.2f} °C")
```

---

## Core Architecture (`Model.py`)

| Class / Function | Description |
| :--- | :--- |
| **`Surface`** | Stores panel geometry (`tilt`, `azimuth`, `sky_view_factor`, `char_length`), optical properties (`solar_reflectivity`, `thermal_emissivity`, `spectral_absorptivity`), thermal mass (`thickness`, `density`, `spec_heat_cap`), rear insulation (`back_conductance`), and solar/aerodynamic configuration. |
| **`ThermalHeatTransfer`** | Computes spectral blackbody emission (`P_outgoing`), clear-sky and cloudy downwelling atmospheric radiation (`sky_spectrum`, `downwelling_sky_radiation`), ground temperature (`ground_temperature`), and net radiative flux (`P_incoming - P_outgoing`). |
| **`SurfaceSimulator`** | Integrates the coupled transient ODEs for surface temperature $T_{\text{surf}}(t)$ and condensed water mass $m_{\text{cond}}(t)$ using `scipy.integrate.solve_ivp`. |
| **`equil_solver`** | Finds steady-state equilibrium surface temperatures ($q_{\text{net}} = 0$) across solar irradiance (`T_ghi`) and thermal emissivity (`T_ghi_LWIRemm`) grids. |
| **`emissivity_model`** | Two-band step emissivity model ($\lambda_{\text{cut}} = 3.0\,\mu\text{m}$ default) and legacy square-root transition model (`model='Tristram_legacy'`). |
| **`spectral_humidCorr`** | Vapor-pressure-parameterized clear-sky atmospheric spectral emissivity model (with `model='tristram'` available for legacy linear-RH comparisons). |
| **`calculate_plane_of_array_irradiance`** | Decomposes GHI via `erbs_australia` or `erbs`, transposes onto the tilted plane via `pvlib`, and applies Martin & Ruiz IAM and bezel shading. |
| **`conv_heat_transfer_coeff`** | Evaluates convective heat transfer coefficients $h_c$ for `'Maestre-Valero'`, `'Richards'`, `'Sharples'`, and `'Combined-TARP-*'` models. |
| **`get_tarp_natural_convection`** | Computes tilt- and size-dependent Rayleigh-number natural convection with smooth $C^1$ logistic transitions between laminar and turbulent regimes. |
| **`diagnose_strd`** | Statistical and visual diagnostic tool comparing modeled broadband/in-band downwelling long-wave radiation against ERA5 or pyrgeometer `strd` observations. |

---

## Citation

If you use `PyDRC` in your research, please cite:

> Chiu, M., Theau, E., Harrison, A., de Sterke, C. M., Alexander, T. J., & Neto, C. (2025). *Passively Cooled Paint-Like Coatings for Atmospheric Water Capture*. **Advanced Functional Materials**, 36(19), 2519108. https://doi.org/10.1002/adfm.202519108

---

## License

This project is released for academic and research use. Please refer to the repository license or contact the authors for commercial licensing details.
