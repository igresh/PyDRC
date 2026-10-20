import inspect
import warnings
import numpy as np
import pvlib
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt

# Use np.trapezoid if available (NumPy >= 2.0), otherwise fall back to np.trapz
trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


class Surface(object):
    """
    The surface class contains all relevant information about the surface material and position.
    """
    def __init__(self,
                 solar_reflectivity=0.95,
                 spectral_absorptivity=None,
                 tilt=0.0,
                 azimuth=0.0,
                 sky_view_factor=None,
                 surrounds_albedo=0.15,
                 name=None,
                 latitude=-33.8688,
                 longitude=151.2093,
                 thickness=8e-4,
                 spec_heat_cap=890,
                 density=2710,
                 convection_model='Maestre-Valero',
                 char_length=None,
                 back_conductance=0.0,
                 insulation_conductivity=None,
                 insulation_thickness=None,
                 ground_h_eff=None,
                 ground_emissivity=0.95,
                 thermal_emissivity=0.95,
                 use_iam=True,
                 iam_ar=0.16,
                 bezel_cutoff_angle=85.0,
                 transposition_model='perez-driesse',
                 use_effective_wind=False,
                 wind_f_min=0.40,
                 timezone='Australia/Sydney',
                 decomposition_model='erbs_australia',
                 min_diffuse_fraction=0.085):
        """
        timezone (str): geographic timezone for solar calculations if dataset timestamps are naive (default 'Australia/Sydney').
        decomposition_model (str): GHI to DNI/DHI decomposition model (default 'erbs_australia').
            'erbs_australia' uses a continuous Australian low-turbidity modified Erbs model with lower clear-sky diffuse floor.
            'erbs' uses standard Erbs (1982) with 0.165 clear-sky floor.
        min_diffuse_fraction (float): clear-sky diffuse fraction floor for 'erbs_australia' (default 0.085).
        ground_h_eff (float or None): effective heat transfer coefficient of the surrounding ground,
            W/(m^2 K). Enables the GHI-driven ground temperature model (typ. 25-35 for concrete/paving).
            If None, the surroundings are assumed to be at ambient air temperature.
        ground_emissivity (float): broadband LWIR emissivity of the ground (typ. 0.90-0.97).
            The ground's solar albedo is taken from surrounds_albedo.
        thermal_emissivity (float): broadband thermal infrared emissivity (default 0.95).
            Used to populate the step emissivity model when spectral_absorptivity is None.
        use_iam (bool): whether to apply angle-of-incidence modifiers (IAM) for solar reflection (default True).
        iam_ar (float): angular losses coefficient for the Martin & Ruiz IAM model (default 0.16).
            Lower values (0.10-0.13) model matte/textured/porous surfaces; higher values (0.18-0.25) model glossy/specular surfaces.
        bezel_cutoff_angle (float or None): angle of incidence in degrees beyond which direct beam solar radiation
            is shaded by the mounting frame/bezel lip (default 85.0 deg). Set to None or 90.0 to disable.
        transposition_model (str): sky diffuse transposition model used by pvlib (default 'perez-driesse').
            Options include 'perez-driesse', 'haydavies', 'reindl', 'isotropic', 'klucher', 'king'.
            For equator-opposing (anti-solar) tilted surfaces, 'haydavies' or 'reindl' avoids fictitious horizon brightening.
        use_effective_wind (bool): whether to calculate and use effective wind speed accounting for
            windward vs. leeward wake sheltering behind tilted panels (default False).
        wind_f_min (float): minimum wind speed fraction in a direct leeward wake (default 0.40).
        """
        self._solar_reflectivity = float(solar_reflectivity) if solar_reflectivity is not None else 0.95
        self._thermal_emissivity = float(thermal_emissivity) if thermal_emissivity is not None else 0.95

        # Initialize spectral_absorptivity with automatic synchronization
        if spectral_absorptivity is None:
            self._spectral_absorptivity = emissivity_model(
                LWIRemm=self._thermal_emissivity,
                SWIRemm=1.0 - self._solar_reflectivity
            )
        elif isinstance(spectral_absorptivity, (int, float)):
            self._thermal_emissivity = float(spectral_absorptivity)
            self._spectral_absorptivity = emissivity_model(
                LWIRemm=self._thermal_emissivity,
                SWIRemm=1.0 - self._solar_reflectivity
            )
        else:
            self._spectral_absorptivity = spectral_absorptivity
            # If a model with solar_reflectivity or SWIRemm was passed, sync it with surface's solar_reflectivity
            if hasattr(self._spectral_absorptivity, 'solar_reflectivity'):
                if solar_reflectivity is not None:
                    self._spectral_absorptivity.solar_reflectivity = self._solar_reflectivity
                else:
                    self._solar_reflectivity = self._spectral_absorptivity.solar_reflectivity
            elif hasattr(self._spectral_absorptivity, 'SWIRemm'):
                if solar_reflectivity is not None:
                    self._spectral_absorptivity.SWIRemm = 1.0 - self._solar_reflectivity
                else:
                    self._solar_reflectivity = 1.0 - self._spectral_absorptivity.SWIRemm

        self.tilt = tilt
        self.azimuth = azimuth
        self.name = name
        self.surrounds_albedo = surrounds_albedo
        self.latitude = latitude
        self.longitude = longitude
        self.thickness = thickness
        self.spec_heat_cap = spec_heat_cap
        self.density = density
        self._sky_view_factor = sky_view_factor
        self.convection_model = convection_model
        self.char_length = char_length
        self._back_conductance = back_conductance
        self.insulation_conductivity = insulation_conductivity
        self.insulation_thickness = insulation_thickness
        self.ground_h_eff = ground_h_eff
        self.ground_emissivity = ground_emissivity
        self.use_iam = use_iam
        self.iam_ar = float(iam_ar) if iam_ar is not None else 0.16
        self.bezel_cutoff_angle = float(bezel_cutoff_angle) if bezel_cutoff_angle is not None else None
        self.transposition_model = str(transposition_model) if transposition_model is not None else 'perez-driesse'
        self.use_effective_wind = bool(use_effective_wind)
        self.wind_f_min = float(wind_f_min) if wind_f_min is not None else 0.40
        self.timezone = str(timezone) if timezone is not None else 'Australia/Sydney'
        self.decomposition_model = str(decomposition_model) if decomposition_model is not None else 'erbs_australia'
        self.min_diffuse_fraction = float(min_diffuse_fraction) if min_diffuse_fraction is not None else 0.085

        # Warn if non-zero panel tilt is used with the Perez transposition model
        if abs(float(self.tilt)) > 1e-3 and 'perez' in str(self.transposition_model).lower():
            warnings.warn(
                f"Using transposition_model='{self.transposition_model}' with a non-zero panel tilt ({self.tilt}°). "
                "The Perez model assumes isotropic horizon brightening across all azimuths, which can overestimate "
                "diffuse irradiance on tilted surfaces (especially those facing away from the equator). "
                "Consider using 'haydavies' or 'reindl' instead.",
                UserWarning,
                stacklevel=2
            )
            self._perez_tilt_warned = True

    @property
    def solar_reflectivity(self):
        if hasattr(self._spectral_absorptivity, 'solar_reflectivity'):
            return self._spectral_absorptivity.solar_reflectivity
        return self._solar_reflectivity

    @solar_reflectivity.setter
    def solar_reflectivity(self, R):
        self._solar_reflectivity = float(R)
        if hasattr(self._spectral_absorptivity, 'solar_reflectivity'):
            self._spectral_absorptivity.solar_reflectivity = float(R)
        elif hasattr(self._spectral_absorptivity, 'SWIRemm'):
            self._spectral_absorptivity.SWIRemm = 1.0 - float(R)

    @property
    def solar_absorptivity(self):
        return 1.0 - self.solar_reflectivity

    @solar_absorptivity.setter
    def solar_absorptivity(self, alpha):
        self.solar_reflectivity = 1.0 - float(alpha)

    @property
    def thermal_emissivity(self):
        if hasattr(self._spectral_absorptivity, 'LWIRemm'):
            return self._spectral_absorptivity.LWIRemm
        return self._thermal_emissivity

    @thermal_emissivity.setter
    def thermal_emissivity(self, eps):
        self._thermal_emissivity = float(eps)
        if hasattr(self._spectral_absorptivity, 'LWIRemm'):
            self._spectral_absorptivity.LWIRemm = float(eps)

    @property
    def spectral_absorptivity(self):
        return self._spectral_absorptivity

    @spectral_absorptivity.setter
    def spectral_absorptivity(self, model):
        if isinstance(model, (int, float)):
            self._thermal_emissivity = float(model)
            model = emissivity_model(LWIRemm=self._thermal_emissivity, SWIRemm=1.0 - self._solar_reflectivity)
        self._spectral_absorptivity = model
        if hasattr(model, 'solar_reflectivity'):
            model.solar_reflectivity = self._solar_reflectivity
        elif hasattr(model, 'SWIRemm'):
            model.SWIRemm = 1.0 - self._solar_reflectivity

    @property
    def spectral_adsorptivity(self):
        """Spelling alias for spectral_absorptivity."""
        return self.spectral_absorptivity

    @spectral_adsorptivity.setter
    def spectral_adsorptivity(self, model):
        self.spectral_absorptivity = model

    @property
    def back_conductance(self):
        if self.insulation_conductivity is not None and self.insulation_thickness is not None and self.insulation_thickness > 0:
            return self.insulation_conductivity / self.insulation_thickness
        return self._back_conductance

    @back_conductance.setter
    def back_conductance(self, value):
        self._back_conductance = value

    @property
    def sky_view_factor(self):
        if self._sky_view_factor is not None:
            return self._sky_view_factor
        else:
            # Isotropic view factor from a tilted plane to the sky hemisphere,
            # assuming an unobstructed horizon over flat ground: F_sky = (1 + cos β) / 2
            return (1 + np.cos(np.deg2rad(self.tilt))) / 2

    @property
    def heat_cap_per_m2(self):
        return self.spec_heat_cap*self.thickness*self.density


class SurfaceSimulator(object):
    def __init__(self,
                 surface,
                 prad_function,
                 exp_df,
                 use_measured_strd=False
                 ):
        self.surface = surface
        self.Prad_function = prad_function
        self.Prad_function.surface = surface
        self.exp_df = exp_df
        self.use_measured_strd = use_measured_strd

        self.interp_Tamb = self.create_interpolator('Tamb')
        self.interp_Humid = self.create_interpolator('Humidity')
        self.interp_wspd = self.create_interpolator('Windspeed')
        self.interp_pres = self.create_interpolator('Barometric pressure, kPa')
        self.interp_Psolar = self.create_interpolator('irradiance on tilted panels')
        self.interp_cloud = self.create_interpolator('Cloud cover')
        # GHI is used for the ground-temperature model; optional for backwards compatibility
        self.interp_GHI = self.create_interpolator('GHI') if 'GHI' in self.exp_df.columns else None
        # strd can optionally drive the downwelling atmospheric radiation if present
        self.interp_strd = self.create_interpolator('strd') if 'strd' in self.exp_df.columns else None

        self.heatcap_water = 4184

    def create_interpolator(self, name):
        return interp1d(self.exp_df['Time, s'], self.exp_df[name], kind='linear', bounds_error=False, fill_value=(self.exp_df[name].iloc[0], self.exp_df[name].iloc[-1]))

    def _ghi(self, t):
        return float(self.interp_GHI(t)) if self.interp_GHI is not None else None

    def _derivs(self, t, Y):
        # Physically, condensed mass cannot be negative
        condensed_mass = max(0.0, Y[0])
        Tsurf = Y[1]
        c = Y[2]

        Tamb = self.interp_Tamb(t)
        pressure = self.interp_pres(t)
        windspeed = self.interp_wspd(t)
        humidity = self.interp_Humid(t)
        irrad = self.interp_Psolar(t)

        heat_cap_surf = 1/(self.surface.heat_cap_per_m2+condensed_mass*self.heatcap_water)
        cond_rate = deriv_m(Tamb, humidity, Tsurf, windspeed, pressure, tilt=self.surface.tilt, model=self.surface.convection_model, char_length=self.surface.char_length)
        dmdt = cond_rate-0.01*condensed_mass #I think this is just hacking in an evaporation rate?
        dcdt = 0.01*condensed_mass # c represents cumulative evaporated mass

        # Clamp derivatives if we reach 0 condensed mass and evaporation exceeds condensation
        if condensed_mass == 0.0 and dmdt < 0:
            dmdt = 0.0
            dcdt = 0.0

        strd_val = float(self.interp_strd(t)) if (self.use_measured_strd and self.interp_strd is not None) else None

        dTdt = heat_cap_surf * heat_balance_equation(
            Tsurf_guess=[Tsurf], Tamb=Tamb, irradiance=irrad, humidity=humidity, windspeed=windspeed,
            reflectivity=self.surface.solar_reflectivity, radht=self.Prad_function, cloud_cover=self.interp_cloud(t),
            cond_rate=cond_rate, model=self.surface.convection_model, char_length=self.surface.char_length,
            back_conductance=getattr(self.surface, 'back_conductance', 0.0),
            ghi=self._ghi(t), strd=strd_val
        )
        return [dmdt,dTdt,dcdt]

    def solve(self):
        return solve_ivp(self._derivs,
                          [0,self.exp_df['Time, s'].iloc[-1]],
                          [0,self.interp_Tamb(0), 0])

    def run_simulation(self):
        """
        Runs the solver and interpolates results back onto the experimental time grid,
        returning a populated DataFrame directly compatible with plotting.
        """
        sol = self.solve()
        t_exp = self.exp_df['Time, s'].values

        interp_mass = interp1d(sol.t, sol.y[0], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_Tsurf = interp1d(sol.t, sol.y[1], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_c = interp1d(sol.t, sol.y[2], kind='linear', bounds_error=False, fill_value="extrapolate")

        sim_df = self.exp_df.copy()
        sim_df['Theory - condensed_mass'] = np.maximum(0.0, interp_mass(t_exp))
        sim_df['Theory - Tsurf'] = interp_Tsurf(t_exp)
        sim_df['Theory - c'] = interp_c(t_exp)

        # Calculate condensation rate
        cond_rates = []
        for t, Tsurf in zip(t_exp, sim_df['Theory - Tsurf']):
            Tamb = self.interp_Tamb(t)
            pressure = self.interp_pres(t)
            windspeed = self.interp_wspd(t)
            humidity = self.interp_Humid(t)
            cond_rates.append(deriv_m(Tamb, humidity, Tsurf, windspeed, pressure, tilt=self.surface.tilt, model=self.surface.convection_model, char_length=self.surface.char_length))
        sim_df['Theory - cond rate'] = np.array(cond_rates)

        # Modelled ground temperature (equals Tamb if ground_h_eff is None or GHI unavailable).
        # Uses the same measured strd as the solver when use_measured_strd=True.
        use_strd = self.use_measured_strd and self.interp_strd is not None
        sim_df['Theory - Tground'] = [
            self.Prad_function.ground_temperature(float(self.interp_Tamb(t)), ghi=self._ghi(t),
                                                  humidity=float(self.interp_Humid(t)),
                                                  cloud_cover=float(self.interp_cloud(t)),
                                                  strd=float(self.interp_strd(t)) if use_strd else None)
            for t in t_exp
        ]

        # NOTE: the two downwelling diagnostics below are always the *unscaled* model prediction,
        # even when use_measured_strd=True, so that they can be compared against strd.
        # Modelled downwelling atmospheric thermal radiation (W/m^2) on a horizontal blackbody
        # In-band: integrated over the panel's simulation wavelength range (e.g. 2.5 - 25 um)
        sim_df['Theory - downwelling_sky_flux'] = [
            self.Prad_function.downwelling_sky_radiation(
                float(self.interp_Tamb(t)),
                humidity=float(self.interp_Humid(t)),
                cloud_cover=float(self.interp_cloud(t)),
                broadband=False
            )
            for t in t_exp
        ]
        # Broadband: integrated over the full meteorological thermal IR spectrum (2.5 - 100 um, directly comparable to ERA5 strd)
        sim_df['Theory - strd_broadband'] = [
            self.Prad_function.downwelling_sky_radiation(
                float(self.interp_Tamb(t)),
                humidity=float(self.interp_Humid(t)),
                cloud_cover=float(self.interp_cloud(t)),
                broadband=True
            )
            for t in t_exp
        ]
        sim_df['Theory - strd'] = sim_df['Theory - strd_broadband']

        # If measured or ERA5 strd is present in the dataset, compute diagnostic residuals
        if 'strd' in self.exp_df.columns:
            sim_df['strd'] = self.exp_df['strd'].values
            sim_df['Diagnostic - strd_error_broadband'] = sim_df['Theory - strd_broadband'] - sim_df['strd']
            sim_df['Diagnostic - strd_error_inband'] = sim_df['Theory - downwelling_sky_flux'] - sim_df['strd']

        # Cumulative condensation is the current condensed mass plus cumulative evaporated mass
        sim_df['Theory - cumulative condensation'] = sim_df['Theory - condensed_mass'] + sim_df['Theory - c']

        return sim_df


def erbs_australia(ghi, solar_zenith, times, min_df=0.085, min_cos_zenith=0.065, max_zenith=87.0):
    """
    Continuous Modified Erbs decomposition model adapted for low-turbidity
    Australian atmospheres.

    For overcast / cloudy skies (kt <= 0.22), behaves identically to standard Erbs (1982).
    For clear Australian skies (kt >= 0.80), plateaus at min_df (default 0.085 = 8.5% diffuse,
    compared to 0.165 in standard Erbs).
    For intermediate skies (0.22 < kt < 0.80), a C1-continuous cubic Hermite spline smoothly
    bridges the overcast slope (-0.09) and the clear-sky floor (0.0) with no kinks or discontinuities.
    """
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    kt = pvlib.irradiance.clearness_index(ghi, solar_zenith, dni_extra,
                                          min_cos_zenith=min_cos_zenith,
                                          max_clearness_index=1.0)

    kt_arr = np.asarray(kt, dtype=float)
    df = np.empty_like(kt_arr)

    # 1. Overcast / Cloudy Skies (kt <= 0.22) - identical to Erbs (1982)
    m1 = kt_arr <= 0.22
    df[m1] = 1.0 - 0.09 * kt_arr[m1]

    # 2. Crisp Clear Australian Skies (kt >= 0.80)
    m3 = kt_arr >= 0.80
    df[m3] = min_df

    # 3. Intermediate Skies (0.22 < kt < 0.80) - C1-continuous Hermite spline
    m2 = (~m1) & (~m3)
    if np.any(m2):
        k = kt_arr[m2]
        L = 0.80 - 0.22
        u = (k - 0.22) / L

        y0 = 1.0 - 0.09 * 0.22   # 0.9802
        y1 = min_df
        dy0 = -0.09 * L          # -0.0522
        dy1 = 0.0

        h00 = 2.0 * (u**3) - 3.0 * (u**2) + 1.0
        h10 = (u**3) - 2.0 * (u**2) + u
        h01 = -2.0 * (u**3) + 3.0 * (u**2)

        df[m2] = y0 * h00 + dy0 * h10 + y1 * h01

    dhi = df * ghi
    cos_z = np.cos(np.deg2rad(solar_zenith))
    dni = np.maximum(0.0, (ghi - dhi) / np.maximum(cos_z, 0.05))

    bad_values = (solar_zenith > max_zenith) | (ghi <= 0.0)
    dni = np.where(bad_values, 0.0, dni)
    dhi = np.where(bad_values, ghi, dhi)

    res = pd.DataFrame({'dni': dni, 'dhi': dhi, 'kt': kt}, index=times if hasattr(times, '__iter__') else None)
    return res


_DEFAULT_CUTOFF = object()

def calculate_plane_of_array_irradiance(surface, times, exp_ghi, use_iam=None, iam_ar=None,
                                        bezel_cutoff_angle=_DEFAULT_CUTOFF, transposition_model=None,
                                        decomposition_model=None, min_diffuse_fraction=None):
    """
    Calculates plane-of-array (POA) solar irradiance on a tilted surface using pvlib.

    When use_iam=True (default, or inherited from surface.use_iam), accounts for
    angle-dependent Fresnel reflection using the Martin & Ruiz (M&R) IAM model:
      - Direct beam: pvlib.iam.martin_ruiz(aoi, a_r=a_r)
      - Sky & Ground diffuse: pvlib.iam.martin_ruiz_diffuse(surface.tilt, a_r=a_r)
    The angular loss parameter a_r is taken from iam_ar or surface.iam_ar (default 0.16).

    Direct beam obstruction from sample holder bezels/lips is clamped to zero when
    AOI >= bezel_cutoff_angle (default 85.0 deg, inherited from surface.bezel_cutoff_angle).

    The transposition model for sky diffuse irradiance is taken from transposition_model
    or surface.transposition_model (default 'perez-driesse'; 'haydavies' or 'reindl' recommended
    for surfaces tilted away from the sun).

    The decomposition model for splitting GHI into DNI and DHI is taken from decomposition_model
    or surface.decomposition_model (default 'erbs_australia' with min_diffuse_fraction=0.085;
    set to 'erbs' for standard Erbs (1982)).
    """
    if use_iam is None:
        use_iam = getattr(surface, 'use_iam', True)

    if bezel_cutoff_angle is _DEFAULT_CUTOFF:
        bezel_cutoff_angle = getattr(surface, 'bezel_cutoff_angle', 85.0)

    if transposition_model is None:
        transposition_model = getattr(surface, 'transposition_model', 'perez-driesse')

    if decomposition_model is None:
        decomposition_model = getattr(surface, 'decomposition_model', 'erbs_australia')

    if min_diffuse_fraction is None:
        min_diffuse_fraction = getattr(surface, 'min_diffuse_fraction', 0.085)

    # Warn if non-zero panel tilt is used with the Perez transposition model
    tilt_val = getattr(surface, 'tilt', 0.0)
    if abs(float(tilt_val)) > 1e-3 and 'perez' in str(transposition_model).lower():
        if not getattr(surface, '_perez_tilt_warned', False):
            warnings.warn(
                f"Using transposition_model='{transposition_model}' with a non-zero panel tilt ({tilt_val}°). "
                "The Perez model assumes isotropic horizon brightening across all azimuths, which can overestimate "
                "diffuse irradiance on tilted surfaces (especially those facing away from the equator). "
                "Consider using 'haydavies' or 'reindl' instead.",
                UserWarning,
                stacklevel=2
            )

    # Ensure times are localized if naive, defaulting to surface timezone
    times_orig = times
    times_calc = times
    if isinstance(times, pd.DatetimeIndex) and times.tz is None:
        tz = getattr(surface, 'timezone', 'Australia/Sydney')
        times_calc = times.tz_localize(tz)
    elif hasattr(times, 'dt') and hasattr(times.dt, 'tz') and times.dt.tz is None:
        tz = getattr(surface, 'timezone', 'Australia/Sydney')
        times_calc = times.dt.tz_localize(tz)

    solar_pos = pvlib.solarposition.get_solarposition(time=times_calc, latitude=surface.latitude, longitude=surface.longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times_calc)
    exp_ghi_calc = exp_ghi
    if isinstance(exp_ghi, pd.Series) and isinstance(exp_ghi.index, pd.DatetimeIndex) and exp_ghi.index.tz is None and getattr(times_calc, 'tz', None) is not None:
        exp_ghi_calc = pd.Series(exp_ghi.values, index=times_calc)

    # GHI to DNI / DHI decomposition
    if decomposition_model == 'erbs_australia':
        decomp = erbs_australia(exp_ghi_calc, solar_pos['zenith'], times_calc, min_df=min_diffuse_fraction)
    elif decomposition_model == 'erbs':
        decomp = pvlib.irradiance.erbs(exp_ghi_calc, solar_pos['zenith'], times_calc)
    elif callable(decomposition_model):
        decomp = decomposition_model(exp_ghi_calc, solar_pos['zenith'], times_calc)
    else:
        raise ValueError(f"Unknown decomposition_model: '{decomposition_model}'. Options are 'erbs_australia', 'erbs', or a callable.")

    poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface.tilt,
            surface_azimuth=surface.azimuth, # south
            solar_zenith=solar_pos['apparent_zenith'],
            solar_azimuth=solar_pos['azimuth'],
            dni=decomp['dni'],
            ghi=exp_ghi_calc,
            dhi=decomp['dhi'],
            dni_extra=dni_extra,
            model=transposition_model,
            albedo=surface.surrounds_albedo
            )

    # Time-varying angle of incidence for direct beam
    aoi = pvlib.irradiance.aoi(
        surface_tilt=surface.tilt,
        surface_azimuth=surface.azimuth,
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth']
    )

    if not use_iam:
        if bezel_cutoff_angle is not None and bezel_cutoff_angle < 90.0:
            poa_direct = poa['poa_direct'].copy()
            if isinstance(poa_direct, pd.Series):
                poa_direct = poa_direct.where(aoi < bezel_cutoff_angle, 0.0)
            else:
                poa_direct = np.where(aoi >= bezel_cutoff_angle, 0.0, poa_direct)
            res = poa_direct + poa['poa_diffuse']
        else:
            res = poa['poa_global']
        if isinstance(res, (pd.Series, pd.DataFrame)) and res.index is not times_orig:
            res.index = times_orig
        return res

    # Angular loss parameter for Martin & Ruiz model
    if iam_ar is None:
        iam_ar = getattr(surface, 'iam_ar', 0.16)

    # 1. Exact tilt-dependent diffuse IAMs (Martin & Ruiz)
    iam_sky, iam_ground = pvlib.iam.martin_ruiz_diffuse(surface.tilt, a_r=iam_ar)

    # Direct beam IAM using Martin & Ruiz model
    iam_beam = pvlib.iam.martin_ruiz(aoi, a_r=iam_ar)

    # Bezel shading cut-off (direct beam blocked by mounting frame lip at high AOI)
    if bezel_cutoff_angle is not None and bezel_cutoff_angle < 90.0:
        if isinstance(iam_beam, pd.Series):
            iam_beam = iam_beam.where(aoi < bezel_cutoff_angle, 0.0)
        else:
            iam_beam = np.where(aoi >= bezel_cutoff_angle, 0.0, iam_beam)

    # 3. Angle-corrected effective plane-of-array irradiance
    poa_effective = (
        poa['poa_direct'] * iam_beam +
        poa['poa_sky_diffuse'] * iam_sky +
        poa['poa_ground_diffuse'] * iam_ground
    )
    if isinstance(poa_effective, (pd.Series, pd.DataFrame)) and poa_effective.index is not times_orig:
        poa_effective.index = times_orig
    return poa_effective


class equil_solver(object):
    def __init__(self, surface, radht, Tamb, humidity=50, pressure=101.325, windspeed=0, cloud_cover=0, strd=None):
        self.surface = surface
        self.radht = radht
        self.humidity = humidity
        self.pressure = pressure
        self.windspeed = windspeed
        self.cloud_cover = cloud_cover
        self.Tamb = Tamb
        self.strd = strd

    def T_ghi(self, ghi, reflectivity):
        self.surface.solar_reflectivity = reflectivity
        self.radht.update_surface_cache()

        ghi_basis = np.linspace(0,1300,13)
        Tsurfs = []
        initial_guess = [self.Tamb]

        # Step 2: Use fsolve to find the exact Tc that makes the heat_balance_equation return 0.
        # We pass all the environmental variables through the 'args' tuple.
        for i_ghi in ghi_basis:
            Tc_equilibrium, = fsolve(
                heat_balance_equation,
                x0=initial_guess,
                args=(self.Tamb, i_ghi, self.surface.solar_reflectivity, self.radht, self.humidity, self.windspeed, self.cloud_cover, 0.0, self.surface.convection_model, self.surface.char_length, getattr(self.surface, 'back_conductance', 0.0), i_ghi, self.strd)
            )
            Tsurfs.append(Tc_equilibrium)
        GHI_interp = interp1d(ghi_basis, np.array(Tsurfs), kind='linear')

        return GHI_interp(ghi) - self.Tamb

    def T_ghi_LWIRemm(self, ghi, reflectivity, LWIRemm):
        self.surface.solar_reflectivity = reflectivity
        self.surface.thermal_emissivity = LWIRemm
        self.radht.update_surface_cache()

        ghi_basis = np.linspace(0,1300,13)
        Tsurfs = []
        initial_guess = [self.Tamb]

        # Step 2: Use fsolve to find the exact Tc that makes the heat_balance_equation return 0.
        # We pass all the environmental variables through the 'args' tuple.
        for i_ghi in ghi_basis:
            Tc_equilibrium, = fsolve(
                heat_balance_equation,
                x0=initial_guess,
                args=(self.Tamb, i_ghi, self.surface.solar_reflectivity, self.radht, self.humidity, self.windspeed, self.cloud_cover, 0.0, self.surface.convection_model, self.surface.char_length, getattr(self.surface, 'back_conductance', 0.0), i_ghi, self.strd)
            )
            Tsurfs.append(Tc_equilibrium)
        GHI_interp = interp1d(ghi_basis, np.array(Tsurfs), kind='linear')

        return GHI_interp(ghi) - self.Tamb


def make_dataframe(df_section, surface):
    """
    Somewhat dodgy: the calculations for the solar position get done in here
    """
    new_df = pd.DataFrame(index=df_section.index)
    new_df['Time, s'] = (df_section.index - df_section.index[0]).total_seconds()
    new_df['Barometric pressure, kPa'] = 0.1*df_section[('General', 'Barometric pressure, mbar')]
    new_df['Tsurf'] = df_section[(surface.name, 'Temp avg delta')] + df_section[('General', 'Air temp, ˚C')]
    new_df['Tamb'] = df_section[('General', 'Air temp, ˚C')]
    new_df['Humidity'] = df_section[('General', 'RH, %')]
    # Wind speed and optional wind direction / effective wind speed
    raw_windspeed = df_section[('General', 'Windspeed, m/s')]
    new_df['Windspeed'] = raw_windspeed
    new_df['Windspeed_raw'] = raw_windspeed

    wind_dir_col = None
    for candidate in [
        ('General', 'WindDir, ˚deg'),
        ('General', 'Wind direction, ˚'),
        ('General', 'Wind direction, deg'),
        ('General', 'Wind direction'),
        ('General', 'WindDir'),
        ('General', 'Wind dir, ˚'),
        ('General', 'Wind dir'),
        ('ERA5_Weather', 'Wind direction, ˚'),
        ('ERA5_Weather', 'Wind direction'),
    ]:
        if candidate in df_section.columns:
            wind_dir_col = candidate
            break

    if wind_dir_col is not None:
        new_df['Wind direction'] = df_section[wind_dir_col]
        f_min = getattr(surface, 'wind_f_min', 0.40)
        v_eff = calculate_effective_wind_speed(
            wind_speed=raw_windspeed,
            wind_direction=new_df['Wind direction'],
            surface_azimuth=surface.azimuth,
            tilt=surface.tilt,
            f_min=f_min
        )
        new_df['Effective windspeed, m/s'] = v_eff
        new_df['Windspeed_effective'] = v_eff
        if getattr(surface, 'use_effective_wind', False):
            new_df['Windspeed'] = v_eff

    era5_cloud = df_section[('ERA5_Weather', 'Total cloud cover, fraction')]
    daylight_series = None
    if hasattr(df_section, 'Category') and hasattr(df_section.Category, '__getitem__') and 'Daylight' in df_section.Category:
        daylight_series = df_section.Category['Daylight']
    elif ('Category', 'Daylight') in df_section.columns:
        daylight_series = df_section[('Category', 'Daylight')]

    if daylight_series is not None and ('General', 'Cloud loss, fraction') in df_section.columns:
        daylight_bool = daylight_series.fillna(False).astype(bool)
        loss_cloud = df_section[('General', 'Cloud loss, fraction')]
        combined_cloud = np.where(daylight_bool & loss_cloud.notnull(), loss_cloud, era5_cloud)
        new_df['Cloud cover'] = pd.Series(combined_cloud, index=df_section.index, dtype=float)
    else:
        new_df['Cloud cover'] = era5_cloud.astype(float)
    new_df['GHI'] = df_section[('General', 'Solar flux density, W/m^2')]
    new_df['irradiance on tilted panels'] = calculate_plane_of_array_irradiance(surface=surface,
                                                                                 times=df_section.index,
                                                                                 bezel_cutoff_angle=surface.bezel_cutoff_angle,
                                                                                 exp_ghi=df_section[('General', 'Solar flux density, W/m^2')])
    
    dew_col = (surface.name, 'Dew collected, mL/m^2')
    if dew_col in df_section.columns:
        new_df['Dew collected, mL/m2'] = df_section[dew_col]
    else:
        new_df['Dew collected, mL/m2'] = 0.0

    strd_col = ('ERA5_Thermal', 'Surface long-wave (thermal) radiation downwards, W/m^2')
    if strd_col not in df_section.columns:
        strd_col = None
        for candidate in [
            ('ERA5_Weather', 'Surface long-wave (thermal) radiation downwards, W/m^2'),
            ('General', 'Surface long-wave (thermal) radiation downwards, W/m^2'),
            ('ERA5_Weather', 'Surface thermal radiation downwards, W/m^2'),
            ('ERA5_Weather', 'strd'),
            ('General', 'strd'),
        ]:
            if candidate in df_section.columns:
                strd_col = candidate
                break

    if strd_col is not None and strd_col in df_section.columns:
        # strd is assumed to already be a mean flux in W/m^2
        new_df['strd'] = df_section[strd_col]

    return new_df

## -----------
def calculate_effective_wind_speed(wind_speed, wind_direction, surface_azimuth=180.0, tilt=0.0, f_min=0.40):
    """
    Calculates the effective local wind speed over an inclined surface, accounting for
    windward impingement vs. leeward wake separation (sheltering behind the tilted panel).

    Parameters:
    -----------
    wind_speed : float, np.ndarray, or pd.Series
        Free-stream wind speed in m/s.
    wind_direction : float, np.ndarray, or pd.Series
        Meteorological wind direction in degrees from North (0-360 deg: 0 = North, 90 = East, 180 = South, 270 = West).
    surface_azimuth : float, default 180.0
        Surface facing direction (degrees from North, 180 = South).
    tilt : float, default 0.0
        Surface tilt angle in degrees from horizontal.
    f_min : float, default 0.40
        Minimum wind fraction in a direct leeward wake (default 0.40, based on Loveday & Taki / DOE-2).

    Returns:
    --------
    v_eff : float, np.ndarray, or pd.Series
        Effective wind speed in m/s.
    """
    if wind_direction is None:
        return wind_speed

    # Relative angle between wind approach vector and surface azimuth in radians
    delta_gamma = np.radians((wind_direction - surface_azimuth + 180.0) % 360.0 - 180.0)

    # Windward alignment factor: 1.0 (head-on windward) down to 0.0 (direct leeward from behind)
    u = (1.0 + np.cos(delta_gamma)) / 2.0

    # Scale wake depth with panel tilt: horizontal panels (tilt=0) have no bluff-body wake;
    # flow over an inclined plate reaches full aerodynamic wake separation by tilt >= 20 deg
    tilt_deg = np.clip(tilt, 0.0, 90.0)
    tilt_ramp = np.clip(tilt_deg / 20.0, 0.0, 1.0)
    f_min_eff = 1.0 - (1.0 - f_min) * tilt_ramp

    reduction_factor = f_min_eff + (1.0 - f_min_eff) * u
    v_eff = wind_speed * reduction_factor

    if isinstance(v_eff, pd.Series):
        return v_eff.combine_first(wind_speed)
    elif isinstance(v_eff, np.ndarray):
        nan_mask = np.isnan(v_eff)
        if np.any(nan_mask):
            v_eff = v_eff.copy()
            v_eff[nan_mask] = np.asarray(wind_speed)[nan_mask]
        return v_eff
    elif np.isnan(v_eff):
        return wind_speed
    return v_eff

def get_tarp_natural_convection(T_surface, T_ambient, tilt, char_length=None):
    """
    Computes natural convection heat transfer coefficient.
    If char_length is provided, uses size-dependent McAdams/ASHRAE correlations.
    Otherwise, defaults to the size-independent George N. Walton (1983) model.
    T_surface and T_ambient are in Celsius, tilt is in degrees, char_length in meters.
    """
    delta_T = T_surface - T_ambient
    abs_delta_T = abs(delta_T)
    if abs_delta_T < 1e-5:
        return 0.0
    
    theta = np.deg2rad(tilt)
    cos_theta = np.cos(theta)
    abs_cos_theta = abs(cos_theta)
    
    is_upward = (tilt <= 90.0)
    is_heating = (delta_T > 0.0)
    is_unstable = (is_upward and is_heating) or (not is_upward and not is_heating)
    
    if char_length is not None and char_length > 0.0:
        # Size-dependent natural convection based on Rayleigh number
        # Ra_L = g * beta * delta_T * L^3 / (nu * alpha)
        # Approximate scaling factor for air at standard temperature: 1.6e8
        Ra_L = 1.6e8 * abs_delta_T * (char_length ** 3)
        
        # Horizontal plate component (characteristic length L_c = As / P = char_length)
        if is_unstable:
            if Ra_L < 1e7:
                h_nat_horiz = 1.32 * ((abs_delta_T / char_length) ** 0.25)
            else:
                h_nat_horiz = 1.52 * (abs_delta_T ** (1/3))
        else:
            h_nat_horiz = 0.59 * ((abs_delta_T / char_length) ** 0.25)
            
        # Vertical plate component (characteristic length L_vert = 4 * As / P = 4 * char_length)
        L_vert = 4.0 * char_length
        Ra_L_vert = 1.6e8 * abs_delta_T * (L_vert ** 3)
        if Ra_L_vert < 1e9:
            h_nat_vert = 1.42 * ((abs_delta_T / L_vert) ** 0.25)
        else:
            h_nat_vert = 1.31 * (abs_delta_T ** (1/3))
            
        # Interpolate based on tilt angle
        sin_theta = np.sin(theta)
        h_nat = h_nat_horiz * abs_cos_theta + h_nat_vert * abs(sin_theta)
    else:
        # Size-independent Walton model
        if is_unstable:
            h_nat = 9.482 * (abs_delta_T ** (1/3)) / (7.283 - abs_cos_theta)
        else:
            h_nat = 1.810 * (abs_delta_T ** (1/3)) / (1.382 + abs_cos_theta)
        
    return h_nat

def conv_heat_transfer_coeff(T_surface, windspeed, T_ambient=None, tilt=0.0, model='Maestre-Valero', char_length=None):
    """
    Calculates convective heat transfer coefficient.
    T_surface: Surface temperature in Celsius
    windspeed: Wind speed in m/s
    T_ambient: Ambient air temperature in Celsius (optional, required for Combined-TARP models)
    tilt: Panel tilt angle in degrees (optional, required for Combined-TARP models)
    char_length: Characteristic length L = As/P in meters (optional, triggers size-dependent scaling)
    """
    if model == 'Maestre-Valero':
        T_ref = T_ambient if T_ambient is not None else T_surface
        return 7.6 + 6.6 * windspeed * (511 + 294) / (511 + T_ref + 273.15)
    elif model == 'Richards':
        T_ref = T_ambient if T_ambient is not None else T_surface
        return 5.9 + 4.1 * windspeed * (511 + 294) / (511 + T_ref + 273.15)
    elif model == 'Sharples':
        return 8 + 2.5 * windspeed
        
    # TARP/Combined models (combination of natural + forced component)
    if T_ambient is None:
        h_nat = 0.0
    else:
        h_nat = get_tarp_natural_convection(T_surface, T_ambient, tilt, char_length=char_length)
        
    # Scale forced components by (L_ref / L)^0.5
    # The reference length L_ref corresponds to the characteristic length L = As/P of the panels in the base study.
    # Maestre-Valero / Richards: 1.0 m square condenser foils -> L_ref = As/P = 0.25 m
    # Sharples: 1.8 m x 0.9 m solar collector -> L_ref = As/P = 0.30 m
    if model == 'Combined-TARP-Maestre':
        L_ref = 0.25
        forced_scale = (L_ref / char_length) ** 0.5 if (char_length is not None and char_length > 0.0) else 1.0
        forced_term = 6.6 * windspeed * (511 + 294) / (511 + T_surface + 273.15)
        return h_nat + forced_term * forced_scale
    elif model == 'Combined-TARP-Richards':
        L_ref = 0.25
        forced_scale = (L_ref / char_length) ** 0.5 if (char_length is not None and char_length > 0.0) else 1.0
        forced_term = 4.1 * windspeed * (511 + 294) / (511 + T_surface + 273.15)
        return h_nat + forced_term * forced_scale
    elif model == 'Combined-TARP-Sharples':
        # Sharples (1998) used a 1.8m x 0.9m flat plate solar collector.
        # For a 1.8m x 0.9m panel: L_ref = As/P = (1.8 * 0.9) / (2 * (1.8 + 0.9)) = 0.3 m.
        L_ref = 0.3
        forced_scale = (L_ref / char_length) ** 0.5 if (char_length is not None and char_length > 0.0) else 1.0
        forced_term = 2.5 * windspeed
        return h_nat + forced_term * forced_scale
    else:
        raise ValueError(f'Unknown model: {model}')

def Pconvection(T_amb, T_surf, windspeed, tilt=0.0, model='Maestre-Valero', char_length=None):
    hc = conv_heat_transfer_coeff(T_surface=T_surf, windspeed=windspeed, T_ambient=T_amb, tilt=tilt, model=model, char_length=char_length)
    return hc * (T_amb - T_surf)

def Pconduction(T_amb, T_surf, back_conductance=0.0):
    """
    Calculates conductive heat transfer flux with ambient / boundary in W/m^2.
    back_conductance is in W/(m^2 K).
    Positive flux corresponds to heat entering the surface from ambient.
    """
    return back_conductance * (T_amb - T_surf)

def deriv_m(T_amb, H, T_surf, windspeed, p, tilt=0.0, model='Maestre-Valero', char_length=None):
    Ca = 1005

    Tdew = Tdewpoint(T_amb, H)
    if T_surf <= Tdew:
        hc = conv_heat_transfer_coeff(T_surface=T_surf, windspeed=windspeed, T_ambient=T_amb, tilt=tilt, model=model, char_length=char_length)
        val = ((hc * 0.622) / (Ca * p)) * (VPsat(Tdew) - VPsat(T_surf))
    else:
        val = 0
    return val

def VPsat(T_amb):
    """
    T_amb (float): Temperature in degrees celsius
    returns (float): vapor pressure in kPa
    """
    val = 0.61121 * np.exp((18.678 - T_amb / 234.5) * (T_amb / (257.14 + T_amb)))
    #val = saturation_vapor_pressure(T_amb*units.degC).to('kPa').magnitude # Super slow...

    return val

def Tdewpoint(T_amb, humidity):
    """
    T_amb (float): Temperature in degrees celsius
    humidity (float): Humidity in percent
    """
    return np.power(humidity/100,1/8)*(112+0.9*T_amb)+0.1*T_amb-112

def planck_law(wavelength, T_kelvin):
    """
    Wavelength in meters
    Temperature in Kelvin
    Calculates the spectral radiant exitance of a blackbody.
    Returns power per unit area per unit wavelength (W/m^3).
    """
    numerator = 3.746914726083474e-16 # 2 * np.pi * H_CONST * (C_CONST ** 2)
    HxCoKb = 0.014404347826086955 # H_CONST * C_CONST / Kb_CONST

    exponent = HxCoKb / ( wavelength * T_kelvin)
    exponent = np.clip(exponent, 0.1, 450) # Works for both scalars and arrays
    denominator = (wavelength ** 5) * (np.exp(exponent) - 1.0)

    return numerator / denominator

class emissivity_model(object):
    """
    Two-band step emissivity model:
    - lambda < cutoff_wavelength (default 3.0 µm): SWIRemm (solar absorptivity = 1 - solar_reflectivity)
    - lambda >= cutoff_wavelength (default 3.0 µm): LWIRemm (thermal infrared emissivity, default 0.95)
    """
    def __init__(self, LWIRemm=0.95, SWIRemm=0.02, model=None, cutoff_wavelength=3.0e-6):
        self.LWIRemm = LWIRemm
        self.SWIRemm = SWIRemm
        self.cutoff_wavelength = cutoff_wavelength
        self.model = 'step' if (model is None or model == 'Tristram' or model == 'step') else model

    @property
    def solar_reflectivity(self):
        return 1.0 - self.SWIRemm

    @solar_reflectivity.setter
    def solar_reflectivity(self, R):
        self.SWIRemm = 1.0 - R

    def tristram_emissivity(self, wavelength):
        """Legacy Tristram square-root transition model (retained for backward compatibility)."""
        wav = np.atleast_1d(wavelength)
        emis = np.full_like(wav, np.nan, dtype=float)

        visIR_mask = np.logical_and(wav >= 0.1e-6, wav <= 2.5e-6)
        midIR_mask = np.logical_and(wav > 2.5e-6, wav <= 8e-6)
        LWIR_mask = wav > 8e-6

        emis[visIR_mask] = self.SWIRemm
        emis[midIR_mask] = self.LWIRemm * np.sqrt((wav[midIR_mask] - 2.5e-6) / (8e-6 - 2.5e-6))
        emis[LWIR_mask] = self.LWIRemm

        if emis.size == 1:
            return float(emis[0])
        else:
            return emis

    def step_emissivity(self, wavelength):
        wav = np.asarray(wavelength)
        emis = np.where(wav < self.cutoff_wavelength, self.SWIRemm, self.LWIRemm)
        if wav.ndim == 0:
            return float(emis)
        return emis

    def __call__(self, wavelength):
        if self.model == 'Tristram_legacy':
            return self.tristram_emissivity(wavelength)
        else:
            return self.step_emissivity(wavelength)

def spectral_humidCorr_tristram (wavelength, humidity):
    """
    Legacy empirical atmospheric humidity correction factor used by Tristram.
    (Preserved from main git branch for backward compatibility / baseline comparisons).

    Parameters:
    -----------
    wavelength : float or array-like
        Wavelength in meters.
    humidity : float
        Relative humidity in % (0 - 100).

    Returns:
    --------
    float or ndarray:
        Emissivity correction factor:
          - 0.1 - 2.5 µm: 0.0
          - 2.5 - 13.0 µm: 0.005 * humidity
          - > 13.0 µm: 1.0
    """
    wav = np.atleast_1d(wavelength)
    h_corr = np.full_like(wav, np.nan, dtype=float)

    visIR_mask = np.logical_and(wav >= 0.1e-6, wav <= 2.5e-6)
    midIR_mask = np.logical_and(wav > 2.5e-6, wav <= 8e-6)
    LWIR_mask = np.logical_and(wav > 8e-6, wav <= 13e-6)
    # Wavelengths > 13 µm act as a blackbody in the atmosphere due to H2O/CO2 absorption
    farIR_mask = wav > 13e-6

    h_corr[visIR_mask] = 0
    h_corr[midIR_mask] = 0.005 * humidity
    h_corr[LWIR_mask] = 0.005 * humidity
    h_corr[farIR_mask] = 1.0

    if h_corr.size == 1:
        return float(h_corr[0])
    else:
        return h_corr

# Aliases for backward compatibility
tristram_spectral_humidCorr = spectral_humidCorr_tristram
spectral_humidCorr_legacy = spectral_humidCorr_tristram

def spectral_humidCorr (wavelength, humidity, Tamb=None, model=None):
    """
    Physical spectral atmospheric emissivity model for clear skies.
    (Replaces the empirical correction factor used by Tristram: 0.005*humidity in 2.5-13 um, 1.0 above 13 um.)
    
    Parameterized by actual water vapor partial pressure (e in hPa) and ambient temperature (Tamb),
    incorporating realistic atmospheric absorption features matching libRadtran / MODTRAN spectroscopy:
      - < 2.5 µm: Solar transparent window (atmospheric emissivity ~ 0.0)
      - 2.5 - 3.0 µm: Opaque 2.7 µm H2O/CO2 absorption band (eps = 1.0)
      - 3.0 - 3.8 µm: Smooth transition wing from 2.7 µm band down to plateau
      - 3.8 - 7.8 µm: Saturated moisture-dependent plateau (eps_plat)
      - 7.8 - 8.3 µm: Smooth transition wing rolling off into the primary window
      - 8.3 - 12.5 µm: Primary atmospheric transparency window (water vapor continuum + 9.6 µm ozone)
      - 12.5 - 13.5 µm: Smooth transition wing connecting into the 15 µm CO2 band
      - 13.5 - 16.5 µm: Opaque CO2 15 µm band (eps = 1.0)
      - > 16.5 µm: Rotational water vapor bands
    
    Parameters:
    -----------
    wavelength : float or array-like
        Wavelength in meters.
    humidity : float
        Relative humidity in % (0 - 100).
    Tamb : float, optional
        Ambient air temperature in degrees Celsius. Defaults to 20.0 °C if not provided
        for backwards compatibility with 2-argument calls.
    model : str, optional
        Model to evaluate. Defaults to the calibrated physical model (Option B).
        If 'Tristram', 'Tristram_legacy', or 'legacy', evaluates the legacy empirical Tristram model.
    """
    if model in ('Tristram', 'Tristram_legacy', 'legacy', 'tristram'):
        return spectral_humidCorr_tristram(wavelength, humidity)

    wav = np.atleast_1d(wavelength)
    w_um = wav * 1e6
    h_corr = np.zeros_like(wav, dtype=float)

    if Tamb is None:
        Tamb = 20.0

    # Actual water vapor partial pressure in hPa
    e_kpa = (humidity / 100.0) * VPsat(Tamb)
    e_hpa = np.maximum(0.001, e_kpa * 10.0)

    # Base levels for primary atmospheric window (8.3 - 12.5 µm)
    eps_cont = 1.0 - np.exp(-0.035 * e_hpa - 0.0006 * (e_hpa**2))
    eps_win = 0.045 + (1.0 - 0.045) * eps_cont

    # Moisture-dependent saturated plateau height for short wavelengths (3.8 - 7.8 µm).
    # In dry air, optical depth in the lower troposphere is reduced and downward emission
    # originates from colder layers aloft (effective emissivity ~0.50-0.60 referenced to surface Tamb);
    # in humid air, near-surface layer saturates (effective emissivity ~0.85-0.90).
    eps_plat = 0.35 + (0.88 - 0.35) * (1.0 - np.exp(-0.10 * e_hpa))

    def _s(x):
        # Clamped cubic smoothstep
        xc = np.clip(x, 0.0, 1.0)
        return xc * xc * (3.0 - 2.0 * xc)

    # 0. < 2.50 µm: Solar transparent window (high transmission, negligible thermal emission)
    m0 = w_um < 2.50
    h_corr[m0] = 0.0

    # 1. 2.50 - 3.00 µm: Opaque H2O/CO2 2.7 µm band (saturated absorption, eps = 1.0)
    m1 = (w_um >= 2.50) & (w_um <= 3.00)
    h_corr[m1] = 1.0

    # 2. 3.00 - 3.80 µm: Smooth transition wing from 2.7 µm band down to plateau
    m2 = (w_um > 3.00) & (w_um < 3.80)
    s2 = _s((3.80 - w_um[m2]) / (3.80 - 3.00))
    h_corr[m2] = eps_plat + (1.0 - eps_plat) * s2

    # 3. 3.80 - 7.80 µm: Continuous saturated plateau (libRadtran structure)
    m3 = (w_um >= 3.80) & (w_um <= 7.80)
    h_corr[m3] = eps_plat

    # 4. 7.80 - 8.30 µm: Smooth roll-off from plateau into primary atmospheric window
    m4 = (w_um > 7.80) & (w_um < 8.30)
    s4 = _s((8.30 - w_um[m4]) / (8.30 - 7.80))
    h_corr[m4] = eps_win + (eps_plat - eps_win) * s4

    # 5. 8.30 - 12.50 µm: Primary atmospheric window with smooth Ozone 9.6 µm peak
    # Effective stratospheric ozone emission at ~215-225 K yields an effective downward emissivity of ~0.30
    m5 = (w_um >= 8.30) & (w_um <= 12.50)
    o3_peak = 0.30 * np.exp(-((w_um[m5] - 9.6) / 0.35)**2)
    h_corr[m5] = np.maximum(eps_win, o3_peak)

    # 6. 12.50 - 13.50 µm: Smooth transition wing connecting from window level up to CO2 15 µm band (1.0)
    m6 = (w_um > 12.50) & (w_um < 13.50)
    s6 = _s((w_um[m6] - 12.50) / (13.50 - 12.50))
    h_corr[m6] = eps_win + (1.0 - eps_win) * s6

    # 7. 13.50 - 16.50 µm: Opaque CO2 15 µm band
    m7 = (w_um >= 13.50) & (w_um <= 16.50)
    h_corr[m7] = 1.0

    # 8. > 16.50 µm: Rotational water vapor bands with moisture-dependent optical depth
    m8 = w_um > 16.50
    h_corr[m8] = 0.42 + (1.0 - 0.42) * (1.0 - np.exp(-0.20 * e_hpa))

    h_corr = np.clip(h_corr, 0.0, 1.0)

    if h_corr.size == 1:
        return float(h_corr[0])
    else:
        return h_corr

class ThermalHeatTransfer (object):
    def __init__ (self, low_wavelength, high_wavelength, num_wavelengths,
                  surface=None,
                  humidity_model=None,
                  cloud_temperature_delTamb=-8.0): # Offset between cloud-base temp and surface ambient (default -8.0 K matches tropospheric lapse and ERA5 overcast radiation)
        self.low_wavelength = low_wavelength
        self.high_wavelength = high_wavelength
        self.num_wavelengths = num_wavelengths
        self.cloud_temperature_delTamb = cloud_temperature_delTamb

        self._surface = surface

        self._humidity_model_takes_Tamb = None
        self._humidity_model_checked = None
        if humidity_model is None:
            self.spectral_humidity_model = lambda wavelength, humidity: np.zeros_like(np.asarray(wavelength), dtype=float)
        else:
            self.spectral_humidity_model = humidity_model

        self.wavelengths = np.linspace(self.low_wavelength, self.high_wavelength, num_wavelengths)

        if self._surface is not None:
            self.absorptivities = self.surface.spectral_absorptivity(self.wavelengths)
            self.sky_view_factor = self._surface.sky_view_factor
        else:
            self.absorptivities = None
            self.sky_view_factor = 1

    @property
    def surface (self):
        return self._surface

    @surface.setter
    def surface(self, surface):
        self._surface = surface
        self.absorptivities = self._surface.spectral_absorptivity(self.wavelengths)
        self.sky_view_factor = self._surface.sky_view_factor
        print('Set spectral_absorptivity & sky_view_factor from surface')

    def update_surface_cache(self):
        """
        Manually syncs the thermal model with the surface.
        Call this immediately after modifying parameters of the assigned surface.
        """
        if self._surface is not None:
            self.absorptivities = self._surface.spectral_absorptivity(self.wavelengths)
            self.sky_view_factor = self._surface.sky_view_factor
        else:
            self.absorptivities = None
            self.sky_view_factor = 1

    def _eval_humidity_model(self, wavs, humidity, T_amb):
        """
        Evaluates the assigned spectral humidity model, passing T_amb only if the model accepts it
        (a 'Tamb' keyword, or a third positional parameter). Models with a 2-argument signature
        are called as model(wavs, humidity) for backwards compatibility. Errors raised inside the
        model are propagated rather than retried.
        """
        if self._humidity_model_takes_Tamb is None or self._humidity_model_checked is not self.spectral_humidity_model:
            self._humidity_model_checked = self.spectral_humidity_model
            self._humidity_model_takes_Tamb = self._accepts_Tamb(self.spectral_humidity_model)
        if self._humidity_model_takes_Tamb:
            return self.spectral_humidity_model(wavs, humidity, Tamb=T_amb)
        return self.spectral_humidity_model(wavs, humidity)

    @staticmethod
    def _accepts_Tamb(model):
        try:
            params = inspect.signature(model).parameters
        except (TypeError, ValueError):
            return False  # e.g. builtins/ufuncs such as np.zeros_like
        if 'Tamb' in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def cloud_temperature(self, tamb):
        """
        tamb (float): Temperature in degrees celsius
        returns cloud temperature in degrees celsius
        """
        return tamb + self.cloud_temperature_delTamb

    def P_outgoing(self, T_surf):
        spectrum = planck_law(self.wavelengths, T_surf + 273.15) * self.absorptivities
        return trapezoid(y=spectrum, x=self.wavelengths)

    def sky_spectrum(self, T_amb, humidity=0, cloud_cover=0, strd=None):
        """
        Downwelling spectral sky radiance (W/m^3) incident on a horizontal blackbody,
        i.e. NOT weighted by the panel's spectral absorptivity.

        If strd (W/m^2) is provided, the downwelling spectrum is scaled to match
        the measured/ERA5 broadband downwelling flux.
        """
        bb_amb = planck_law(self.wavelengths, T_amb + 273.15)
        humidity_spectrum = bb_amb * self._eval_humidity_model(self.wavelengths, humidity, T_amb)
        if cloud_cover > 0.05:
            cloud_spectrum = planck_law(self.wavelengths, self.cloud_temperature(T_amb) + 273.15)
            raw_sky = cloud_spectrum * cloud_cover + (1.0 - cloud_cover) * humidity_spectrum
        else:
            raw_sky = humidity_spectrum

        scale = self._strd_scale(T_amb, humidity, cloud_cover, strd)
        return raw_sky * scale

    def _strd_scale(self, T_amb, humidity, cloud_cover, strd):
        """
        Factor scaling the modelled broadband downwelling flux to the measured/ERA5 strd (W/m^2).
        Returns 1.0 if strd is None, non-positive or NaN. The last result is cached, because
        sky_spectrum is called several times with identical arguments per derivative evaluation
        (panel sky term and ground temperature), and the broadband integral is comparatively costly.
        """
        if strd is None or not (strd > 0):
            return 1.0
        key = (float(T_amb), float(humidity), float(cloud_cover), float(strd),
               self.cloud_temperature_delTamb, id(self.spectral_humidity_model))
        cached = getattr(self, '_strd_scale_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        broadband_flux = self.downwelling_sky_radiation(T_amb, humidity=humidity, cloud_cover=cloud_cover, broadband=True)
        scale = strd / broadband_flux if broadband_flux > 0 else 1.0
        self._strd_scale_cache = (key, scale)
        return scale

    def ground_temperature(self, T_amb, ghi=None, humidity=0, cloud_cover=0, strd=None):
        """
        Quasi-steady, linearised energy balance of horizontal ground (Celsius):

            T_g = T_amb + [(1 - albedo) * GHI - eps_g * (L_bb(T_amb) - L_sky)] / h_eff

        albedo  : surface.surrounds_albedo (same value used for ground-reflected solar)
        eps_g   : surface.ground_emissivity
        h_eff   : surface.ground_h_eff, W/(m^2 K); lumps convection, linearised radiation
                  and conduction into the ground. If None, T_g = T_amb (legacy behaviour).
        L_bb, L_sky are integrated over the same wavelength range as the panel model.
        """
        surf = self._surface
        h_eff = getattr(surf, 'ground_h_eff', None) if surf is not None else None
        if ghi is None or h_eff is None or h_eff <= 0:
            return T_amb

        albedo = getattr(surf, 'surrounds_albedo', 0.15)
        eps_g = getattr(surf, 'ground_emissivity', 0.95)

        L_bb_amb = trapezoid(y=planck_law(self.wavelengths, T_amb + 273.15), x=self.wavelengths)
        L_sky = trapezoid(y=self.sky_spectrum(T_amb, humidity=humidity, cloud_cover=cloud_cover, strd=strd), x=self.wavelengths)

        net_lw_loss = eps_g * (L_bb_amb - L_sky)
        return T_amb + ((1 - albedo) * max(ghi, 0.0) - net_lw_loss) / h_eff

    def downwelling_sky_radiation(self, T_amb, humidity=0, cloud_cover=0, broadband=True):
        """
        Calculates downwelling longwave radiation from the sky on a horizontal blackbody (W/m^2).
        
        Parameters:
        -----------
        T_amb : float
            Ambient temperature in Celsius.
        humidity : float
            Relative humidity (%).
        cloud_cover : float
            Cloud cover fraction (0.0 to 1.0).
        broadband : bool, default True
            If True, integrates across the full meteorological thermal infrared spectrum (2.5 - 100 um),
            directly corresponding to ERA5 'strd' or a broadband pyrgeometer.
            If False, integrates across self.wavelengths (the panel's simulation band).
        """
        if not broadband:
            spectrum = self.sky_spectrum(T_amb, humidity=humidity, cloud_cover=cloud_cover)
            return trapezoid(y=spectrum, x=self.wavelengths)
        else:
            if not hasattr(self, '_broadband_wavs'):
                self._broadband_wavs = np.linspace(2.5e-6, 100.0e-6, 300)
            wavs = self._broadband_wavs
            bb_amb = planck_law(wavs, T_amb + 273.15)
            h_spec = bb_amb * self._eval_humidity_model(wavs, humidity, T_amb)
            if cloud_cover > 0.05:
                cloud_spec = planck_law(wavs, self.cloud_temperature(T_amb) + 273.15)
                full_sky = cloud_spec * cloud_cover + (1.0 - cloud_cover) * h_spec
            else:
                full_sky = h_spec
            return trapezoid(y=full_sky, x=wavs)

    def effective_sky_emissivity(self, T_amb, humidity=0, cloud_cover=0, broadband=True):
        """
        Calculates effective broadband sky emissivity: eps_sky = L_sky / (sigma * T_amb^4).
        """
        sigma = 5.670374e-8
        T_k = T_amb + 273.15
        L_sky = self.downwelling_sky_radiation(T_amb, humidity=humidity, cloud_cover=cloud_cover, broadband=broadband)
        return L_sky / (sigma * (T_k ** 4))

    def P_incoming(self, T_amb, humidity=0, cloud_cover=0, ghi=None, strd=None):
        assert self.sky_view_factor is not None, "Sky view factor must be set"
        # Surroundings (ground) term: blackbody at the modelled ground temperature
        T_ground = self.ground_temperature(T_amb, ghi=ghi, humidity=humidity, cloud_cover=cloud_cover, strd=strd)
        spectrum1 = planck_law(self.wavelengths, T_ground + 273.15) * self.absorptivities
        integral_1 = trapezoid(y=spectrum1, x=self.wavelengths)

        # Sky term. Clearsky, incident radiation is due to humidity
        spectrum2 = self.sky_spectrum(T_amb, humidity=humidity, cloud_cover=cloud_cover, strd=strd) * self.absorptivities
        integral_2 = trapezoid(y=spectrum2, x=self.wavelengths)

        return (1-self.sky_view_factor) * integral_1 + (self.sky_view_factor) * integral_2

    def __call__(self, T_amb, T_surf, humidity=0, cloud_cover=0, ghi=None, strd=None):
        return self.P_incoming(T_amb, humidity=humidity, cloud_cover=cloud_cover, ghi=ghi, strd=strd) - self.P_outgoing(T_surf)



def heat_balance_equation(Tsurf_guess, Tamb, irradiance, reflectivity, radht, humidity, windspeed, cloud_cover, cond_rate=0, model='Maestre-Valero', char_length=None, back_conductance=None, ghi=None, strd=None):
    """
    The objective function that we want to equal zero.
    Calculates the net heat flux for a given guess of surface temperature (Tc_guess).
    ghi: global horizontal irradiance (W/m^2), used to estimate the ground temperature.
         If None, the ground is assumed to be at ambient temperature.
    strd: downwelling longwave radiation (W/m^2) from ERA5 or pyrgeometer.
          If provided, scales the incident sky spectrum directly.
    """
    Lv = 2256*1e3 # latent heat of vapourisation

    # Calculate individual heat fluxes
    P_rad = radht(T_surf=Tsurf_guess[0], T_amb=Tamb, humidity=humidity, cloud_cover=cloud_cover, ghi=ghi, strd=strd)
    P_sol = (1 - reflectivity) * irradiance
    tilt = radht.surface.tilt if (hasattr(radht, 'surface') and radht.surface is not None) else 0.0
    P_conv = Pconvection(Tamb, Tsurf_guess[0], windspeed, tilt=tilt, model=model, char_length=char_length)
    P_lat = Lv * cond_rate # 0, unless condensation rate provided

    # Conductive heat flux through insulation with ambient air (T_boundary = T_amb)
    if back_conductance is None:
        if hasattr(radht, 'surface') and radht.surface is not None:
            back_conductance = getattr(radht.surface, 'back_conductance', 0.0)
        else:
            back_conductance = 0.0
    P_cond = Pconduction(Tamb, Tsurf_guess[0], back_conductance=back_conductance)

    # print (f'LWIR out: {P_out:.1f}, LWIR in: {P_inc:.1f}, solar: {P_sol:.1f}')
    # Calculate net flux (Outgoing minus all incoming/generated heat)
    # At equilibrium, this should equal 0.
    net_flux = P_rad + P_sol + P_conv + P_lat + P_cond

    return net_flux

def plot_simulation(df, relT=False):
    fig, [ax1, ax2] = plt.subplots(2,1, sharex=True)
    ax1t = ax1.twinx()
    ax1.set_zorder(ax1t.get_zorder() + 1)
    ax1.patch.set_visible(False)

    if relT == True:
        offset = df['Tamb']
        ybounds = [-7,7]
    else:
        offset = 0
        ybounds = [6,24]

    ax1t.fill_between(df.index, df['GHI'], color='xkcd:sun yellow', alpha=0.5, edgecolor=None)
    ax1.plot(df.index, df['Tamb']-offset, color='k', lw=1, ls=':')
    ax1.plot(df.index, df['Tsurf']-offset, color='k')
    ax1.plot(df.index, df['Theory - Tsurf']-offset, color='k', ls='--')

    # Plot experimental cumulative dew collected in L/m2
    ax2.plot(df.index, np.cumsum(df['Dew collected, mL/m2'])/1000, color='k', label='Experiment')
    # Plot theoretical cumulative condensation in L/m2 (which matches the unit of experimental cumsum)
    if 'Theory - cumulative condensation' in df.columns:
        ax2.plot(df.index, df['Theory - cumulative condensation'], color='k', ls='--', label='Theory')
    else:
        # Fallback if the user only integrated cond rate
        ax2.plot(df.index, df['Theory - cond rate'], color='k', ls='--', label='Theory (Rate)')
    ax2.legend()

    ax1.set(xlim=[df.index[1], df.index[-2]], ylim=ybounds, ylabel='Temperature, C')
    ax1t.set(ylim=[0,999], ylabel='Solar radiation [W/m2]')
    ax2.set(ylabel='Dew collected, L/m2')


def diagnose_strd(df, radht=None, plot=True, title='Downwelling Longwave Radiation (strd) Diagnostic'):
    """
    Diagnostic tool comparing modelled downwelling atmospheric thermal radiation against measured/ERA5 'strd'.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from make_dataframe or run_simulation.
        Must contain 'Tamb', 'Humidity', 'Cloud cover'.
        If 'strd' is present, full statistical comparisons are generated.
    radht : ThermalHeatTransfer, optional
        Used to compute 'Theory - strd_broadband' and 'Theory - downwelling_sky_flux'
        if they are not already present in df.
    plot : bool, default True
        Whether to generate and return diagnostic plots.
    title : str, default 'Downwelling Longwave Radiation (strd) Diagnostic'
        Title for the diagnostic plot.
        
    Returns:
    --------
    stats : dict
        Statistical error metrics and performance breakdown.
    fig : matplotlib.figure.Figure (optional, only if plot=True)
    """
    df = df.copy()

    # If theory columns are missing, compute them using radht if available
    if 'Theory - strd_broadband' not in df.columns or 'Theory - downwelling_sky_flux' not in df.columns:
        if radht is None:
            raise ValueError(
                "Neither 'Theory - strd_broadband' is in df nor was a 'radht' (ThermalHeatTransfer) instance provided. "
                "Please run run_simulation() first or pass radht=..."
            )
        df['Theory - downwelling_sky_flux'] = [
            radht.downwelling_sky_radiation(row['Tamb'], humidity=row['Humidity'], cloud_cover=row['Cloud cover'], broadband=False)
            for _, row in df.iterrows()
        ]
        df['Theory - strd_broadband'] = [
            radht.downwelling_sky_radiation(row['Tamb'], humidity=row['Humidity'], cloud_cover=row['Cloud cover'], broadband=True)
            for _, row in df.iterrows()
        ]

    has_measured = 'strd' in df.columns and not df['strd'].dropna().empty

    stats = {
        'n_samples': len(df),
        'mean_theory_broadband': float(df['Theory - strd_broadband'].mean()),
        'mean_theory_inband': float(df['Theory - downwelling_sky_flux'].mean()),
        'has_measured_strd': has_measured
    }

    if has_measured:
        valid_mask = df['strd'].notnull() & df['Theory - strd_broadband'].notnull()
        valid_df = df[valid_mask]
        
        diff_broadband = valid_df['Theory - strd_broadband'] - valid_df['strd']
        diff_inband = valid_df['Theory - downwelling_sky_flux'] - valid_df['strd']

        stats['mean_measured_strd'] = float(valid_df['strd'].mean())
        stats['mbe_broadband'] = float(diff_broadband.mean())
        stats['rmse_broadband'] = float(np.sqrt((diff_broadband ** 2).mean()))
        stats['mbe_inband'] = float(diff_inband.mean())
        stats['rmse_inband'] = float(np.sqrt((diff_inband ** 2).mean()))

        # Clear-sky subset (cloud <= 0.1) -> Isolates atmospheric water vapor & humidity model
        clear_mask = valid_df['Cloud cover'] <= 0.1
        if clear_mask.any():
            stats['n_clear_sky'] = int(clear_mask.sum())
            stats['clear_sky_mean_measured'] = float(valid_df.loc[clear_mask, 'strd'].mean())
            stats['clear_sky_mean_model'] = float(valid_df.loc[clear_mask, 'Theory - strd_broadband'].mean())
            stats['clear_sky_mbe'] = float(diff_broadband[clear_mask].mean())
            stats['clear_sky_rmse'] = float(np.sqrt((diff_broadband[clear_mask] ** 2).mean()))

        # Overcast subset (cloud >= 0.8) -> Isolates cloud temperature & emissivity model
        overcast_mask = valid_df['Cloud cover'] >= 0.8
        if overcast_mask.any():
            stats['n_overcast'] = int(overcast_mask.sum())
            stats['overcast_mean_measured'] = float(valid_df.loc[overcast_mask, 'strd'].mean())
            stats['overcast_mean_model'] = float(valid_df.loc[overcast_mask, 'Theory - strd_broadband'].mean())
            stats['overcast_mbe'] = float(diff_broadband[overcast_mask].mean())
            stats['overcast_rmse'] = float(np.sqrt((diff_broadband[overcast_mask] ** 2).mean()))

    # Print summary report
    print("=" * 65)
    print("         DOWNWELLING LONGWAVE (strd) DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f"Total samples: {stats['n_samples']}")
    print(f"Modelled downwelling (broadband, 2.5-100 um): {stats['mean_theory_broadband']:.1f} W/m^2")
    print(f"Modelled downwelling (in-band):              {stats['mean_theory_inband']:.1f} W/m^2")

    if has_measured:
        print(f"Measured/ERA5 strd downwelling:             {stats['mean_measured_strd']:.1f} W/m^2")
        print(f"Overall Broadband Bias (MBE):               {stats['mbe_broadband']:+.1f} W/m^2")
        print(f"Overall Broadband RMSE:                     {stats['rmse_broadband']:.1f} W/m^2")
        print("-" * 65)
        print("PHYSICAL BREAKDOWN:")
        if 'n_clear_sky' in stats:
            print(f"* Clear-sky (Cloud <= 0.1, N={stats['n_clear_sky']}):")
            print(f"  - ERA5: {stats['clear_sky_mean_measured']:.1f} W/m^2 | Model: {stats['clear_sky_mean_model']:.1f} W/m^2")
            print(f"  - MBE: {stats['clear_sky_mbe']:+.1f} W/m^2 | RMSE: {stats['clear_sky_rmse']:.1f} W/m^2")
            if stats['clear_sky_mbe'] < -20:
                print("  -> Underestimating clear sky longwave (air is more opaque than modeled).")
            elif stats['clear_sky_mbe'] > 20:
                print("  -> Overestimating clear sky longwave (air is more transparent than modeled).")
        if 'n_overcast' in stats:
            print(f"* Overcast (Cloud >= 0.8, N={stats['n_overcast']}):")
            print(f"  - ERA5: {stats['overcast_mean_measured']:.1f} W/m^2 | Model: {stats['overcast_mean_model']:.1f} W/m^2")
            print(f"  - MBE: {stats['overcast_mbe']:+.1f} W/m^2 | RMSE: {stats['overcast_rmse']:.1f} W/m^2")
    else:
        print("\nNote: 'strd' column not detected in DataFrame.")
        print("Once you download ERA5 'strd' and load it into your DataFrame, rerun")
        print("diagnose_strd(df) to see full bias, RMSE, clear-sky vs overcast breakdown.")
    print("=" * 65)

    if not plot:
        return stats

    # Diagnostic Plots
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_time = fig.add_subplot(gs[0, :])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_resid = fig.add_subplot(gs[1, 1])

    # 1. Time series plot
    if has_measured:
        ax_time.plot(df.index, df['strd'], label='ERA5 / Measured strd', color='tab:blue', lw=2)
    ax_time.plot(df.index, df['Theory - strd_broadband'], label='Model (Broadband)', color='tab:orange', lw=1.8, ls='--')
    ax_time.plot(df.index, df['Theory - downwelling_sky_flux'], label='Model (In-band)', color='tab:green', lw=1.2, ls=':')
    ax_time.set_ylabel('Downwelling Flux [W/m²]')
    ax_time.set_title(title, fontweight='bold', fontsize=12)

    ax_cloud = ax_time.twinx()
    ax_cloud.fill_between(df.index, df['Cloud cover'], alpha=0.15, color='gray', label='Cloud cover')
    ax_cloud.set_ylabel('Cloud Cover [0-1]', color='gray')
    ax_cloud.set_ylim([0, 1.2])

    lines_1, labels_1 = ax_time.get_legend_handles_labels()
    lines_2, labels_2 = ax_cloud.get_legend_handles_labels()
    ax_time.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.85)

    if has_measured:
        # 2. Scatter plot (Model vs Measured)
        sc = ax_scatter.scatter(valid_df['strd'], valid_df['Theory - strd_broadband'],
                                c=valid_df['Humidity'], cmap='viridis', alpha=0.75, edgecolors='none', s=25)
        cbar = plt.colorbar(sc, ax=ax_scatter)
        cbar.set_label('RH [%]')

        min_val = min(valid_df['strd'].min(), valid_df['Theory - strd_broadband'].min()) - 15
        max_val = max(valid_df['strd'].max(), valid_df['Theory - strd_broadband'].max()) + 15
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.2, label='1:1 Line')
        ax_scatter.set_xlim([min_val, max_val])
        ax_scatter.set_ylim([min_val, max_val])
        ax_scatter.set_xlabel('ERA5 / Measured strd [W/m²]')
        ax_scatter.set_ylabel('Modelled Broadband strd [W/m²]')
        ax_scatter.set_title(f'1:1 Comparison (RMSE={stats["rmse_broadband"]:.1f} W/m²)')
        ax_scatter.legend(loc='upper left')

        # 3. Residuals vs Relative Humidity
        diff = valid_df['Theory - strd_broadband'] - valid_df['strd']
        cloud_colors = valid_df['Cloud cover'].fillna(0.0) if 'Cloud cover' in valid_df.columns else np.zeros(len(valid_df))
        sc_resid = ax_resid.scatter(valid_df['Humidity'], diff, c=cloud_colors, cmap='coolwarm', alpha=0.75, s=25, vmin=0.0, vmax=1.0)
        cbar_resid = plt.colorbar(sc_resid, ax=ax_resid)
        cbar_resid.set_label('Cloud cover [0-1]')
        ax_resid.axhline(0, color='k', ls='--', lw=1.2)
        
        # Fit trendline against humidity if enough points
        mask_fit = valid_df['Humidity'].notnull() & diff.notnull()
        if mask_fit.sum() > 5 and valid_df.loc[mask_fit, 'Humidity'].nunique() > 2:
            p = np.polyfit(valid_df.loc[mask_fit, 'Humidity'], diff[mask_fit], 1)
            rh_line = np.linspace(valid_df.loc[mask_fit, 'Humidity'].min(), valid_df.loc[mask_fit, 'Humidity'].max(), 50)
            ax_resid.plot(rh_line, np.polyval(p, rh_line), color='crimson', lw=1.5, label=f'Trend ({p[0]:+.2f} W/m² per %RH)')
            ax_resid.legend(loc='best')

        ax_resid.set_xlabel('Relative Humidity [%]')
        ax_resid.set_ylabel('Error: Model - Measured [W/m²]')
        ax_resid.set_title('Residuals vs Humidity (Color = Cloud Cover)')
    else:
        ax_scatter.text(0.5, 0.5, "No 'strd' data available to plot scatter.", ha='center', va='center')
        ax_resid.text(0.5, 0.5, "No 'strd' data available to plot residuals.", ha='center', va='center')

    plt.tight_layout()
    return stats, fig